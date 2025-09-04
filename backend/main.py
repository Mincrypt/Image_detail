# backend/main.py
import os
import io
import base64
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
import pytesseract

import torch
from transformers import pipeline
import torchvision.transforms as T
from torchvision import models as tv_models

app = FastAPI(title="Multimodal Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory conversation history (simple)
CONVERSATIONS: List[Dict[str, Any]] = []

# Minimal model container
class Models:
    sentiment = None
    summarizer = None
    zero_shot = None
    toxicity = None
    imagenet_model = None
    imagenet_idx2label = None

models = Models()

# Image transforms
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

TOXIC_KEYWORDS = {"hate","idiot","stupid","ugly","trash","kill","screw","bastard","damn","fuc","asshole","bitch","slut"}

# Request model for JSON base64 input
class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None

# -----------------------
# Lazy model loaders
# -----------------------
def ensure_nlp_models():
    """Load NLP models lazily and choose smaller variants to save disk space."""
    if models.sentiment is None:
        try:
            models.sentiment = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        except Exception:
            models.sentiment = None

    if models.summarizer is None:
        try:
            # small-ish summarizer
            models.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-xsum-1-1",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception:
            models.summarizer = None

    if models.zero_shot is None:
        try:
            # smaller MNLI-based distilbart variant
            models.zero_shot = pipeline(
                "zero-shot-classification",
                model="valhalla/distilbart-mnli-12-1",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception:
            models.zero_shot = None

    if models.toxicity is None:
        try:
            models.toxicity = pipeline(
                "text-classification",
                model="unitary/unbiased-toxic-roberta",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception:
            models.toxicity = None

def ensure_imagenet():
    """Load a lightweight torchvision model and imagenet labels (fallback to placeholders)."""
    if models.imagenet_model is None:
        try:
            m = tv_models.mobilenet_v2(pretrained=True)
        except Exception:
            m = tv_models.resnet18(pretrained=True)
        m.eval()
        models.imagenet_model = m

        try:
            here = os.path.dirname(__file__)
            labels_file = os.path.join(here, "imagenet_classes.txt")
            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as fh:
                    classes = [c.strip() for c in fh.readlines()]
            else:
                classes = [f"class_{i}" for i in range(1000)]
        except Exception:
            classes = [f"label_{i}" for i in range(1000)]
        models.imagenet_idx2label = classes

# -----------------------
# Helpers
# -----------------------
def read_imagefile_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def image_classify_coarse(pil_img: Image.Image) -> Dict[str, Any]:
    ensure_imagenet()
    model = models.imagenet_model
    if model is None:
        return {"coarse_category": "unknown", "top_labels": []}
    tensor = img_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        topk = torch.topk(probs, k=5)
        labels = []
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
            label = models.imagenet_idx2label[idx] if models.imagenet_idx2label else f"idx_{idx}"
            labels.append({"label": label, "score": float(score)})
    # coarse mapping heuristics
    text_labels = " ".join([l["label"].lower() for l in labels])
    if any(w in text_labels for w in ["person", "man", "woman", "boy", "girl", "bride", "groom", "soldier"]):
        coarse = "person"
    elif any(w in text_labels for w in ["room", "restaurant", "shop", "indoor", "stadium", "kitchen", "bathroom"]):
        coarse = "scene"
    else:
        coarse = "object"
    return {"coarse_category": coarse, "top_labels": labels}

def detect_toxic_text_score(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    ensure_nlp_models()
    if models.toxicity is not None:
        try:
            results = models.toxicity(text)
            max_score = 0.0
            for r in results:
                label = r.get("label", "").lower()
                score = r.get("score", 0.0)
                if any(k in label for k in ["toxic", "insult", "threat", "abusive", "hate"]):
                    max_score = max(max_score, float(score))
            return float(max_score)
        except Exception:
            pass
    # fallback keyword heuristic
    text_low = text.lower()
    count = sum(1 for w in TOXIC_KEYWORDS if w in text_low)
    return min(1.0, count / 5.0)

# -----------------------
# Main endpoint
# -----------------------
@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    json_payload: Optional[AnalyzeRequest] = None,
    debug: Optional[int] = Query(0, description="set debug=1 to return full raw analysis")
):
    """
    Accepts:
      - multipart/form-data with 'text' and 'file'
      - OR JSON {"text": "...", "image_base64": "..."}
    Returns: by default a clean summary JSON (Sentiment, Topic, Image, OCR, Toxicity, Response).
    If ?debug=1 is set, returns the full raw analysis including nlp and cv sections.
    """
    # read image bytes
    img_bytes = None
    if file is not None:
        try:
            img_bytes = await file.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read uploaded file.")
    elif json_payload and json_payload.image_base64:
        try:
            img_bytes = base64.b64decode(json_payload.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image.")

    input_text = (text or (json_payload.text if json_payload else "") or "").strip()
    if not input_text and img_bytes is None:
        raise HTTPException(status_code=400, detail="Empty request: provide text or image.")

    # ---------------- NLP ----------------
    ensure_nlp_models()
    nlp_results: Dict[str, Any] = {}

    if input_text:
        # Sentiment
        try:
            if models.sentiment:
                sent = models.sentiment(input_text[:1000])
                label = sent[0].get("label", "Unknown")
                score = float(sent[0].get("score", 0.0))
                nlp_results["text_sentiment"] = {"label": label.capitalize(), "score": score}
            else:
                nlp_results["text_sentiment"] = {"label": "Unknown", "score": 0.0}
        except Exception:
            nlp_results["text_sentiment"] = {"label": "Unknown", "score": 0.0}

        # Summarization (if long)
        try:
            if models.summarizer and len(input_text.split()) > 20:
                summary = models.summarizer(input_text, max_length=60, min_length=15, do_sample=False)[0]["summary_text"]
            else:
                summary = input_text if len(input_text.split()) <= 30 else " ".join(input_text.split()[:30]) + "..."
            nlp_results["text_summary"] = summary
        except Exception:
            nlp_results["text_summary"] = input_text

        # Zero-shot topic classification
        try:
            candidate_labels = ["news", "review", "comment", "complaint", "question", "social", "product feedback"]
            if models.zero_shot:
                zero = models.zero_shot(input_text, candidate_labels)
                nlp_results["topic_classification"] = {"best_label": zero["labels"][0], "scores": list(zip(zero["labels"], [float(s) for s in zero["scores"]]))}
            else:
                nlp_results["topic_classification"] = {"best_label": "other", "scores": []}
        except Exception:
            nlp_results["topic_classification"] = {"best_label": "other", "scores": []}

        # Toxicity detection in text
        try:
            tox_score = detect_toxic_text_score(input_text)
            nlp_results["text_toxicity_score"] = tox_score
            nlp_results["text_toxicity_flag"] = tox_score >= 0.5
        except Exception:
            nlp_results["text_toxicity_score"] = 0.0
            nlp_results["text_toxicity_flag"] = False
    else:
        nlp_results = {
            "text_sentiment": {"label": "Neutral", "score": 0.0},
            "text_summary": "",
            "topic_classification": {"best_label": "none", "scores": []},
            "text_toxicity_score": 0.0,
            "text_toxicity_flag": False
        }

    # ---------------- CV ----------------
    cv_results: Dict[str, Any] = {}
    if img_bytes:
        try:
            pil = read_imagefile_to_pil(img_bytes)
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        # OCR
        try:
            ocr_text = pytesseract.image_to_string(pil).strip()
            cv_results["ocr_text"] = ocr_text
            cv_results["ocr_present"] = bool(ocr_text)
            ocr_tox = detect_toxic_text_score(ocr_text) if ocr_text else 0.0
            cv_results["ocr_toxicity_score"] = ocr_tox
            cv_results["ocr_toxicity_flag"] = ocr_tox >= 0.5
        except Exception:
            cv_results["ocr_text"] = ""
            cv_results["ocr_present"] = False
            cv_results["ocr_toxicity_score"] = 0.0
            cv_results["ocr_toxicity_flag"] = False

        # Image classification coarse
        try:
            cls = image_classify_coarse(pil)
            cv_results["image_classification"] = cls
        except Exception:
            cv_results["image_classification"] = {"coarse_category": "unknown", "top_labels": []}

        cv_results["face_emotion_detected"] = None  # optional extension
    else:
        cv_results = {
            "ocr_text": "",
            "ocr_present": False,
            "ocr_toxicity_score": 0.0,
            "ocr_toxicity_flag": False,
            "image_classification": {"coarse_category": "unknown", "top_labels": []},
            "face_emotion_detected": None
        }

    # ---------------- Fusion logic (multimodal response) ----------------
    auto_response = ""
    text_sentiment_label = nlp_results.get("text_sentiment", {}).get("label", "Neutral").lower()
    image_coarse = cv_results.get("image_classification", {}).get("coarse_category", "")

    # Toxicity rule wins
    if nlp_results.get("text_toxicity_flag") or cv_results.get("ocr_toxicity_flag"):
        auto_response = "Warning: Abusive/toxic content detected. Please be respectful. This submission may be flagged."
    else:
        # Negative + angry face (face_emotion_detected not implemented -> heuristic fallback)
        found_angry = False
        if cv_results.get("ocr_text") and any(w in cv_results["ocr_text"].lower() for w in ["angry","hate","mad","furious"]):
            found_angry = True
        if text_sentiment_label == "negative" and found_angry:
            auto_response = "We're sorry that you're upset. We hear you — if you'd like, please share more details so we can help."
        elif "i love" in (input_text or "").lower() and image_coarse == "object":
            auto_response = "Thanks for the positive feedback! We're happy you like the product."
        else:
            if text_sentiment_label == "positive":
                auto_response = "Thanks for the positive message! Glad you liked it."
            elif text_sentiment_label == "negative":
                auto_response = "Sorry to hear that. We appreciate the feedback and will work on it."
            else:
                auto_response = "Thanks for sharing. Here's what we found from your submission."

    # ---------------- Build neat summary (optimized output) ----------------
    summary_view: Dict[str, Any] = {}

    # Sentiment
    summary_view["Sentiment"] = nlp_results.get("text_sentiment", {}).get("label", "Unknown")

    # Topic mapping
    topic = nlp_results.get("topic_classification", {}).get("best_label", "other")
    topic_map = {
        "review": "Food/Restaurant Review",
        "complaint": "Complaint",
        "comment": "Comment",
        "question": "Question",
        "news": "News",
        "social": "Social",
        "product feedback": "Product Feedback"
    }
    summary_view["Topic"] = topic_map.get(topic, topic.capitalize())

    # Image category humanized
    coarse = cv_results.get("image_classification", {}).get("coarse_category", "unknown")
    if coarse == "scene":
        img_label = "Restaurant/Indoor"
    elif coarse == "person":
        img_label = "Person"
    elif coarse == "object":
        img_label = "Object"
    else:
        img_label = "Unknown"
    summary_view["Image"] = img_label

    # OCR
    ocr_text = cv_results.get("ocr_text", "")
    summary_view["OCR"] = f"“{ocr_text}”" if ocr_text else "None"

    # Toxicity percent
    tox_score = max(nlp_results.get("text_toxicity_score", 0.0), cv_results.get("ocr_toxicity_score", 0.0))
    summary_view["Toxicity"] = f"{int(tox_score * 100)}%"

    # Response (short label)
    if "sorry" in auto_response.lower() or text_sentiment_label == "negative":
        summary_view["Response"] = "Apology + reassurance"
    elif "warning" in auto_response.lower():
        summary_view["Response"] = "Warning"
    else:
        summary_view["Response"] = "Polite Acknowledgment"

    # Save interaction (store summary only to prevent huge logs)
    CONVERSATIONS.append({
        "text": input_text,
        "has_image": bool(img_bytes),
        "summary": summary_view
    })

    # If debug=1 return full raw analysis, else return only summary_view
    if debug:
        full = {
            "nlp": nlp_results,
            "cv": cv_results,
            "automated_response": auto_response,
            "summary_view": summary_view
        }
        return JSONResponse(content=full)
    else:
        return JSONResponse(content=summary_view)
