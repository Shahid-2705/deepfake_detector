import sys
import os

# --------------------------------------------------
# Add project root to path
# --------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

from insightface.app import FaceAnalysis


# ================= CONFIG =================
MODEL_PATH = "models/face_deepfake_b3_best.pth"
IMAGE_PATH = "image.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================


# --------------------------------------------------
# Risk Level Mapping
# --------------------------------------------------
def get_risk_level(prob):
    if prob < 0.20:
        return "Very Low"
    elif prob < 0.40:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    elif prob < 0.80:
        return "High"
    else:
        return "Critical"


# --------------------------------------------------
# Load EfficientNet-B3 (MUST match training)
# --------------------------------------------------
def load_model():
    model = models.efficientnet_b3(weights=None)

    # Replace classifier (same as training)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Initialize InsightFace Detector
# --------------------------------------------------
def init_face_detector():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    return app


# --------------------------------------------------
# Preprocessing (EfficientNet-B3 Resolution)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Detect and Crop Largest Face
# --------------------------------------------------
def detect_and_crop(image_path, detector):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image path or unreadable file.")

    faces = detector.get(img)

    if len(faces) == 0:
        return None

    # Select largest detected face
    face = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
    )

    x1, y1, x2, y2 = map(int, face.bbox)

    # Clamp values to image boundaries
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face_crop)


# --------------------------------------------------
# Inference
# --------------------------------------------------
def run_inference(image_path):
    model = load_model()
    detector = init_face_detector()

    face_image = detect_and_crop(image_path, detector)

    if face_image is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "face_detected": False,
            "message": "No face detected. Deepfake analysis requires visible human face."
        }

    input_tensor = transform(face_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        fake_prob = torch.sigmoid(logits).item()

    risk_level = get_risk_level(fake_prob)

    result = {
        "timestamp": datetime.now().isoformat(),
        "face_detected": True,
        "ai_risk_score": round(fake_prob, 4),
        "risk_level": risk_level,
        "verdict": "High AI Risk" if fake_prob > 0.6 else "Low AI Risk",
        "confidence_note": "Probability represents model confidence that the face contains synthetic manipulation artifacts."
    }

    return result


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    output = run_inference(IMAGE_PATH)

    print("\n=== PRODUCTION AI DEEPFAKE FORENSIC REPORT ===\n")
    print(json.dumps(output, indent=4))
