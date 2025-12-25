import tensorflow as tf
import cv2
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "ocular_disease_model.keras"
IMAGE_PATH = "1012_right.jpg"
IMG_SIZE = (224, 224)

# ⚠️ IMPORTANT:
# This order MUST match the training notebook exactly
CLASSES = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other"
]


# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# LOAD & PREPROCESS IMAGE
# -----------------------------
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise ValueError("❌ Failed to read image. Check file format.")

# Convert BGR → RGB (VERY IMPORTANT)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img = cv2.resize(img, IMG_SIZE)

# Normalize (must match training)
img = img.astype("float32") / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)

# -----------------------------
# PREDICTION
# -----------------------------
pred = model.predict(img)

print("\n📊 Disease Prediction Results:\n")

for disease, probability in zip(CLASSES, pred[0]):
    print(f"{disease:<15}: {probability:.2f}")

# -----------------------------
# FINAL DECISION (optional)
# -----------------------------
THRESHOLD = 0.5

print("\n🩺 Detected Conditions (prob > 0.5):")
found = False
for disease, probability in zip(CLASSES, pred[0]):
    if probability >= THRESHOLD:
        print(f"➡️ {disease} ({probability:.2f})")
        found = True

if not found:
    print("➡️ No disease detected above threshold")
