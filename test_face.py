import os
import sys
import cv2
from deepface import DeepFace

# Ensure patched deepface is prioritized
sys.path.insert(0, "./patched_deepface")

# Force CPU (avoid Metal crashes on macOS)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

# Load test image
img_path = os.path.expanduser("~/Desktop/test_face.jpg")
img = cv2.imread(img_path)

if img is None:
    print("❌ Failed to load image.")
    sys.exit(1)

print(f"✅ Image loaded: {img.shape}")

# Run analysis
print("\n🔍 Running analysis...")

try:
    analysis = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion", "age", "gender", "race"],
        detector_backend="retinaface",
        enforce_detection=True
    )
    print("\n✅ Analysis complete:")
    print(analysis)
except Exception as e:
    print("\n❌ Analysis failed:")
    print(e)
