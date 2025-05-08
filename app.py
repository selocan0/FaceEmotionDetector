import sys
sys.path.insert(0, './patched_deepface')  # 👈 makes sure we use your custom DeepFace

from flask import Flask, render_template, request
from deepface import DeepFace
import os

app = Flask(__name__)

# Create upload directory
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        image = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        try:
            # Perform facial attribute analysis
            analysis = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            # Extract relevant attributes
            result = {
                "emotion": analysis[0]["dominant_emotion"],
                "age": analysis[0]["age"],
                "gender": analysis[0]["dominant_gender"],
                "race": analysis[0]["dominant_race"]
            }

        except Exception as e:
            result = {"error": str(e)}

        return render_template("index.html", result=result, image_path=image_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
