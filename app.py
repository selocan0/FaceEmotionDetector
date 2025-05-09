import os
from flask import Flask, render_template, request
from deepface import DeepFace

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part in request"
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "No file selected"
            else:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                try:
                    analysis = DeepFace.analyze(
                        img_path=filepath,
                        actions=["emotion"],
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    result = {
                        "emotion": analysis[0]["dominant_emotion"]
                    }

                except Exception as e:
                    error = f"❌ Analysis failed: {str(e)}"

    return render_template("index.html", result=result, error=error)

# ✅ Render-compatible run block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
