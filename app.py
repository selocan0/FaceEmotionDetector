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
                    result = DeepFace.analyze(
                        img_path=filepath,
                        actions=["emotion", "age", "gender", "race"],
                        detector_backend="retinaface",  # More reliable
                        enforce_detection=False         # Avoid crash on no face
                    )
                except Exception as e:
                    error = f"❌ Analysis failed: {str(e)}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
