import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from predict import predict_action

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    video_filename = None
    video_ext = None
    warning = None
    error = None

    if request.method == "POST":
        if "video" not in request.files:
            error = "No file part found."
            return render_template("index.html", error=error)

        file = request.files["video"]

        if file.filename == "":
            error = "Please select a video file."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            prediction, confidence = predict_action(file_path)
            video_filename = filename
            video_ext = filename.rsplit(".", 1)[1].lower()

            if video_ext != "mp4":
                warning = "Prediction is done. Browser playback works best with MP4. Some AVI or MOV files may not play in the page."

        else:
            error = "Only .mp4, .avi, and .mov files are allowed."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        video_filename=video_filename,
        video_ext=video_ext,
        warning=warning,
        error=error
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)