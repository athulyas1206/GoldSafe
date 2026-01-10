from flask import Flask, render_template, request, send_file
import os
from ml.similarity import find_top_k_similar

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files.get("image")

    if not image:
        return "No image uploaded"

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    results = find_top_k_similar(image_path)

    return render_template(
    "results.html",
    query_image=image_path,   # KEEP FULL PATH
    results=results
)

# 🔥 SERVE DATASET IMAGES FROM DRIVE
@app.route("/dataset_image")
def dataset_image():
    path = request.args.get("path")

    if not path or not os.path.exists(path):
        return "Image not found", 404

    return send_file(path)

if __name__ == "__main__":
    app.run(debug=True)
