from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Helper function to check file type
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Image upload + category
@app.route("/upload", methods=["POST"])
def upload():
    category = request.form.get("category")
    image = request.files.get("image")

    if not category or not image:
        return "Category or Image missing"

    if not allowed_file(image.filename):
        return "File type not allowed. Please upload an image."

    # Here: image is in memory, can be passed to ML model
    # e.g., image.read() or PIL.Image.open(image)
    
    # For now, just return confirmation
    return f"Image received for category: {category}"


# Run server
if __name__ == "__main__":
    app.run(debug=True)
