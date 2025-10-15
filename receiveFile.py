from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # allow React to call backend

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")  # multiple files support
    results = []

    for file in files:
        try:
            img = Image.open(file.stream)
            width, height = img.size
            results.append({
                "filename": file.filename,
                "width": width,
                "height": height
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({"processed_files": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
