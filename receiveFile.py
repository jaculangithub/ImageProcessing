from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from umlClassdiagram import process_class_diagram
from umlStateDiagram import process_state_diagram
from umlSequencediagram import process_sequence_diagram
from umlActivitydiagram import process_activity_diagram

app = Flask(__name__)
CORS(app)  # allow React to call backend

@app.route("/upload", methods=["POST"])
def upload():
    # Get the diagram type from form data
    diagram_type = request.form.get("diagramType", "")
    
    # Get uploaded files
    files = request.files.getlist("files")  # multiple files support
    results = []

    for file in files:
        try:
            # Process based on diagram type
            if diagram_type == "class":
                # Process class diagram
                result = process_class_diagram(file)
                results.append({
                    "filename": file.filename,
                    "diagram_type": diagram_type,
                    "processed_data": result,
                    "message": f"Successfully processed {diagram_type} diagram"
                })
                
            elif diagram_type == "state":
                result = process_state_diagram(file)
                results.append({
                    "filename": file.filename,
                    "diagram_type": diagram_type,
                    "processed_data": result,
                    "message": f"Successfully processed {diagram_type} diagram"
                })
                
            elif diagram_type == "state":
                result = process_sequence_diagram(file)
                results.append({
                    "filename": file.filename,
                    "diagram_type": diagram_type,
                    "processed_data": result,
                    "message": f"Successfully processed {diagram_type} diagram"
                })
            
            elif diagram_type == "sequence":
                result = process_state_diagram(file)
                results.append({
                    "filename": file.filename,
                    "diagram_type": diagram_type,
                    "processed_data": result,
                    "message": f"Successfully processed {diagram_type} diagram"
                })
            elif diagram_type == "activity":
                result = process_activity_diagram(file)
                results.append({
                    "filename": file.filename,
                    "diagram_type": diagram_type,
                    "processed_data": result,
                    "message": f"Successfully processed {diagram_type} diagram"
                })
            else:
                # For other diagram types, return basic info
                img = Image.open(file.stream)
                width, height = img.size
                
                results.append({
                    "filename": file.filename,
                    "width": width,
                    "height": height,
                    "diagram_type": diagram_type,
                    "message": f"Basic processing for {diagram_type} diagram (full processing not implemented)"
                })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "diagram_type": diagram_type
            })

    return jsonify({
        "processed_files": results,
        "diagram_type_received": diagram_type
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)