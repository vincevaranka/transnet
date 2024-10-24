import os
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from document_index import save_document_to_elasticsearch, search_documents

# Create Flask application
app = Flask(__name__)

# Path for storing uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to handle image upload and process it
@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Save the file to the uploads directory
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Perform OCR on the image
    extracted_text = extract_text_from_image(file_path)
    
    # Index the extracted text into Elasticsearch
    save_document_to_elasticsearch(filename=file.filename, content=extracted_text)
    
    return jsonify({'message': 'Document processed', 'extracted_text': extracted_text})

# Route to handle searching for documents by keyword
@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({'error': 'No keyword provided'}), 400

    # Search for documents in Elasticsearch
    results = search_documents(keyword)

    # Format results into JSON
    documents = [{'filename': hit.filename, 'content': hit.content} for hit in results]
    
    return jsonify({'documents': documents})

# Function to extract text from an image using Pytesseract
def extract_text_from_image(image_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(image)
    
    return extracted_text

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
