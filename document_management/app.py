import os
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from document_index import save_document_to_elasticsearch, search_documents

# Create Flask application
app = Flask(__name__)

# Path for storing uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to handle image or PDF upload and process it
@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if the file is a supported format
    if not (file.filename.endswith(('.png', '.jpg', '.jpeg', '.pdf'))):
        return jsonify({'error': 'Unsupported file type. Please upload a PNG, JPG, JPEG, or PDF file.'}), 400

    # Save the file to the uploads directory
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform OCR on the image or PDF
    extracted_text = extract_text_from_file(file_path)
    
    # Index the extracted text into Elasticsearch
    save_document_to_elasticsearch(filename=file.filename, content=extracted_text)
    
    return jsonify({'message': 'Document processed', 'extracted_text': extracted_text})

# Function to extract text from an image or PDF using Pytesseract and PyMuPDF
def extract_text_from_file(file_path):
    extracted_text = ""
    
    if file_path.lower().endswith('.pdf'):
        # Process PDF using PyMuPDF
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text directly from the PDF page
            text = page.get_text()
            if not text.strip():  # If no text is found, use OCR
                # Render the page to an image and perform OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            extracted_text += text + "\n"
        doc.close()
    else:
        # Process image files
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image)
    
    return extracted_text

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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
