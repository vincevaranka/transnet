import pytesseract
from PIL import Image

# Specify the path to the Tesseract executable (only needed on Windows)
pytesseract.pytesseract.tesseract_cmd = r'A:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image from file
image_path = 'source/example1.png'  # Replace with your image path
image = Image.open(image_path)

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(image)

# Print the extracted text
print(text)
