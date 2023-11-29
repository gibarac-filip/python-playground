import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = r'C:\Users\Filip Gibarac\Desktop\python-playground\test_photo.jpg'

image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print(text)