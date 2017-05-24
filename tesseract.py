#using tesseract-ocr engine to convert image to text

from pytesseract import image_to_string
import Image
import os

digits = []

folder = '/media/sarthak/New Volume1/number plate recognition/testing/Numbers'
files = os.listdir(folder)
for f in files:

    textString = image_to_string(Image.open(os.path.join(folder,f)), config= '-psm 10')
    print textString
    digits.append(textString)

#cv2.destroyAllWindows()
print '--------', '\n', digits

