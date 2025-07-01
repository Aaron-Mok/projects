from music21 import converter, stream, environment, metadata, layout
import subprocess
from pdf2image import convert_from_path
import cv2
import numpy as np

# --- Setup paths ---
us = environment.UserSettings()
us['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

# --- Load original score ---
score = converter.parse("assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl")
clean_meta = metadata.Metadata()
score.metadata = clean_meta
score.metadata.composer = ""
score.metadata.title = ""

# --- Build a new Score with both parts in parallel ---
measure_number = 1
m = score.measure(measure_number)

# Write to PDF
pdf_path = "measure_5.pdf"
m.write('musicxml.pdf', fp=pdf_path)

# Convert to PNG
png_path = "measure_5.png"
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
pages[0].save(png_path, "PNG")

# Display in OpenCV
img = cv2.imread(png_path)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Threshold to get binary image
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get bounding box of all contours combined
x, y, w, h = cv2.boundingRect(np.vstack(contours))
# Crop the image
cropped = img[y:y+h, x:x+w]

# Resize and display
resized = cv2.resize(cropped, (cropped.shape[1] // 2, cropped.shape[0] // 2))
cv2.imshow("Cropped Measure", resized)
cv2.moveWindow("Cropped Measure", 100, 100)
cv2.waitKey(0)
cv2.destroyAllWindows()