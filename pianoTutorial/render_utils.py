from music21 import stream, metadata
from pdf2image import convert_from_path
import cv2
import numpy as np
import os

# Setup paths
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
from music21 import environment
us = environment.UserSettings()
us['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# Ensure output directory exists
os.makedirs("rendered_measures", exist_ok=True)

def render_measure(score, measure_number, scale=0.5):
    # Check if image already exists
    output_path = f"rendered_measures/measure_{measure_number}.png"
    if os.path.exists(output_path):
        return cv2.imread(output_path)

    clean_meta = metadata.Metadata()
    score.metadata = clean_meta
    score.metadata.title = ""
    score.metadata.composer = ""

    # Extract the requested measure
    m = score.measure(measure_number)

    # Export to MusicXML
    pdf_path = f"rendered_measures/temp_measure_{measure_number}.pdf"
    png_path = f"rendered_measures/temp_measure_{measure_number}_raw.png"
    m.write('musicxml.pdf', fp=pdf_path)

    # Convert to PNG from PDF
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    pages[0].save(png_path, "PNG")

    # Load and crop
    img = cv2.imread(png_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = img[y:y+h, x:x+w]

    # Resize
    resized = cv2.resize(cropped, (int(cropped.shape[1] * scale), int(cropped.shape[0] * scale)))

    # Save for reuse
    cv2.imwrite(output_path, resized)

    return resized

# from music21 import converter, stream, environment, metadata, layout
# from pdf2image import convert_from_path
# import cv2
# import numpy as np

# poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Adjust if needed
# us = environment.UserSettings()
# us['musicxmlPath'] = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

# def render_measure(score, measure_number, scale=0.5):

#     clean_meta = metadata.Metadata()
#     score.metadata = clean_meta
#     score.metadata.composer = ""
#     score.metadata.title = ""

#     # --- Build a new Score with both parts in parallel ---
#     measure_number = 1
#     m = score.measure(measure_number)

#     # Write to PDF
#     pdf_path = "measure_5.pdf"
#     m.write('musicxml.pdf', fp=pdf_path)

#     # Convert to PNG
#     png_path = "measure_5.png"
#     pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
#     pages[0].save(png_path, "PNG")

#     # Display in OpenCV
#     img = cv2.imread(png_path)
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Threshold to get binary image
#     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Get bounding box of all contours combined
#     x, y, w, h = cv2.boundingRect(np.vstack(contours))
#     # Crop the image
#     cropped = img[y:y+h, x:x+w]

#     # Resize and display
#     # resized = cv2.resize(cropped, (cropped.shape[1] * scale, cropped.shape[0] * scale))

#     return cropped