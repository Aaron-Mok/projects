import os
import tempfile
import matplotlib.pyplot as plt
from music21 import converter, environment
import cv2
from pdf2image import convert_from_path
import numpy as np

# === Setup MuseScore path ===
us = environment.UserSettings()
us['musicxmlPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
us['musescoreDirectPNGPath'] = 'C:\\Program Files\\MuseScore 4\\bin\\MuseScore4.exe'
us['musicxmlPath']
# environment.set('musicxmlPath', r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")  # Adjust if needed

# === Load and parse the MusicXML file ===

score = converter.parse('assets/scores/BOOK-Richard-Clayderman-Music-of-Love_p1.mxl')  # Or use music21.stream.Stream object
pdf_path  = score.write('musicxml.pdf')  # Will export the full page to PNG

images = convert_from_path(pdf_path)

img_pil = images[0]  # Get first page

# Step 3: Convert PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Step 4: Display using OpenCV
cv2.imshow("Sheet Music", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # === Break score into measures ===
# measures = score.parts[0].makeMeasures()

# # === Convert each measure to PNG and store file paths ===
# measure_images = []
# for i, measure in enumerate(measures):
#     tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
#     tmp_png.close()
#     measure.write('musicxml.png', fp=tmp_png.name)
#     measure_images.append(tmp_png.name)

# # === Display all measures (or limit with measure_images[:N]) ===
# for i, path in enumerate(measure_images):
#     img = plt.imread(path)
#     plt.figure(figsize=(8, 2))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"Measure {i+1}")
#     plt.show()

# # === Optional: Clean up temp files ===
# # for path in measure_images:
# #     os.remove(path)
