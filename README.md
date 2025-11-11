# WORKSHOP---5-License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier
# NAME:RANJAN KUMAR G
# REG NO:212223240138
# Aim
The aim is to identify and isolate all human frontal faces within a given static image.

# Materials Required
Python 3.x

opencv-python

matplotlib

An input image (e.g., grpp.jpg)
# Steps to Run
Clone the repository.

Install the required libraries:

Bash

pip install opencv-python matplotlib
Place your input image in the same directory as the script.

Change the image_path variable in the script to match your image's filename.

Run the script:

Bash

python your_script_name.py
# Program
```
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'grpp.jpg'  # <-- Change this to your image filename
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing (optional)
# -------------------------------------------------------------
# Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Histogram Equalization for better contrast
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load or Download Haar Cascade for Face Detection
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Auto-download if not present
if not os.path.exists(cascade_path):
    print("Cascade file not found. Downloading...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect faces using Haar Cascade
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,          # Preprocessed grayscale image
    scaleFactor=1.1,    # Scaling factor between image pyramid layers
    minNeighbors=5,     # Higher value -> fewer false detections
    minSize=(30, 30)    # Minimum object size
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw bounding boxes and save cropped faces
# -------------------------------------------------------------
output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

# -------------------------------------------------------------
# Step 7: Display the final output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()

```
# Output
<img width="648" height="467" alt="image" src="https://github.com/user-attachments/assets/9757e33a-7700-4bd2-b28d-c7d4bcd4b14d" />
<img width="679" height="467" alt="image" src="https://github.com/user-attachments/assets/cc5ca4f4-f4f4-43b2-88b6-702c0bbeaed5" />
<img width="791" height="455" alt="image" src="https://github.com/user-attachments/assets/48d7a32c-d6bc-4265-be75-d6165a71dd8e" />
<img width="680" height="460" alt="image" src="https://github.com/user-attachments/assets/5197366d-505f-4319-b64d-dac127a9718d" />



# Result
The program successfully identifies all frontal faces in the source image and extracts them into a separate folder, demonstrating the effectiveness of the Haar Cascade classifier for face detection.
