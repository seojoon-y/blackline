import sys
import os
from ultralytics import YOLO

INPUT_PATH = "test_input.jpg"
OUTPUT_PATH = "test_output.jpg"

if os.path.isdir("run") == False:
    print(f"Please train model first.")
    sys.exit()

# Load model
model = YOLO("run/weights/best.pt")

# Run inference with the YOLOv8 model on the test image
results = model(INPUT_PATH)
if len(results) != 1:
    print("Error: Expected 1 results.")
    sys.exit()

result = results[0]
boxes = result.boxes  # Boxes object for bounding box outputs
masks = result.masks  # Masks object for segmentation masks outputs
keypoints = result.keypoints  # Keypoints object for pose outputs
probs = result.probs  # Probs object for classification outputs
obb = result.obb  # Oriented boxes object for OBB outputs
# result.show()  # display to screen
result.save(filename=OUTPUT_PATH)  # save to disk
print(f"Image has been saved to {OUTPUT_PATH}")
