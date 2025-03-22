import sys
import os
from ultralytics import YOLO
from Config import RUN_NAME

if os.path.isdir(RUN_NAME) == False:
    print(f"Please train model first.")
    sys.exit()

# Load model
model = YOLO(f"{RUN_NAME}/weights/best.pt")

# Run inference with the YOLOv8n model on the test image
test_image_path = "test_input.jpg"
results = model(test_image_path)
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
result.save(filename=f"test_output.jpg")  # save to disk
