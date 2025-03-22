from ultralytics import YOLO
import os
import sys
from Config import RUN_NAME
import shutil

if os.path.isdir(RUN_NAME):
    print(f"Trained model already exists. Deleting {RUN_NAME} ...")
    shutil.rmtree(RUN_NAME)

# Download:
# 1. https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
# 2. https://ultralytics.com/images/bus.jpg

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8s.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=5, imgsz=640, project=".", name=RUN_NAME)
