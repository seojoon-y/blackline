from ultralytics import YOLO
import os
import sys
from Config import RUN_NAME, PRETRAINED_YOLO_MODEL
import shutil
from Utils import download_file_if_not_exists

download_file_if_not_exists(PRETRAINED_YOLO_MODEL, f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{PRETRAINED_YOLO_MODEL}")

if os.path.isdir(RUN_NAME):
    print(f"Trained model already exists. Deleting {RUN_NAME} ...")
    shutil.rmtree(RUN_NAME)

# Load a COCO-pretrained YOLOv8n model
model = YOLO(PRETRAINED_YOLO_MODEL)

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=5, imgsz=640, project=".", name=RUN_NAME)
