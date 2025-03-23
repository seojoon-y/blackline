from ultralytics import YOLO
import os
import sys
from Config import PRETRAINED_YOLO_MODEL
import shutil
import requests

TRAINING_EPOCHS = 50

def download_file_if_not_exists(local_path, remote_url):
    if os.path.exists(local_path):
        return
    r = requests.get(remote_url)
    with open(local_path, "wb") as f:
        f.write(r.content)

download_file_if_not_exists(PRETRAINED_YOLO_MODEL, f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{PRETRAINED_YOLO_MODEL}")

if os.path.isdir("run"):
    print(f"Trained model already exists. Deleting run ...")
    shutil.rmtree("run")

# Load a COCO-pretrained YOLOv8n model
model = YOLO(PRETRAINED_YOLO_MODEL)

# Display model information (optional)
model.info()

# Train the model on the dataset
results = model.train(data="datasets/samples/data.yaml", epochs=TRAINING_EPOCHS, imgsz=960, project=".", name="run")
