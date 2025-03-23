# Python Development for YOLOv8 & YOLOX 

Basic Requirement for development and procedure for dependency installation:

## 1. Install dependency
   - Install OpenCV: `pip install opencv_python`
   - Install Ultralytics YOLO: `pip install ultralytics`

## 2. Train YOLO model
**Important**: If `run/weights/best.pt` exists, then you can proceed directly to step 3.

### 2.1 Label images
roboflow yolov8

### 2.2

```
python3 Training.py
```
   - if you has recived available YOLO which has been trained for detection blackline, then making a change for "yolov8n.pt" to be part of particular model.
   - if you no model yet, you may trainning YOLOv8 with avaiable dataset which contain black line on top of gold backgound.

## 3. Use the YOLO model 
   - Run python code to make sure the code can be passed run and able to initilized real time detecting via camera.
   - The code will be display 2 panel, the frist panel is result form YOLO and ellipse around blackline and sencond panel is mask of the black line.
