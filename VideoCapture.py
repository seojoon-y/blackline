import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("run/weights/best.pt")

# Open a video capture
cap = cv2.VideoCapture(0)  # 0 for webcam
while True:
   ret, frame = cap.read()
   if not ret:
       break
   # Perform inference
   results = model(frame)
   # Draw results on the frame
   for result in results:
       for *xyxy, conf, cls in result.boxes.data.tolist():
           x1, y1, x2, y2 = map(int, xyxy)
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

           # Here, need additional logic to calculate the angle from the detected shape or lines
           #
           # This might involve custom functions or further processing steps not covered by YOLO directly
   cv2.imshow('Frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()
