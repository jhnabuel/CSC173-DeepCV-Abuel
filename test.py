from ultralytics import YOLO
import cv2
import os

# Get the folder where this script (test.py) lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path correctly
model_path = os.path.join(current_dir, "models", "base_model.pt")

model = YOLO(model_path)
results = model.predict(source="0", show=True, conf=0.25)

# The window will close when you press 'q' or 'Esc' in some versions