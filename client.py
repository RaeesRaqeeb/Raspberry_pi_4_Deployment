# Install on Pi
# pip3 install onnxruntime

import onnxruntime as ort
import numpy as np
import cv2, requests, base64, time
from picamera2 import Picamera2

BACKEND_URL = "https://fresh-fruit-detection-embedded-ai.vercel.app/detect"

# Load ONNX model
session = ort.InferenceSession("model.onnx",
            providers=["CPUExecutionProvider"])

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Check input/output shape
print("Input shape:",  session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW (ONNX needs this)
    return np.expand_dims(img, axis=0)   # add batch dim → [1,3,640,640]

def run(frame):
    return session.run([output_name],
                       {input_name: preprocess(frame)})[0]

def encode(frame):
    small = cv2.resize(frame, (480, 360))
    _, buf = cv2.imencode('.jpg', small,
                [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode()

cam = Picamera2()
cam.start()

while True:
    frame = cam.capture_array()
    dets  = run(frame)
    try:
        requests.post(BACKEND_URL,
            json={"image": encode(frame),
                  "detections": dets.tolist()},
            timeout=2)
    except: pass
    time.sleep(0.1)