
from flask import Flask
from ultralytics import YOLO
# Load a pretrained YOLOv8n model

model = YOLO("best.pt")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello World"


if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8000)