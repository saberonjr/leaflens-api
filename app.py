""" from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load the YOLO model
model = YOLO('best.pt')
# model = YOLO('yolov5s.pt')

def detect_objects(image):
    # Run inference on the input image
    results = model(image)

    # Process each result in the list
    for result in results:
        # Plot the detection results on the original image
        annotated_image = result.plot()

        # Convert the annotated image to bytes
        _, img_encoded = cv2.imencode('.png', annotated_image)

    return img_encoded

@app.route('/', methods=['GET'])
def home():
    return "<p>Hello world</p>"

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Read the image file
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)

    # Convert the image to opencv format (for the YOLO model)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    detection_results = detect_objects(image)

    # Convert the bytes to a file-like object
    img_bytes = io.BytesIO(detection_results)

    # Return the image file
    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8000)
 """
from flask import Flask, request, send_file
from PIL import Image
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load a pretrained YOLOv8n model
#model = YOLO("yolov8s-seg.pt")
#model = YOLO("yolov8s-seg.pt")
model = YOLO("yolov8hp.pt")

@app.route('/detect', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        image = Image.open(file.stream)
        #print(image)
        results = model(image, show_labels=False, show_boxes=True)  # adjust size according to your needs

        result = results[0]
        #print(f"Results {len(results)}")
        im_bgr = result.plot()

        # Convert BGR to RGB
        im_rgb = Image.fromarray(im_bgr[..., ::-1])

        # Save image to a bytes buffer
        buf = io.BytesIO()
        im_rgb.save(buf, format='JPEG')
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8000)