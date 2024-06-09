
from flask import Flask, request, send_file
from PIL import Image
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load a pretrained YOLOv8n model
#model = YOLO("yolov8s-seg.pt")
model = YOLO("best.pt")
#model = YOLO("yolov8hp.pt")
#model = YOLO("yolov8best200.pt")

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
        print("begin predicting")
        results = model(image)#, show_labels=True, show_boxes=True)  # adjust size according to your needs

        result = results[0]
        print(result)
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