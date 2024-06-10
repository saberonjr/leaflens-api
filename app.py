
from flask import Flask, request, send_file, jsonify
from PIL import Image
import requests
from ultralytics import YOLO
import io
import os
import base64
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import openai

app = Flask(__name__)

load_dotenv(find_dotenv())

# Load a pretrained YOLOv8n model
#model = YOLO("yolov8s-seg.pt")
model = YOLO("best.pt")
#model = YOLO("yolov8hp.pt")
#model = YOLO("yolov8best200.pt")

## Set the API key and model name
MODEL="gpt-4o"
OPEN_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPEN_API_KEY))

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


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/analyze-leaf', methods=['POST'])
def analyze_leaf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400


    # Getting the base64 string
    base64_image = image_data = convert_image_to_base64(file) #encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPEN_API_KEY}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are plant pathologist. Given an image of plant leaf, name the plant, the disease if present, and possible remedies. Limit result to 256 tokens."},
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    responseText = response.json()
    print(responseText)
    content = responseText['choices'][0]['message']['content']
    print(content)
    #print(content)
    response = {
        "model_response": content
    }
    return jsonify(response)

def analyze_leaf4():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Convert the uploaded image file to base64
    image_data = convert_image_to_base64(file)



    # Example usage
    
    #token_count = count_tokens(image_data)
    #print(f"Token count: {token_count}")
    #return jsonify({'token_count': token_count})    

    # Create the completion using the OpenAI API
    completion = client.chat.completions.create(
        model="gpt-4o",  # Adjust the model version as required
        messages=[
            {"role": "system", "content": "You are plant pathologist. Given an image of plant leaf, name the plant, the disease if present, and possible remedies. Limit result to 256 tokens."},
            {"role": "user", "content": image_data}  # Assuming the model can handle base64 images as text
        ]
    )

    result = completion.choices[0].message.content

    # Extract and return the relevant information
    response = {
        "model_response": result
    }
    return jsonify(response)

def convert_image_to_base64(image_file):
    """Converts image file to base64 string."""
    buffer = BytesIO()
    image_file.save(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8000)