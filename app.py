
from flask import Flask, request, send_file, jsonify
from PIL import Image
import requests
from ultralytics import YOLO
import io
import os
import base64
from io import BytesIO
from openai import OpenAI
import openai
app = Flask(__name__)

# Load a pretrained YOLOv8n model
#model = YOLO("yolov8s-seg.pt")
model = YOLO("best.pt")
#model = YOLO("yolov8hp.pt")
#model = YOLO("yolov8best200.pt")

## Set the API key and model name
MODEL="gpt-4o"
OPEN_API_KEY = "sk-proj-nSSh5COvFT01PIg0H0oRT3BlbkFJZJRLhYR8TanJ6t9T1mRF"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-nSSh5COvFT01PIg0H0oRT3BlbkFJZJRLhYR8TanJ6t9T1mRF"))

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

def count_tokens(text):
    return len(openai.Tokenizer().encode(text))

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

    # Path to your image
    #image_path = "path_to_your_image.jpg"

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
    content = responseText['choices'][0]['message']['content']
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
    return jsonify(result)

    # Extract and return the relevant information
    response = {
        "model_response": completion.choices[0].message['content']
    }
    return jsonify(response)

def convert_image_to_base64(image_file):
    """Converts image file to base64 string."""
    buffer = BytesIO()
    image_file.save(buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def analyze_leaf3():
    user_message = request.json.get('user_message', '')
    if not user_message:
        return jsonify({'error': 'No user message provided'}), 400
    completion = client.chat.completions.create(
    model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in diseases of plants. You can recognize the diseases by examining the leaves of plants. When you are presented with an image of a leaf with signs of disease, you give more information about the plant and the disease information. You also give information about possible remedies for the plant disease."}, # <-- This is the system message that provides context to the model
            {"role": "user", "content": user_message}  # <-- This is the user message for which the model will generate a response
        ]
    )


    print("Assistant: " + completion.choices[0].message.content)
    return jsonify(completion.choices[0].message.content)

def analyze_leaf2():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Assuming the file is a proper image
    # For simplicity, directly sending it as binary data might not be optimal or correct depending on API specs
    img_data = file.read()

    # Example placeholder - replace with actual OpenAI API call
    response = call_openai_api(img_data)

    return jsonify(response)

def call_openai_api(image_data):
    # This is a placeholder function. You need to implement the actual API call to OpenAI.
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    # Construct the data payload as per OpenAI's requirements
    data = {
        'prompt': 'Analyze this leaf',  # Example prompt, adjust based on actual requirement
        'image': image_data,
        'model': 'image-gpt-3'  # Example model, use the correct one
    }
    response = requests.post('https://api.openai.com/v1/call', headers=headers, json=data)
    return response.json()

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8000)