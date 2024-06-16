
from PIL import Image
import requests
from ultralytics import YOLO
import numpy as np
import io
import os
import cv2
import base64
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import openai

import uvicorn
import logging
from fastapi import FastAPI, Form, Request, status, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from threading import Thread
from typing import Callable, Any

app = FastAPI(title="UTS VerdeTech LeafLens API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

# Load a pretrained YOLOv8n model
# Load the YOLO model
try:
    model = YOLO("best.pt" )
    #modelClassifier = YOLO("leaflens-classification.pt")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed")



## Set the API key and model name
MODEL="gpt-4o"

# temporary only:

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY))

def detect_objects(image):
    try:
        # Run inference on the input image
        results = model(image)
        if not results:
            return None
        # Process each result in the list
        for result in results:
            # Plot the detection results on the original image
            annotated_image = result.plot()

            # Convert the annotated image to bytes
            _, img_encoded = cv2.imencode('.png', annotated_image)

        return img_encoded
    except Exception as e:
        logging.error(f"Error in detect_objects: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    print('Request for index page received')
    return templates.TemplateResponse('index.html', {"request": request})

@app.post('/hello', response_class=HTMLResponse)
async def hello(request: Request, name: str = Form(...)):
    if name:
        print('Request for hello page received with name=%s' % name)
        return templates.TemplateResponse('hello.html', {"request": request, 'name':name})
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return RedirectResponse(request.url_for("index"), status_code=status.HTTP_302_FOUND)


@app.get('/favicon.ico')
async def favicon():
    file_name = 'favicon.ico'
    file_path = './static/' + file_name
    return FileResponse(path=file_path, headers={'mimetype': 'image/vnd.microsoft.icon'})



#@app.route('/detect', methods=['POST'])

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Invalid file type")

    # Read image into memory
    image_data = await file.read()
    # Convert to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Call the object detection method
    result = detect_objects(img)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to process image")
    
    print("Result type:", type(result))
    print("Result array shape:", result.shape)  # Useful if result is a NumPy array

    if result.size == 0:
        raise HTTPException(status_code=500, detail="Image processing failed or returned empty result")

    return Response(content=result.tobytes(), media_type="image/png")
    # Convert the results back to a file response
    #return FileResponse(result.tobytes(), media_type="image/png", filename="result.png")
    


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#@app.route('/analyze-leaf', methods=['POST'])
@app.post("/analyze-leaf")
async def detect(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type")


    # Getting the base64 string
    base64_image = await convert_image_to_base64(file)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
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
    response = {
        "model_response": content
    }
    return response


async def convert_image_to_base64(image_file: UploadFile):
    """Converts image file to base64 string."""
    contents = await image_file.read()
    return base64.b64encode(contents).decode('utf-8')


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8001)