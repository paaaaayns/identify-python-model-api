import utils.utils as utils

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from torchvision import transforms
from unet import UNet
from PIL import Image

import torch
import cv2
import numpy as np
import io
import os
import pickle
import base64
import gzip
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_path = "../models/yolo_model.pt"
unet_path = "../models/unet.pth"
image_path = "../images/"

if not os.path.exists(image_path):
    os.makedirs(image_path)

yolo_model = YOLO(yolo_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
unet_model = UNet(in_channels=3, num_classes=1).to(device)
unet_model.load_state_dict(torch.load(unet_path, map_location=torch.device(device)))
unet_model.eval()

def yolo_task(image):
    print("Running YOLO Task...")
    results = yolo_model.predict(image)

    iris_data = []
    pupil_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])

            center_x, center_y, radius = map(float, utils.bbox_to_circle(x1, y1, x2, y2))

            if class_id == 0: # Iris Data
                iris_data.append((center_x, center_y, radius))

            elif class_id == 1: # Pupil Data
                pupil_data.append((center_x, center_y, radius))

    return iris_data, pupil_data

def unet_task(image):
    print("Running UNET Task...")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    image = transform(image).float().to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred_mask = unet_model(image)

    image = image.squeeze(0).cpu().detach()
    image = image.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    pred_mask = pred_mask.numpy()
    pred_mask = cv2.resize(pred_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
    pred_mask = (pred_mask * 255).astype(np.uint8)

    print("Finish")
    return pred_mask


def process_iris(image_bytes):
    """ Helper function to process an iris image """
    image = Image.open(io.BytesIO(image_bytes))
    image = utils.preprocess_image(image)

    iris_data, pupil_data = yolo_task(image)
    pred_mask = unet_task(image)

    localized_iris = utils.localize_iris(image, (iris_data[0], pupil_data[0]))
    normalized_iris, normalized_mask, normalized_iris_masked = utils.normalize_iris(image, pred_mask, (iris_data[0], pupil_data[0]))

    iris_blob, mask_blob = utils.extract_iris_code(normalized_iris, normalized_mask)

    # Save images for debugging
    # cv2.imwrite(image_path + "normalized_iris_mask.png", normalized_iris_masked)
    # cv2.imwrite(image_path + "normalized_iris.png", normalized_iris)
    # cv2.imwrite(image_path + "localized_iris.png", localized_iris)
    # cv2.imwrite(image_path + "normalized_mask.png", normalized_mask)
    # cv2.imwrite(image_path + "pred_mask.png", pred_mask)

    return iris_blob, mask_blob


@app.post('/fast-api/store')
async def store(left_iris: UploadFile = File(...), right_iris: UploadFile = File(...)):
    left_image = await left_iris.read()
    right_image = await right_iris.read()

    left_iris_blob, left_mask_blob = process_iris(left_image)
    right_iris_blob, right_mask_blob  = process_iris(right_image)

    # Compress and encode iris codes
    left_iris_base64 = base64.b64encode(gzip.compress(pickle.dumps(left_iris_blob))).decode('utf-8')
    left_mask_base64 = base64.b64encode(gzip.compress(pickle.dumps(left_mask_blob))).decode('utf-8')
    right_iris_base64 = base64.b64encode(gzip.compress(pickle.dumps(right_iris_blob))).decode('utf-8')
    right_mask_base64 = base64.b64encode(gzip.compress(pickle.dumps(right_mask_blob))).decode('utf-8')

    response_data = {
        "success": True,
        "message": "Image processed successfully.",
        "data": {
            "left_iris_code": left_iris_base64,
            "left_mask_code": left_mask_base64,
            "right_iris_code": right_iris_base64,
            "right_mask_code": right_mask_base64,
        },
    }

    return JSONResponse(content=response_data)

@app.post('/fast-api/search')
async def search(iris: UploadFile = File(...), stored_irises: str = Form(...)):
    image = await iris.read()

    uploaded_iris_code, uploaded_mask_code = process_iris(image)

    print(f"Done processing iris image.")

    stored_irises = json.loads(stored_irises)

    best_hamming_distance = 1.0

    for iris in stored_irises:
        stored_iris_code = gzip.decompress(base64.b64decode(iris['iris_code']))
        stored_mask_code = gzip.decompress(base64.b64decode(iris['mask_code']))

        print(stored_iris_code[:100])

        stored_iris_code = pickle.loads(stored_iris_code)
        stored_mask_code = pickle.loads(stored_mask_code)

        hamming_distance, shift, _ = utils.compute_shifted_hamming_distance(uploaded_iris_code, uploaded_mask_code, stored_iris_code, stored_mask_code)

        if hamming_distance < best_hamming_distance:
            best_hamming_distance = hamming_distance
            best_shift = shift
            patient_ulid = iris['patient_ulid']

    if best_hamming_distance <= 0.38:
        print(f"Match Found: {best_hamming_distance:.4f}")
        print(f"Shift: {best_shift}")
        print(f"Patient ULID: {patient_ulid}")

        response_data = {
            "success": True,
            "message": "Image processed successfully.",
            "data": {
                "patient_ulid": patient_ulid,
            },
        }
    else:
        print(f"No match found.")

        response_data = {
            "success": False,
            "message": "No match found.",
        }

    return JSONResponse(content=response_data)

@app.post('/compare')           # debugger
async def compare_iris(image1: UploadFile = File(...), image2: UploadFile = File(...)):

    image1 = await image1.read()
    image2 = await image2.read()

    print(f"\n\n--- Processing image1 ---")
    image1_iris_blob, image1_mask_blob = process_iris(image1)
    print(f"\n\n--- Processing image2 ---")
    image2_iris_blob, image2_mask_blob  = process_iris(image2)

    print(f"\n\n--- Computing Hamming Distance ---")
    best_hamming_distance, best_shift, is_match = utils.compute_shifted_hamming_distance(image1_iris_blob, image1_mask_blob, image2_iris_blob, image2_mask_blob)

    match_status = "✅ Matched" if is_match else "❌ Not Matched"

    print(f"\n--- Iris Comparison Result ---")
    print(f"Hamming Distance : {best_hamming_distance:.4f}")
    print(f"Best Shift       : {best_shift}")
    print(f"Result           : {match_status}")

    return match_status
    # sample curl command
    # curl -X POST "http://127.0.0.1:8000/compare" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "image1=@\"C:\Users\Acer\Documents\Projects\identify\Files\Datasets\irishield-capture\01\L\P01L03.bmp\"" -F "image2=@\"C:\Users\Acer\Documents\Projects\identify\Files\Datasets\irishield-capture\01\L\P01L01.bmp\""