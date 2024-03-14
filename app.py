import cv2
import json
import torch
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form

from config import settings
from modeling.models import SimpleCNN, SimpleResNet, ComplexResNet, ComplexResNetV2

app = FastAPI()
logging.basicConfig(level=logging.INFO)

num_classes = settings.num_classes
model_type = settings.model_type
device = settings.device

file_json = "label_mapping.json"
file_desc_json = open(file_json)
label_mapping = json.load(file_desc_json)
logging.info(label_mapping)

file_model_local = f"./trained_models/{model_type}/{model_type}.pt"
file_model_cont = f"/data/models/{model_type}/{model_type}.pt"

logging.info(f"model_type: {model_type}")
if model_type == "simple_cnn":
    overhead_mnist_model = SimpleCNN(num_classes=num_classes)
elif model_type == "simple_resnet":
    overhead_mnist_model = SimpleResNet(num_classes=num_classes)
elif model_type == "medium_simple_resnet":
    overhead_mnist_model = SimpleResNet(
        list_num_res_units_per_block=[4, 4], num_classes=num_classes
    )
elif model_type == "deep_simple_resnet":
    overhead_mnist_model = SimpleResNet(
        list_num_res_units_per_block=[6, 6], num_classes=num_classes
    )
elif model_type == "complex_resnet":
    overhead_mnist_model = ComplexResNet(
        list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
    )
elif model_type == "complex_resnet_v2":
    overhead_mnist_model = ComplexResNetV2(
        list_num_res_units_per_block=[4, 4, 4], num_classes=num_classes
    )

try:
    logging.info(f"loading model from {file_model_local}")
    overhead_mnist_model.load_state_dict(
        torch.load(file_model_local, map_location=device)
    )
except:
    logging.info(f"loading model from {file_model_cont}")
    overhead_mnist_model.load_state_dict(
        torch.load(file_model_cont, map_location=device)
    )
overhead_mnist_model.to(device)
overhead_mnist_model.eval()


def get_prediction(img_arr: np.ndarray) -> str:
    """
    ---------
    Arguments
    ---------
    img_arr: ndarray
        a numpy array of the image

    -------
    Returns
    -------
    pred_label_str : str
        a string representing the label of the prediction
    """
    img_arr = np.expand_dims(np.expand_dims(img_arr, 0), 0)
    img_arr = img_arr.astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_arr).float()
    img_tensor = img_tensor.to(device, dtype=torch.float)
    pred_logits = overhead_mnist_model(img_tensor)
    pred_label = torch.argmax(pred_logits, dim=1)

    pred_label_arr = pred_label.detach().cpu().numpy()
    pred_label_arr = np.squeeze(pred_label_arr)
    pred_label_str = label_mapping[str(pred_label_arr)]
    return pred_label_str


@app.get("/info")
def get_app_info() -> dict:
    """
    -------
    Returns
    -------
    dict_info : dict
        a dictionary with info to be sent as a response to get request
    """
    dict_info = {"app_name": settings.app_name, "version": settings.version}
    return dict_info


@app.post("/predict")
def _file_upload(image_file: UploadFile = File(...)) -> dict:
    """
    ---------
    Arguments
    ---------
    image_file: object
        an object of type UploadFile

    -------
    Returns
    -------
    response_json : dict
        a dict as a response json for the post request
    """
    logging.info(image_file)
    img_str = image_file.file.read()
    img_decoded = cv2.imdecode(np.frombuffer(img_str, np.uint8), 0)
    pred_label_str = get_prediction(img_decoded)
    response_json = {"name": image_file.filename, "prediction": pred_label_str}
    logging.info(response_json)
    return response_json
