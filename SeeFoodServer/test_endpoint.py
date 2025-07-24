import base64
import requests
import sys
import json

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_prediction_request(image_path):
    base64_img = encode_image_to_base64(image_path)
    url = ""
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"image": base64_img})

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_predict.py path_to_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    send_prediction_request(image_path)