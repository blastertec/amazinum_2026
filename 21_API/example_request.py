"""Send an image to the running API and print the prediction.

Usage: python example_request.py [image] [url]
"""

import base64
import json
import sys

import requests

image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_digit.png"
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/predict"

with open(image_path, "rb") as f:
    payload = {"image": base64.b64encode(f.read()).decode()}

print(f"POST {url}  ({image_path})")
resp = requests.post(url, json=payload, timeout=10)
print("status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
