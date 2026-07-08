"""Handwritten digit recognizer: FastAPI JSON API + Gradio drawing UI.

Run with:  python app.py   (or: uvicorn app:app --port 8000)
UI:        http://127.0.0.1:8000/ui
API docs:  http://127.0.0.1:8000/docs
"""

import base64
import io
import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
# The Gradio UI talks to the API over plain HTTP, like any external client.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

if not MODEL_PATH.exists():
    raise SystemExit("model.joblib not found, run train.py first")
model = joblib.load(MODEL_PATH)


# --- model input preparation ------------------------------------------------

def prepare(image: Image.Image):
    """Turn an arbitrary drawing into the 28x28 format MNIST digits use.

    MNIST digits are white-on-black, scaled to fit a 20x20 box and placed
    on a 28x28 field so that their center of mass is in the middle. A raw
    canvas screenshot looks nothing like that, so we redo the same
    normalization here. Returns a (1, 784) array, or None if the image
    contains no ink.
    """
    if image.mode == "RGBA":  # flatten transparency onto white
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image)
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if gray.mean() > 127:  # dark strokes on light background -> invert
        gray = 255.0 - gray
    gray[gray < 20] = 0  # drop faint anti-aliasing noise

    ys, xs = np.nonzero(gray)
    if len(xs) == 0:
        return None
    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # fit into 20x20 keeping aspect ratio
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = np.asarray(
        Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    )

    # place on 28x28 so the center of mass ends up in the middle
    total = digit.sum()
    ys, xs = np.indices(digit.shape)
    cy = (digit * ys).sum() / total
    cx = (digit * xs).sum() / total
    y0 = min(max(int(round(13.5 - cy)), 0), 28 - new_h)
    x0 = min(max(int(round(13.5 - cx)), 0), 28 - new_w)

    field = np.zeros((28, 28), dtype=np.float32)
    field[y0:y0 + new_h, x0:x0 + new_w] = digit
    return field.reshape(1, 784) / 255.0


# --- API ----------------------------------------------------------------------

class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image file (PNG or JPEG)")


class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(title="Digit recognizer", description="MNIST digit recognition API")


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
def health():
    return {"status": "ok", "model": "MLPClassifier", "classes": 10}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image)))
        image.load()
    except Exception:
        raise HTTPException(400, "image must be a base64-encoded image file")

    x = prepare(image)
    if x is None:
        raise HTTPException(400, "the image is empty, draw a digit first")

    probs = model.predict_proba(x)[0]
    digit = int(np.argmax(probs))
    return PredictResponse(
        digit=digit,
        confidence=round(float(probs[digit]), 4),
        probabilities={str(i): round(float(p), 4) for i, p in enumerate(probs)},
    )


# --- Gradio UI ----------------------------------------------------------------

def recognize(sketch):
    # Sketchpad hands over a dict of layers, "composite" is the merged image
    image = sketch["composite"] if isinstance(sketch, dict) else sketch
    if image is None:
        raise gr.Error("Draw a digit first")

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    payload = {"image": base64.b64encode(buf.getvalue()).decode()}

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        raise gr.Error(f"API is not reachable: {exc}")
    if resp.status_code != 200:
        raise gr.Error(resp.json().get("detail", "prediction failed"))

    return resp.json()["probabilities"]


with gr.Blocks(title="Digit recognizer") as demo:
    gr.Markdown("## Digit recognizer\nDraw a digit (0-9) and press Recognize.")
    with gr.Row():
        pad = gr.Sketchpad(
            label="Canvas",
            canvas_size=(280, 280),
            brush=gr.Brush(colors=["#000000"], default_size=18, color_mode="fixed"),
            type="numpy",
        )
        result = gr.Label(label="Prediction", num_top_classes=3)
    button = gr.Button("Recognize", variant="primary")
    button.click(recognize, inputs=pad, outputs=result)

app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
