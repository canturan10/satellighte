import argparse
from io import BytesIO
from typing import Optional

import numpy as np
import uvicorn
from PIL import Image
import sys
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import satellighte as sat

from fastapi import FastAPI, File, UploadFile, HTTPException


app = FastAPI(
    title="Satellighte API",
    description=sat.__description__,
    version=sat.__version__,
    license_info={
        "name": sat.__license__,
        "url": sat.__license_url__,
    },
    contact={
        "name": sat.__author__,
    },
)


@app.on_event("startup")
def load_artifacts():
    if not hasattr(app.state, "model"):
        app.state.model = sat.Classifier.from_pretrained("mobilenetv2_default_eurosat")
        app.state.model.eval()


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.post("/")
def image_filter(img: UploadFile = File(...)):
    original_image = Image.open(img.file)

    filtered_image = BytesIO()
    original_image.save(filtered_image, "JPEG")
    filtered_image.seek(0)

    return StreamingResponse(filtered_image, media_type="image/jpeg")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not an image.",
        )

    try:
        import time

        time.sleep(10)
        contents = await file.read()
        image = np.array(read_imagefile(contents).convert("RGB"))
        predicted_class = app.state.model.predict(image)

        return predicted_class
    except Exception as error:
        e = sys.exc_info()[1]
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runs the API locally.")
    parser.add_argument(
        "--port", help="The port to listen for requests on.", type=int, default=8080
    )
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
