import argparse
import sys
from io import BytesIO
import torch
import numpy as np
import satellighte as sat
import uvicorn
from PIL import Image

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi

tags_metadata = [
    {
        "name": "Predict",
        "description": "Satellighte is an image classification.",
        "externalDocs": {
            "description": "External Docs for Library: ",
            "url": "https://satellighte.readthedocs.io/",
        },
    },
    {
        "name": "Information",
        "description": "Information about the library and the service.",
        "externalDocs": {
            "description": "Project Homepage: ",
            "url": "https://canturan10.github.io/satellighte/",
        },
    },
]

app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Satellighte API",
        version=sat.__version__,
        description=sat.__description__,
        routes=app.routes,
        tags=tags_metadata,
        license_info={
            "name": sat.__license__,
            "url": sat.__license_url__,
        },
        contact={
            "name": sat.__author__,
        },
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/canturan10/satellighte/master/src/satellighte.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
def load_artifacts():
    if not hasattr(app.state, "model"):
        app.state.model = sat.Classifier.from_pretrained("mobilenetv2_default_eurosat")
        app.state.model.eval()
        app.state.model.to("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("shutdown")
def empty_cache():
    # clear Cuda memory
    torch.cuda.empty_cache()


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.get("/", tags=["Information"])
def read_root():
    return {
        "Satellighte": f"{sat.__version__}",
        "Description": f"{sat.__description__}",
        "Author": f"{sat.__author__}",
    }


@app.post("/predict/", tags=["Predict"])
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not an image.",
        )

    try:
        contents = await file.read()
        image = np.array(read_imagefile(contents).convert("RGB"))
        predicted_class = app.state.model.predict(image)

        return predicted_class
    except Exception:
        e_info = sys.exc_info()[1]
        raise HTTPException(
            status_code=500,
            detail=str(e_info),
        )


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Runs the API server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the API on.",
    )
    parser.add_argument(
        "--port",
        help="The port to listen for requests on.",
        type=int,
        default=8080,
    )
    parser.add_argument(
        "--workers",
        help="Number of workers to use.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--reload",
        help="Reload the model on each request.",
        action="store_true",
    )
    parser.add_argument(
        "--use-colors",
        help="Enable user-friendly color output.",
        action="store_true",
    )

    args = parser.parse_args()
    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        use_colors=args.use_colors,
    )
