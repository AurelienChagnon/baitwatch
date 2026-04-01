"""Web API."""
import argparse
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile
from PIL import Image

from baitwatch.app import detect_fishes
from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.domains.prediction_result import PredictionResult
from baitwatch.infra.registry import load_model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: RUF029
    """Application's lifespan.

    Put resources that should be initialized once before the app,
    and dropped when closing app.
    """
    # Startup: Initialize resources
    app.state.models = {
        FishDetectionEnum.FONF: load_model(FishDetectionEnum.FONF),
        FishDetectionEnum.IFSP: load_model(FishDetectionEnum.IFSP),
    }
    yield
    # Shutdown: Clean up resources


app = FastAPI(
    lifespan=lifespan,
    title="Baitwatch API",
    description="Project to detect fishes in photographs.",
    version="1.0.0",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


@app.post("/detect-fishes/")
async def detect(
        detection_type: FishDetectionEnum,
        image_file: UploadFile,
) -> PredictionResult | dict[str, str]:
    """Request a fish detection on given image, according to the detection type.

    Args:
        detection_type (FishDetectionEnum): Type of detection to use.
        image_file (UploadFile): image to detect fishes from.

    Returns:
         Nothing for now
    """
    # Ensure Enum object is used
    detection_type = FishDetectionEnum(detection_type)

    # Cast file into image file
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Get associated model
    model = app.state.models.get(detection_type, None)
    if model is None:
        return {"error": f"No model found for detection type {detection_type.value}"}

    results = detect_fishes(model, detection_type, image)
    return PredictionResult.from_predict_result(results).model_dump(mode="json")


@app.get("/ping/")
async def ping() -> list[str]:
    """PING.

    Returns:
        PONG
    """
    return ["pong"]


def main() -> None:
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Baitwatch API Server")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')

    args = parser.parse_args()

    uvicorn.run(
        "baitwatch.interfaces.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
