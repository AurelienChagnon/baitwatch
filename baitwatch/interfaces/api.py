"""Web API."""
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status
from PIL import Image

from baitwatch.app import detect_fishes
from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.domains.prediction_result import PredictionResult
from baitwatch.infra.registry import load_model
from baitwatch.logger import logger


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
    root_path="/api/v1",
    title="Baitwatch API",
    description="Project to detect fishes in photographs.",
    version="1.0.0",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

router = APIRouter(prefix="/fish", tags=["fish-detection"])


@router.post(
    "/detect-fishes/",
    status_code=status.HTTP_200_OK,
    responses={
        400: {
            "description": "Bad Request - Invalid file type or file too large",
            "content": {
                "application/json": {
                    "example": {
                        "detail":
                            "Invalid image file type. "
                            "Supported types: image/jpeg, image/png, image/webp"
                    }
                }
            }
        },
        413: {
            "description": "Payload Too Large - File size exceeds limit",
            "content": {
                "application/json": {
                    "example": {"detail": "File size exceeds maximum limit of 10MB"}
                }
            }
        },
        422: {
            "description": "Unprocessable Entity - Invalid image data",
            "content": {
                "application/json": {
                    "example": {"detail": "Unable to process image file"}
                }
            }
        },
        500: {
            "description":
                "Internal Server Error - Model not available for the requested detection type",
            "content": {
                "application/json": {
                    "example": {"detail": "Model not available for detection type: fonf"}
                }
            }
        }
    }
)
async def detect(
        detection_type: FishDetectionEnum,
        image_file: Annotated[
            UploadFile,
            File(description="Image file to analyze (JPEG, PNG, or WebP)")
        ],
) -> PredictionResult:
    """Request a fish detection on given image, according to the detection type.

    Args:
        detection_type (FishDetectionEnum): Type of detection to use.
        image_file (UploadFile): image to detect fishes from.

    Returns:
        PredictionResult: Result of the fish detection.

    Raises:
        HTTPException: If the model is not available for the given detection type.
        HTTPException: If the image file is invalid.
        HTTPException: If the image file is too large.
    """
    logger.info(f"Received fish detection request: type={detection_type.value}, "
                f"file={image_file.filename}")

    # Validate content type
    allowed_content_types = {"image/jpeg", "image/png", "image/webp"}
    if image_file.content_type not in allowed_content_types:
        logger.warning(f"Invalid content type: {image_file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file type. Supported types: {', '.join(allowed_content_types)}"
        )

    # Ensure Enum object is used
    detection_type = FishDetectionEnum(detection_type)

    # Read and validate file size
    contents = await image_file.read()
    max_file_size = 10 * 1024 * 1024  # 10MB
    if len(contents) > max_file_size:
        logger.warning(f"File size too large: {len(contents)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {max_file_size // (1024 * 1024)}MB"
        )

    logger.debug(f"Image file size: {len(contents)} bytes")

    # Parse image file
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.debug(f"Image dimensions: {image.size}")
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to process image file. Please ensure it's a valid image."
        ) from e

    # Get associated model
    model = app.state.models.get(detection_type, None)
    if model is None:
        logger.error(f"No model found for detection type {detection_type.value}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model not available for detection type: {detection_type.value}"
        )

    logger.info(f"Running fish detection with model for {detection_type.value}")
    try:
        results = detect_fishes(model, detection_type, image)
        prediction_result = PredictionResult.from_predict_result(results)
    except Exception as e:
        logger.error(f"Error during fish detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during fish detection"
        ) from e
    else:
        logger.info(f"Fish detection completed successfully: {prediction_result}")
        return prediction_result


app.include_router(router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message
    """
    return {"message": "Welcome to Baitwatch API"}


@app.get("/ping/")
async def ping() -> list[str]:
    """PING.

    Returns:
        PONG
    """
    return ["pong"]
