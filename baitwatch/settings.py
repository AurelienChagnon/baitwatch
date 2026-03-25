"""Project Settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings

PROJECT_PATH = Path(__file__).absolute().parent.parent
DATASET_NAME = "training_data_species_grouped"

# Load environment variable to overwrite default settings
load_dotenv(PROJECT_PATH / ".env")


class DatasetSettings(BaseSettings):
    """Settings about dataset used for training."""
    RAW_DATA_PATH: DirectoryPath = PROJECT_PATH / "data" / "raw"
    PROCESSED_DATA_PATH: DirectoryPath = PROJECT_PATH / "data" / "processed"
    ORIGINAL_SIZE: tuple[int, int] = (1080, 1920)  # Tensorflow: height width


class FonfSettings(BaseSettings):
    """Settings for preprocessing."""
    PREPROCESS_IMG_SIZE: tuple[int, int] = (256, 144)  # OpenCV: width height


class IfspSettings(BaseSettings):
    """Settings for preprocessing."""
    CROP_IMG_SIZE: tuple[int, int] = (105, 256)  # Tensorflow: height width


class ModelSettings(BaseSettings):
    """Settings about models."""
    MODEL_LOCAL_PATH: DirectoryPath = PROJECT_PATH / "models"
    MODEL_TARGET: str = "local"


class CloudSettings(BaseSettings):
    """Settings about the Google Cloud project / buckets..."""
    BUCKET_NAME: str = "baitwatch-bucket"


dataset_settings = DatasetSettings()
fonf_settings = FonfSettings()
ifsp_settings = IfspSettings()
model_settings = ModelSettings()
cloud_settings = CloudSettings()
