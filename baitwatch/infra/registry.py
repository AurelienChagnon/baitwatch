from pathlib import Path
from time import strftime

from google.cloud import storage
from tensorflow import keras

from baitwatch.domains.fish_detection import FishDetectionEnum
from baitwatch.settings import cloud_settings, model_settings


def save_model(
    model: keras.Model,
    model_type: FishDetectionEnum,
    path: Path = model_settings.MODEL_LOCAL_PATH
) -> None:
    """Save the given model in given path and in the Cloud."""
    model_path = path / model_type.value
    print(f"⏳ Saving model locally at {model_path}...")

    if not model_path.exists():
        model_path.mkdir(parents=True)

    timestamp = strftime("%Y%m%d-%H%M%S")
    model_name = f"model_{timestamp}.keras"
    model.save(model_path / model_name)

    print(f"✅ Model {model_name} saved locally at {model_path}")

    if model_settings.MODEL_TARGET == "gcs":
        print("⏳ Saving model on GCS...")

        client = storage.Client()
        bucket = client.bucket(cloud_settings.BUCKET_NAME)
        blob = bucket.blob(f"models/{model_type.value}/{model_name}")
        blob.upload_from_filename(model_path / model_name)

        print("✅ Model saved to GCS")


def load_model(
    model_type: FishDetectionEnum,
    path: Path = model_settings.MODEL_LOCAL_PATH,
    model_name: str = ""
) -> keras.Model:
    """Load the model from local or Cloud.

    If no model name is passed, return the last model in the path.
    """
    print(f"⏳ Loading model for {model_type.value}...")
    path = path / model_type.value

    if model_settings.MODEL_TARGET == "local":

        if not path.exists():
            raise FileNotFoundError(f"Path or directory does not exists: {path}")

        models = [file_path for file_path in path.iterdir() if file_path.name.endswith(".keras")]
        if not models:
            raise FileNotFoundError(f"No keras model found at {path}")

        # Get the last model (creation date) when no name passed
        if not model_name:
            models.sort(key=lambda model_path: model_path.stat().st_ctime)
            model_name = models[-1].name

        if path / model_name not in models:
            raise FileNotFoundError(f"Model {model_name} not found at {path}")

        model = keras.models.load_model(path / model_name)
        print(f"✅ Model {model_name} loaded")

    elif model_settings.MODEL_TARGET == "gcs":
        print("⏳ Load latest model from GCS...")

        client = storage.Client()
        # Don't get the bucket from client.bucket as rights can be different
        blobs = list(client.list_blobs(cloud_settings.BUCKET_NAME, prefix=f"models/{model_type.value}"))

        # Latest model
        latest_blob = max(blobs, key=lambda x: x.updated)

        # Only get the model file name not the full GCS Bucket path
        latest_blob_name = latest_blob.name.split("/")[-1]

        # Create path if downloaded for first ime
        if not path.exists():
            path.mkdir(parents=True)

        latest_model_path_to_save = path / latest_blob_name
        latest_blob.download_to_filename(latest_model_path_to_save)

        model = keras.models.load_model(latest_model_path_to_save)

        print(f"✅ Latest model downloaded from cloud storage {latest_blob_name}")

    else:
        # Unknown model target
        raise ValueError(f"Unknown model target {model_settings.MODEL_TARGET}")

    return model
