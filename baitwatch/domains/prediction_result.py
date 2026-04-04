"""Define domains for different predictions types."""

import numpy as np
from pydantic import BaseModel, Field

# Species mapping for IFSP (Individual Fish Species Prediction)
SPECIES_MAPPING = {
    0: {"name": "Carcharhiniformes", "common_name": "Ground sharks"},
    1: {"name": "Chrysophrys auratus", "common_name": "Australasian Snapper"},
    2: {"name": "Moridae", "common_name": "Morid cods"},
    3: {"name": "Perciformes_sandy", "common_name": "Sandy-habitat perch-like fishes"},
    4: {"name": "Perciformes_silver", "common_name": "Silver perch-like fishes"},
    5: {"name": "Ray", "common_name": "Rays"},
    6: {"name": "Scorpaeniformes", "common_name": "Scorpionfish & flatheads"},
    7: {"name": "Tetradontiformes", "common_name": "Pufferfish & filefish"},
}

# FONF (Fish Or No Fish) mapping
FONF_MAPPING = {
    0: "no_fish",
    1: "fish",
}


class PredictionResult(BaseModel):
    """Model for interfacing prediction results from keras Models with web interface."""

    detection_type: str = Field(..., description="Type of detection performed (fonf or ifsp)")
    prediction: str = Field(..., description="Human-readable prediction result")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the prediction"
    )
    class_id: int = Field(..., description="Numeric class identifier")
    class_name: str | None = Field(None, description="Scientific/taxonomic name (IFSP only)")
    common_name: str | None = Field(None, description="Common name (IFSP only)")

    @classmethod
    def from_predict_result(
        cls, result: list[list[float]], detection_type: str
    ) -> "PredictionResult":
        """Build a PredictionResult object from a keras.Model.predict result.

        Args:
            result (list[list[float]]): A list of lists where each list represents
                the probability if fish detection on each images
            detection_type (str): Type of detection ("fonf" or "ifsp")

        Returns:
            PredictionResult: A PredictionResult object with enhanced information
        """
        # Format of fonf result: [[<proba_class_1>]]
        # Format of ifsp result: [[<proba_class_0>, ..., <proba_class_7>]]
        res = result[0]
        probability = max(res)

        if len(res) == 1:
            # FONF: Binary classification
            threshold = 0.5
            class_id = 1 if probability > threshold else 0
            # Reverse probability when class 0
            confidence = probability if class_id else 1 - probability
            prediction = FONF_MAPPING[class_id]

            return cls(
                detection_type=detection_type,
                prediction=prediction,
                confidence=float(confidence),
                class_id=class_id,
                class_name=None,
                common_name=None,
            )
        else:
            # IFSP: Multi-class species classification
            class_id = int(np.argmax(res))
            confidence = float(probability)
            species_info = SPECIES_MAPPING.get(
                class_id, {"name": "Unknown", "common_name": "Unknown"}
            )

            return cls(
                detection_type=detection_type,
                prediction=species_info["name"],
                confidence=confidence,
                class_id=class_id,
                class_name=species_info["name"],
                common_name=species_info["common_name"],
            )
