"""Define domains for different predictions types."""

import numpy as np
from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Model for interfacing prediction results from keras Models with web interface."""
    probability: float
    class_id: int

    @classmethod
    def from_predict_result(cls, result: list[list[float]]) -> 'PredictionResult':
        """Build a PredictionResult object from a keras.Model.predict result.

        Args:
            result (list[list[float]]): A list of lists where each list represents
                the probability if fish detection on each images

        Returns:
            PredictionResult: A PredictionResult object
        """
        # Format of fonf result: [[<proba_class_1>]]
        # Format of ifsp result: [[<proba_class_0>, ..., <proba_class_7>]]
        res = result[0]
        probability = max(res)
        threshold = 0.5
        if len(res) == 1:
            class_id = 1 if probability > threshold else 0
            # Reverse probability when class 0
            probability = probability if class_id else 1 - probability
        else:
            class_id = np.argmax(res)
        return PredictionResult(probability=probability, class_id=class_id)
