"""Define fish detection related domains."""

from enum import StrEnum


class FishDetectionEnum(StrEnum):
    """Supported type of fish detection.

    FONF: Fish Or No Fish
    IFSP = Individual Fish Species Prediction
    """

    FONF = "fonf"
    IFSP = "ifsp"
    # TODO: add support for WAW
    # WAW = "waw"
