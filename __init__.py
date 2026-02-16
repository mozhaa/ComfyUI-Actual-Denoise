from .nodes import (
    AccurateDenoise,
    AccurateDenoiseInverse,
    AccurateDenoiseInverseStep,
    AccurateDenoiseStep,
)

NODE_CLASS_MAPPINGS = {
    "AccurateDenoise": AccurateDenoise,
    "AccurateDenoiseStep": AccurateDenoiseStep,
    "AccurateDenoiseInverse": AccurateDenoiseInverse,
    "AccurateDenoiseInverseStep": AccurateDenoiseInverseStep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccurateDenoise": "Accurate Denoise",
    "AccurateDenoiseStep": "Accurate Denoise (step)",
    "AccurateDenoiseInverse": "Accurate Denoise Inverse",
    "AccurateDenoiseInverseStep": "Accurate Denoise Inverse (step)",
}
