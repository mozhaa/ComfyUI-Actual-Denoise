from .nodes import AccurateDenoise, AccurateDenoiseStep

NODE_CLASS_MAPPINGS = {
    "AccurateDenoise": AccurateDenoise,
    "AccurateDenoiseStep": AccurateDenoiseStep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccurateDenoise": "Accurate Denoise",
    "AccurateDenoiseStep": "Accurate Denoise (step)",
}
