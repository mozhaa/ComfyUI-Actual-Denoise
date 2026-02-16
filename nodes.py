import comfy.samplers
import torch


class AccurateDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "SCHEDULER")
    FUNCTION = "recompute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def recompute(self, model, scheduler, steps, denoise):
        if denoise <= 0.0:
            return (0.0,)
        if denoise >= 1.0:
            return (1.0,)

        total_sigma_steps = 1000
        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, total_sigma_steps
        ).cpu()

        sigma0 = sigmas[0].item()
        threshold = denoise * sigma0

        idx = (sigmas < threshold).nonzero()
        if len(idx) == 0:
            return (1.0,)

        idx_val = idx[0, 0].item()
        recomputed_denoise = idx_val / total_sigma_steps
        return (recomputed_denoise, scheduler)
