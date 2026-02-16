import comfy.samplers


def get_sigma_index(model, scheduler, actual_denoise, total_sigma_steps=1000):
    if actual_denoise <= 0.0:
        return total_sigma_steps
    if actual_denoise >= 1.0:
        return 0

    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(
        model_sampling, scheduler, total_sigma_steps
    ).cpu()

    sigma0 = sigmas[0].item()
    threshold = actual_denoise * sigma0

    idx = (sigmas < threshold).nonzero()
    if len(idx) == 0:
        return total_sigma_steps

    return idx[0, 0].item()


class AccurateDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "actual_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "scheduler")
    FUNCTION = "recompute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def recompute(self, model, scheduler, actual_denoise):
        total_sigma_steps = 1000
        idx = get_sigma_index(model, scheduler, actual_denoise, total_sigma_steps)

        denoise = 1.0 - (idx / total_sigma_steps)

        denoise = max(0.0, min(1.0, denoise))
        return (denoise, scheduler)


class AccurateDenoiseStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "actual_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("start_at_step", "steps", "scheduler")
    FUNCTION = "recompute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def recompute(self, model, scheduler, actual_denoise, steps):
        total_sigma_steps = 1000
        idx = get_sigma_index(model, scheduler, actual_denoise, total_sigma_steps)

        start_at_step = int(round(idx * steps / total_sigma_steps))

        start_at_step = max(0, min(steps, start_at_step))
        return (start_at_step, steps, scheduler)
