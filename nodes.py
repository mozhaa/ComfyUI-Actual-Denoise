import comfy.samplers


def get_sigmas(model, scheduler):
    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, 1000).cpu()
    return sigmas, sigmas[0].item()


def find_sigma_index(sigmas, target_sigma_ratio):
    if target_sigma_ratio <= 0.0:
        return len(sigmas)
    if target_sigma_ratio >= 1.0:
        return 0

    threshold = target_sigma_ratio * sigmas[0].item()
    idx = (sigmas < threshold).nonzero()
    return idx[0, 0].item() if len(idx) > 0 else len(sigmas)


def get_sigma_at_fraction(sigmas, step_fraction):
    n = len(sigmas)
    idx = int(round(step_fraction * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return sigmas[idx].item()


class AccurateDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "target_sigma_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("denoise", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, target_sigma_ratio):
        sigmas, _ = get_sigmas(model, scheduler)
        idx = find_sigma_index(sigmas, target_sigma_ratio)
        denoise = 1.0 - (idx / len(sigmas))
        denoise = max(0.0, min(1.0, denoise))
        return (denoise, scheduler)


class AccurateDenoiseStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "target_sigma_ratio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("INT", "INT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("start_at_step", "steps", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, target_sigma_ratio, steps):
        sigmas, _ = get_sigmas(model, scheduler)
        idx = find_sigma_index(sigmas, target_sigma_ratio)
        start_at_step = int(round(idx * steps / len(sigmas)))
        start_at_step = max(0, min(steps, start_at_step))
        return (start_at_step, steps, scheduler)


class AccurateDenoiseInverse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("target_sigma_ratio", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, denoise):
        if denoise <= 0.0:
            return (0.0, scheduler)
        if denoise >= 1.0:
            return (1.0, scheduler)

        step_fraction = 1.0 - denoise
        sigmas, sigma0 = get_sigmas(model, scheduler)
        sigma_at_step = get_sigma_at_fraction(sigmas, step_fraction)
        actual_ratio = sigma_at_step / sigma0
        return (actual_ratio, scheduler)


class AccurateDenoiseInverseStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("target_sigma_ratio", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, start_at_step, steps):
        start_at_step = max(0, min(steps, start_at_step))
        step_fraction = start_at_step / steps
        sigmas, sigma0 = get_sigmas(model, scheduler)
        sigma_at_step = get_sigma_at_fraction(sigmas, step_fraction)
        actual_ratio = sigma_at_step / sigma0
        return (actual_ratio, scheduler)
