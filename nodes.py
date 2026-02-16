import comfy.samplers


def get_sigmas(model, scheduler):
    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, 1000).cpu()
    return sigmas, sigmas[0].item()


def find_sigma_index(sigmas, actual_denoise):
    if actual_denoise <= 0.0:
        return len(sigmas)
    if actual_denoise >= 1.0:
        return 0

    threshold = actual_denoise * sigmas[0].item()
    idx = (sigmas < threshold).nonzero()
    return idx[0, 0].item() if len(idx) > 0 else len(sigmas)


def get_sigma_at_fraction(sigmas, step_fraction):
    n = len(sigmas)
    idx = int(round(step_fraction * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return sigmas[idx].item()


class ActualDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "actual_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "FLOAT", "MODEL")
    RETURN_NAMES = ("scheduler", "denoise", "model")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, actual_denoise):
        sigmas, _ = get_sigmas(model, scheduler)
        idx = find_sigma_index(sigmas, actual_denoise)
        denoise = 1.0 - (idx / len(sigmas))
        denoise = max(0.0, min(1.0, denoise))
        return (scheduler, denoise, model)


class ActualDenoiseStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "actual_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "INT", "INT", "MODEL")
    RETURN_NAMES = ("scheduler", "start_at_step", "steps", "model")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, actual_denoise, steps):
        sigmas, _ = get_sigmas(model, scheduler)
        idx = find_sigma_index(sigmas, actual_denoise)
        start_at_step = int(round(idx * steps / len(sigmas)))
        start_at_step = max(0, min(steps, start_at_step))
        return (scheduler, start_at_step, steps, model)


class ActualDenoiseInverse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "FLOAT", "MODEL")
    RETURN_NAMES = ("scheduler", "actual_denoise", "model")
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
        return (scheduler, actual_ratio, model)


class ActualDenoiseInverseStep:
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

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "FLOAT", "MODEL")
    RETURN_NAMES = ("scheduler", "actual_denoise", "model")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, start_at_step, steps):
        start_at_step = max(0, min(steps, start_at_step))
        step_fraction = start_at_step / steps
        sigmas, sigma0 = get_sigmas(model, scheduler)
        sigma_at_step = get_sigma_at_fraction(sigmas, step_fraction)
        actual_ratio = sigma_at_step / sigma0
        return (scheduler, actual_ratio, model)
