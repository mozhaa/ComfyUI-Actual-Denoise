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
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "actual_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("denoise", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, actual_denoise):
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
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "actual_denoise": (
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

    def execute(self, model, scheduler, actual_denoise, steps):
        total_sigma_steps = 1000
        idx = get_sigma_index(model, scheduler, actual_denoise, total_sigma_steps)

        start_at_step = int(round(idx * steps / total_sigma_steps))

        start_at_step = max(0, min(steps, start_at_step))
        return (start_at_step, steps, scheduler)


def get_actual_denoise(model, scheduler, start_at_step, steps, total_sigma_steps=1000):
    start_at_step = max(0, min(steps, start_at_step))
    idx_high = int(round(start_at_step * total_sigma_steps / steps))
    idx_high = max(0, min(total_sigma_steps - 1, idx_high))

    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(
        model_sampling, scheduler, total_sigma_steps
    ).cpu()

    sigma_start = sigmas[idx_high].item()
    sigma0 = sigmas[0].item()
    return sigma_start / sigma0


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
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("FLOAT", comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("actual_denoise", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, denoise, steps):
        if denoise <= 0.0:
            return (0.0, scheduler)
        if denoise >= 1.0:
            return (1.0, scheduler)

        steps_used = int(round(denoise * steps))
        start_at_step = steps - steps_used
        actual_denoise = get_actual_denoise(model, scheduler, start_at_step, steps)
        return (actual_denoise, scheduler)


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
    RETURN_NAMES = ("actual_denoise", "scheduler")
    FUNCTION = "execute"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def execute(self, model, scheduler, start_at_step, steps):
        actual_denoise = get_actual_denoise(model, scheduler, start_at_step, steps)
        return (actual_denoise, scheduler)
