import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import devices, settings, paths


# brought from https://huggingface.co/docs/transformers/main/en/model_doc/git and modified
class GITLargeCaptioning:
    MODEL_REPO = "microsoft/git-large-coco"

    def __init__(self):
        self.processor: AutoProcessor = None
        self.model: AutoModelForCausalLM = None
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.half,
        )

    def load(self):
        if self.model is None or self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_REPO, cache_dir=paths.setting_model_path,
                quantization_config=self.quantization_config,  device_map=devices.device
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_REPO, cache_dir=paths.setting_model_path,
                quantization_config=self.quantization_config,  device_map=devices.device
            )

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.model = None
            self.processor = None
            devices.torch_gc()

    def apply(self, image):
        if self.model is None or self.processor is None:
            return ""
        inputs = self.processor(images=image, return_tensors="pt").to(devices.device)
        ids = self.model.generate(
            pixel_values=inputs.pixel_values.to(torch.float16),
            max_length=settings.current.interrogator_max_length,
        )
        return self.processor.batch_decode(ids, skip_special_tokens=True)
