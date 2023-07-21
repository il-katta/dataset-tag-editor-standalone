import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BitsAndBytesConfig

import devices, settings, paths


class BLIPLargeCaptioning:
    MODEL_REPO = "Salesforce/blip-image-captioning-large"

    def __init__(self):
        self.processor: BlipProcessor = None
        self.model: BlipForConditionalGeneration = None
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.half,
        )

    def load(self):
        if self.model is None or self.processor is None:
            self.processor = BlipProcessor.from_pretrained(
                self.MODEL_REPO, cache_dir=paths.setting_model_path,
                quantization_config=self.quantization_config,  device_map=devices.device
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
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
        inputs = self.processor(images=image, return_tensors="pt").to(devices.device).to(dtype=torch.float16)
        ids = self.model.generate(**inputs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)
