import asyncio
import logging
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a support ticket classifier. "
    "Given a ticket, respond ONLY with a JSON object containing "
    '"category" (billing|technical|account|general) and '
    '"priority" (low|medium|high|critical).'
)

class ModelManager:
    def __init__(self, base_model_name: str, adapter_path: str):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        self._loaded = True
        logger.info("Model ready")

    @property
    def is_loaded(self):
        return self._loaded

    def _sync_generate(self, prompt, max_new_tokens, temperature, top_p):
        if not self._loaded:
            raise RuntimeError("Model is not loaded")
        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def generate(self, prompt, max_new_tokens=64, temperature=0.1, top_p=0.9):
        loop = asyncio.get_event_loop()
        fn = partial(self._sync_generate, prompt, max_new_tokens, temperature, top_p)
        return await loop.run_in_executor(None, fn)

manager = None
