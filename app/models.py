from pydantic import BaseModel, field_validator
from pydantic import ConfigDict

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.1
    top_p: float = 0.9

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 2048:
            raise ValueError("Prompt exceeds maximum length of 2048 characters")
        return v

class GenerateResponse(BaseModel):
    generated_text: str
    model: str
    prompt_length: int
    generated_length: int

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_name: str
    adapter_path: str
