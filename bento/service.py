from typing import Any, List
import bentoml
from pydantic import BaseModel
from bentoml.io import JSON
import numpy as np

llm_runner = bentoml.models.get("llm_cxr_qa:latest").to_runner()

svc = bentoml.Service(name="llm_cxr_qa", runners=[llm_runner])


class LLMCXRResponseDTO(BaseModel):
    generated_text: str
    generated_vq: List[Any]

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


@svc.api(input=bentoml.io.Text(), output=JSON(pydantic_model=LLMCXRResponseDTO))
async def generate(text: str) -> str:
    generated = await llm_runner.async_run(text, max_length=3000)
    return LLMCXRResponseDTO(
        generated_text=generated[0]["generated_text"],
        generated_vq=generated[0]["generated_vq"].tolist(),
    )
