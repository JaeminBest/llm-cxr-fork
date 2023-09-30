import logging
import torch
import bentoml
from custom_pipeline import InstructionTextGenerationPipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pipelines import SUPPORTED_TASKS

print("start")
TASK_NAME = "llm-cxr-qa"
TASK_DEFINITION = {
    "impl": InstructionTextGenerationPipeline,
    "tf": (),
    "pt": (AutoModelForCausalLM,),
    "default": {},
    "type": "text",
}
SUPPORTED_TASKS[TASK_NAME] = TASK_DEFINITION

checkpoint_path = "ckpt/checkpoint-v3-12804s-1e+v2-2e"
print("initialize model start")
qa = InstructionTextGenerationPipeline(
    model=AutoModelForCausalLM(
        checkpoint_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ),
    tokenizer=AutoTokenizer.from_pretrained(checkpoint_path, padding_side="left"),
)

logging.basicConfig(level=logging.DEBUG)

print("save model start")
bentoml.transformers.save_model(
    "llm_cxr_qa",
    pipeline=qa,
    task_name=TASK_NAME,
    task_definition=TASK_DEFINITION,
)
