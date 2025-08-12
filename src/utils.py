from pathlib import Path
from dataclasses import dataclass
from typing import *

@dataclass
class ModelSpec:
    name: str
    max_length: int = 512
    stride: int = 64

MODEL_SPECS: Dict[str, ModelSpec] = {
    "deberta": ModelSpec("microsoft/deberta-v3-base", max_length=512, stride=64),
    "modernbert": ModelSpec("answerdotai/ModernBERT-base", max_length=2048, stride=64),
    "bert": ModelSpec("bert-base-multilingual-cased", max_length=512, stride=64),
    "gpt-oss-20b": ModelSpec(
    name="openai/gpt-oss-20b", max_length=4096, stride=256)
}
