"""Chains"""
from dataclasses import dataclass
from typing import List

from src.base.base_combiner import Combiner
from src.base.base_transformer import Transformer


@dataclass
class Chain:
    """Chain of transformers"""

    output_name: str
    name: str


@dataclass
class TransformerChain(Chain):
    """Defines a transformer chain"""

    output_name: str
    input_name: str
    transformers: List[Transformer]


@dataclass
class CombinerChain(Chain):
    """Defines a combiner chain"""

    input_name1: str
    input_name2: str
    combiner: Combiner
