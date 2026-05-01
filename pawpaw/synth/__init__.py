"""Synthesis stage: spec → training examples via LLM."""
from pawpaw.synth.dedup import dedup
from pawpaw.synth.examples import Pair, generate_all, generate_for_category
from pawpaw.synth.llm import LLM, LlamaCppLLM, complete_json_with_retry, parse_json_strict
from pawpaw.synth.taxonomy import Category, enumerate_categories

__all__ = [
    "LLM",
    "LlamaCppLLM",
    "Category",
    "Pair",
    "dedup",
    "enumerate_categories",
    "generate_all",
    "generate_for_category",
    "complete_json_with_retry",
    "parse_json_strict",
]
