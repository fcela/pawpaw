"""PEFT LoRA training on the chosen base model.

Determinism: transformers.set_seed, torch.use_deterministic_algorithms (warn-only),
single-threaded data loader, fixed shuffle seed. Cross-machine bit-equivalence is
not guaranteed; same-machine bit-equivalence after a dataset cache hit is.

Performance notes:
- SDPA attention (torch 2.0+) enablement uses attn_implementation="sdpa"
- Flash Attention 2 is used when available for better memory efficiency
- Memory-efficient attention falls back when Flash Attention is not available
"""
from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Sequence

import torch


def _disable_transformers_torchvision() -> None:
    """Pawpaw is text-only; broken optional torchvision installs should not break PEFT."""
    try:
        import transformers.utils as transformers_utils
        from transformers.utils import import_utils
    except Exception:
        return

    def unavailable() -> bool:
        return False

    for fn_name in ("is_torchvision_available", "is_torchvision_v2_available"):
        fn = getattr(import_utils, fn_name, None)
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
        setattr(import_utils, fn_name, unavailable)
        setattr(transformers_utils, fn_name, unavailable)


_disable_transformers_torchvision()

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from pawpaw.config import TrainConfig
from pawpaw.synth.examples import Pair
from pawpaw.train.dataset import build_train_records, train_val_split
from pawpaw.train.device import pick_device, pick_dtype

logger = logging.getLogger(__name__)


def _enable_determinism(seed: int) -> None:
    set_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_lora(
    *,
    base_model: str,
    template: str,
    pairs: Sequence[Pair],
    config: TrainConfig,
    output_dir: Path,
) -> Path:
    """Fine-tune a LoRA adapter on the given pairs. Returns the saved adapter directory."""
    _enable_determinism(config.seed)

    device = pick_device()
    dtype = pick_dtype(device)
    max_length = config.max_length

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_pairs, val_pairs = train_val_split(pairs, val_fraction=config.val_fraction, seed=config.seed)
    train_records = build_train_records(template, train_pairs, tokenizer=tokenizer, max_length=max_length)
    val_records = build_train_records(template, val_pairs, tokenizer=tokenizer, max_length=max_length)
    if not train_records:
        raise ValueError("All training records were dropped (check max_length)")

    # Try to use Flash Attention 2 or SDPA for better performance
    attn_implementation = None
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
    except Exception:
        logger.debug("Flash Attention 2 not available, falling back to SDPA")

    if attn_implementation is None and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        # SDPA is available in PyTorch 2.0+
        attn_implementation = "sdpa"

    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if device != "cpu":
        model = model.to(device)

    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.effective_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        report_to=[],
        # Performance optimizations
        dataloader_pin_memory=(device != "cpu"),
        bf16_full_eval=(dtype == torch.bfloat16),
        optim="adamw_torch_fused",  # Use fused AdamW when available
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_records,
        "eval_dataset": val_records or None,
        "data_collator": collator,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
