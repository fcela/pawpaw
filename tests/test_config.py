from pawpaw.config import CompileOptions, SynthConfig, TrainConfig


def test_compile_options_defaults():
    opts = CompileOptions()
    assert opts.base_model == "Qwen/Qwen3-0.6B"
    assert opts.synth.n_per_category == 30
    assert opts.synth.dedup_threshold == 0.85
    assert opts.synth.min_examples == 100
    assert opts.train.lora_rank == 16
    assert opts.train.lora_alpha is None
    assert opts.train.effective_alpha == 32
    assert opts.train.epochs == 3
    assert opts.train.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_synth_config_fingerprint_is_deterministic():
    a = SynthConfig().fingerprint()
    b = SynthConfig().fingerprint()
    assert a == b
    c = SynthConfig(n_per_category=50).fingerprint()
    assert a != c


def test_compile_options_override():
    opts = CompileOptions(train=TrainConfig(lora_rank=32))
    assert opts.train.lora_rank == 32
    assert opts.train.effective_alpha == 64


def test_lora_alpha_auto_derived():
    assert TrainConfig(lora_rank=8).effective_alpha == 16
    assert TrainConfig(lora_rank=32).effective_alpha == 64
    assert TrainConfig(lora_rank=16, lora_alpha=64).effective_alpha == 64


def test_train_config_preset():
    draft = TrainConfig.preset("draft")
    assert draft.lora_rank == 4
    assert draft.epochs == 1
    standard = TrainConfig.preset("standard")
    assert standard.lora_rank == 16
    assert standard.epochs == 3
    custom = TrainConfig.preset("draft", epochs=2)
    assert custom.epochs == 2
    assert custom.lora_rank == 4
