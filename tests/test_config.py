from pawpaw.config import CompileOptions, SynthConfig, TrainConfig, auto_n_threads


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
    assert opts.train.max_length == 1024


def test_synth_config_fingerprint_is_deterministic():
    a = SynthConfig().fingerprint()
    b = SynthConfig().fingerprint()
    assert a == b
    c = SynthConfig(n_per_category=50).fingerprint()
    assert a != c


def test_synth_config_new_fields():
    s = SynthConfig()
    assert s.llm_n_threads is None
    assert s.llm_n_batch == 512
    assert s.llm_n_gpu_layers is None


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


def test_auto_n_threads_returns_positive():
    n = auto_n_threads()
    assert n >= 1
    assert n <= 8


def test_auto_n_threads_env_override(monkeypatch):
    monkeypatch.setenv("PAWPAW_N_THREADS", "4")
    assert auto_n_threads() == 4
