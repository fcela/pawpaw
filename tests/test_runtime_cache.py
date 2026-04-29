from __future__ import annotations

from pawpaw.runtime_cache import _effective_quant, set_preferred_quant, _AVAILABLE_QUANTS


def test_default_quant_is_q6k():
    assert _effective_quant() == "Q6_K"


def test_set_preferred_quant():
    set_preferred_quant("Q4_K_M")
    assert _effective_quant() == "Q4_K_M"
    set_preferred_quant("Q6_K")


def test_set_preferred_quant_rejects_invalid():
    import pytest
    with pytest.raises(ValueError, match="unknown quant"):
        set_preferred_quant("Q3_K_S")


def test_env_overrides_preferred(monkeypatch):
    set_preferred_quant("Q4_K_M")
    monkeypatch.setenv("PAWPAW_BASE_QUANT", "Q8_0")
    assert _effective_quant() == "Q8_0"
    set_preferred_quant("Q6_K")


def test_available_quants():
    assert "Q4_K_M" in _AVAILABLE_QUANTS
    assert "Q6_K" in _AVAILABLE_QUANTS
    assert "Q8_0" in _AVAILABLE_QUANTS
