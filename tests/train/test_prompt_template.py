from pawpaw.train.prompt_template import (
    INPUT_PLACEHOLDER,
    build_prompt_template,
    render_for_training,
    split_template,
)
from pawpaw.synth.examples import Pair


def _p(i, o):
    return Pair(input=i, output=o, category="c", length_bucket="short")


def test_build_prompt_template_contains_spec_demos_placeholder():
    t = build_prompt_template("Classify text", demos=[_p("hi", "greeting"), _p("bye", "farewell")])
    assert "Classify text" in t
    assert "hi" in t
    assert "greeting" in t
    assert INPUT_PLACEHOLDER in t


def test_split_template_returns_prefix_and_suffix():
    t = build_prompt_template("spec", demos=[_p("a", "b")])
    prefix, suffix = split_template(t)
    assert INPUT_PLACEHOLDER not in prefix
    assert INPUT_PLACEHOLDER not in suffix
    assert prefix + INPUT_PLACEHOLDER + suffix == t


def test_render_for_training_substitutes_input_and_appends_output():
    t = build_prompt_template("spec", demos=[_p("a", "b")])
    rendered = render_for_training(t, _p("user input", "expected output"))
    assert "user input" in rendered
    assert rendered.endswith("expected output")
    assert INPUT_PLACEHOLDER not in rendered
