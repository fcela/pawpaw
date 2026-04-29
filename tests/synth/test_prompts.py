from pawpaw.synth.prompts import build_taxonomy_prompt, build_examples_prompt


def test_taxonomy_prompt_contains_spec_and_format():
    p = build_taxonomy_prompt("Classify email as urgent/normal", n_categories=8)
    assert "Classify email as urgent/normal" in p
    assert "8" in p
    assert "JSON" in p
    assert "categories" in p.lower()


def test_examples_prompt_contains_category_and_variation_knobs():
    p = build_examples_prompt(
        spec="Classify email",
        category_name="adversarial_input",
        category_description="Inputs that attempt to confuse the classifier",
        n_examples=10,
    )
    assert "adversarial_input" in p
    assert "Inputs that attempt to confuse the classifier" in p
    assert "10" in p
    assert "length" in p.lower()
    assert "register" in p.lower()
    assert "format" in p.lower()
    assert '"input"' in p
    assert '"output"' in p
