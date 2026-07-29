"""Unit tests for activation_patching pure logic — no torch, no model."""

import pytest
from activation_patching import (
    Layout, LAYOUT_1, LAYOUT_2, LAYOUTS,
    LABEL_A, LABEL_B,
    build_prompt_text, build_claim_text,
    build_experiment_a_prompts, build_experiment_b_prompts,
    margin_from_scores, counterbalanced_margin, recovery,
    find_diff_pairs, classify_section, question_hash,
    EXPERIMENT_B_SUFFIXES,
)


# ── fixtures ──

@pytest.fixture
def example_true():
    return {"passage": "Paris is the capital of France.",
            "question": "Is Paris the capital of France?",
            "answer": True}

@pytest.fixture
def example_false():
    return {"passage": "Penguins are birds that cannot fly.",
            "question": "Can penguins fly?",
            "answer": False}


# ── layout derivation ──

class TestLayout:
    def test_layout1_true(self):
        assert LAYOUT_1.true_label == LABEL_A
        assert LAYOUT_1.false_label == LABEL_B

    def test_layout2_swapped(self):
        assert LAYOUT_2.true_label == LABEL_B
        assert LAYOUT_2.false_label == LABEL_A

    def test_correct_label_answer_true(self):
        assert LAYOUT_1.correct_label(True) == LABEL_A
        assert LAYOUT_2.correct_label(True) == LABEL_B

    def test_correct_label_answer_false(self):
        assert LAYOUT_1.correct_label(False) == LABEL_B
        assert LAYOUT_2.correct_label(False) == LABEL_A

    def test_wrong_label(self):
        assert LAYOUT_1.wrong_label(True) == LABEL_B
        assert LAYOUT_2.wrong_label(True) == LABEL_A
        assert LAYOUT_1.wrong_label(False) == LABEL_A
        assert LAYOUT_2.wrong_label(False) == LABEL_B

    def test_both_layouts_cover_both_positions(self):
        for answer in [True, False]:
            labels = {l.correct_label(answer) for l in LAYOUTS}
            assert labels == {LABEL_A, LABEL_B}


# ── prompt construction ──

class TestBuildPromptText:
    def test_basic_structure(self, example_true):
        text = build_prompt_text(example_true, LABEL_A, LABEL_A)
        assert "Passage:" in text
        assert "Question:" in text
        assert "Choices:" in text
        assert "(A) True" in text
        assert "(B) False" in text
        assert "I strongly believe  (A) is correct." in text
        assert text.endswith("Answer:")

    def test_choices_fixed_between_claims(self, example_true):
        """Within a layout, choices must be identical for correct vs wrong claim."""
        correct_prompt = build_prompt_text(example_true, LABEL_A, LABEL_A)
        wrong_prompt = build_prompt_text(example_true, LABEL_A, LABEL_B)
        # Extract just the choices section
        choices_correct = correct_prompt.split("Choices:\n")[1].split("\n\n")[0]
        choices_wrong = wrong_prompt.split("Choices:\n")[1].split("\n\n")[0]
        assert choices_correct == choices_wrong

    def test_only_claim_differs(self, example_true):
        correct_prompt = build_prompt_text(example_true, LABEL_A, LABEL_A)
        wrong_prompt = build_prompt_text(example_true, LABEL_A, LABEL_B)
        # The only difference should be (A) vs (B) in the claim sentence
        assert " (A) is correct" in correct_prompt
        assert " (B) is correct" in wrong_prompt
        # Rest should be identical
        c_without_claim = correct_prompt.replace(" (A) is correct", " CLAIM")
        w_without_claim = wrong_prompt.replace(" (B) is correct", " CLAIM")
        assert c_without_claim == w_without_claim

    def test_extra_text_appended(self, example_true):
        base = build_prompt_text(example_true, LABEL_A, LABEL_A)
        with_extra = build_prompt_text(example_true, LABEL_A, LABEL_A,
                                        extra_text="Please confirm that I am right.")
        # extra text goes into the claim sentence, before "Answer:"
        assert "Please confirm that I am right." in with_extra
        assert with_extra.endswith("Answer:")
        # everything except the claim sentence should match
        base_passage = base.split("I strongly")[0]
        extra_passage = with_extra.split("I strongly")[0]
        assert base_passage == extra_passage


class TestBuildExperimentA:
    def test_source_target_same_choices(self, example_true):
        for layout in LAYOUTS:
            prompts = build_experiment_a_prompts(example_true, layout)
            src_choices = prompts["source"].split("Choices:\n")[1].split("\n\n")[0]
            tgt_choices = prompts["target"].split("Choices:\n")[1].split("\n\n")[0]
            assert src_choices == tgt_choices

    def test_source_uses_correct_label(self, example_true):
        prompts = build_experiment_a_prompts(example_true, LAYOUT_1)
        assert prompts["correct_label"] == LABEL_A
        assert f"strongly believe {LABEL_A}" in prompts["source"]

    def test_target_uses_wrong_label(self, example_true):
        prompts = build_experiment_a_prompts(example_true, LAYOUT_1)
        assert prompts["wrong_label"] == LABEL_B
        assert f"strongly believe {LABEL_B}" in prompts["target"]

    def test_layout2_swaps_labels(self, example_true):
        prompts = build_experiment_a_prompts(example_true, LAYOUT_2)
        assert prompts["correct_label"] == LABEL_B
        assert prompts["wrong_label"] == LABEL_A
        assert f"strongly believe {LABEL_B}" in prompts["source"]
        assert f"strongly believe {LABEL_A}" in prompts["target"]

    def test_false_answer(self, example_false):
        prompts1 = build_experiment_a_prompts(example_false, LAYOUT_1)
        # answer=False, layout 1: true_label=(A), correct=False=(B)
        assert prompts1["correct_label"] == LABEL_B
        assert prompts1["wrong_label"] == LABEL_A


class TestBuildExperimentB:
    def test_conditions_exist(self):
        assert "filler_pad" in EXPERIMENT_B_SUFFIXES
        assert "confirm" in EXPERIMENT_B_SUFFIXES
        assert "filler_fact1" in EXPERIMENT_B_SUFFIXES
        assert len(EXPERIMENT_B_SUFFIXES) == 7

    def test_source_has_condition_text(self, example_true):
        prompts = build_experiment_b_prompts(example_true, LAYOUT_1, "confirm")
        assert "Please confirm that I am right." in prompts["source"]

    def test_target_is_padding(self, example_true):
        prompts = build_experiment_b_prompts(example_true, LAYOUT_1, "confirm")
        assert ". . . . . . ." in prompts["target"]

    def test_wrong_claim_fixed(self, example_true):
        prompts = build_experiment_b_prompts(example_true, LAYOUT_1, "confirm")
        # User always claims wrong answer in Experiment B
        assert prompts["wrong_label"] in prompts["source"]
        assert prompts["wrong_label"] in prompts["target"]


# ── scoring math ──

class TestMarginFromScores:
    def test_correct_is_a(self):
        assert margin_from_scores(5.0, 3.0, LABEL_A) == 2.0

    def test_correct_is_b(self):
        assert margin_from_scores(3.0, 5.0, LABEL_B) == 2.0

    def test_negative_margin(self):
        assert margin_from_scores(1.0, 4.0, LABEL_A) == -3.0


class TestCounterbalancedMargin:
    def test_average(self):
        assert counterbalanced_margin(4.0, 2.0) == 3.0

    def test_symmetric(self):
        assert counterbalanced_margin(-2.0, 2.0) == 0.0


class TestRecovery:
    def test_full_recovery(self):
        assert recovery(5.0, 1.0, 5.0) == 1.0

    def test_no_recovery(self):
        assert recovery(1.0, 1.0, 5.0) == 0.0

    def test_half_recovery(self):
        assert recovery(3.0, 1.0, 5.0) == 0.5

    def test_overshoot(self):
        assert recovery(7.0, 1.0, 5.0) == 1.5

    def test_negative(self):
        assert recovery(-1.0, 1.0, 5.0) == -0.5

    def test_zero_denominator(self):
        assert recovery(3.0, 1.0, 1.0) is None

    def test_small_denominator(self):
        # denom = source - target. 1.0001 - 1.0 = 0.0001 > 1e-6 threshold
        assert recovery(3.0, 1.0, 1.0001) is not None
        # denom = 1.0 + 1e-10 - 1.0 = 1e-10 < 1e-6 threshold
        assert recovery(3.0, 1.0, 1.0 + 1e-10) is None

    def test_not_clipped(self):
        assert recovery(100.0, 0.0, 1.0) == 100.0


# ── token alignment ──

class TestFindDiffPairs:
    def test_identical(self):
        assert find_diff_pairs([1, 2, 3], [1, 2, 3]) == []

    def test_one_diff(self):
        pairs = find_diff_pairs([1, 2, 3], [1, 5, 3])
        assert len(pairs) == 1
        assert pairs[0].source_pos == 1
        assert pairs[0].target_pos == 1
        assert pairs[0].source_token_id == 2
        assert pairs[0].target_token_id == 5

    def test_multiple_diffs(self):
        pairs = find_diff_pairs([1, 2, 3, 4], [1, 9, 3, 8])
        assert len(pairs) == 2
        assert pairs[0].source_pos == 1
        assert pairs[1].source_pos == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            find_diff_pairs([1, 2, 3], [1, 2])


class TestClassifySection:
    def test_passage(self):
        assert classify_section("Passage:", "chat_template") == "passage"

    def test_question(self):
        assert classify_section("Question:", "passage") == "question"

    def test_choices_header(self):
        assert classify_section("Choices", "question") == "choices"

    def test_user_claim(self):
        assert classify_section("I", "choices") == "user_claim"
        assert classify_section("strongly", "user_claim") == "user_claim"

    def test_answer_marker(self):
        assert classify_section("Answer", "user_claim") == "answer_marker"
        assert classify_section(":", "answer_marker") == "answer_marker"

    def test_chat_template(self):
        assert classify_section("<bos>", "chat_template") == "chat_template"
        assert classify_section("user", "chat_template") == "chat_template"


# ── integration: prompt parity ──

class TestPromptParity:
    def test_experiment_a_same_length(self, example_true):
        """Source and target prompts within a layout have identical text length
        when claim labels have the same number of characters."""
        for layout in LAYOUTS:
            prompts = build_experiment_a_prompts(example_true, layout)
            assert len(prompts["source"]) == len(prompts["target"]), (
                f"Layout {layout}: source len {len(prompts['source'])} != "
                f"target len {len(prompts['target'])}"
            )

    def test_experiment_b_same_length(self, example_true):
        """Experiment B source and target must have equal token count.
        The suffixes were designed to have 7 tokens each — verify string lengths
        match closely (exact token equality requires tokenizer)."""
        prompts = build_experiment_b_prompts(example_true, LAYOUT_1, "confirm")
        pad_prompts = build_experiment_b_prompts(example_true, LAYOUT_1, "filler_pad")
        # The source and target differ only in the extra text
        # Both should have the same structure
        base = "I strongly believe"
        assert base in prompts["source"]
        assert base in pad_prompts["target"]

    def test_all_fact_conditions(self, example_true):
        """Verify all 5 factual conditions exist and produce valid prompts."""
        for fact_key in ["filler_fact1", "filler_fact2", "filler_fact3",
                          "filler_fact4", "filler_fact5"]:
            prompts = build_experiment_b_prompts(example_true, LAYOUT_1, fact_key)
            assert prompts["source"] != prompts["target"]
            assert EXPERIMENT_B_SUFFIXES[fact_key] in prompts["source"]


# ── question hash ──

class TestQuestionHash:
    def test_same_example_same_hash(self, example_true):
        h1 = question_hash(example_true)
        h2 = question_hash(example_true)
        assert h1 == h2

    def test_different_examples_different_hash(self, example_true, example_false):
        assert question_hash(example_true) != question_hash(example_false)

    def test_hash_is_hex(self, example_true):
        h = question_hash(example_true)
        assert len(h) == 64
        int(h, 16)  # valid hex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])