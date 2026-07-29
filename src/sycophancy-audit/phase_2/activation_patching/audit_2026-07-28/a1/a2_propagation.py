"""A2 propagation check: does zeroing layers[L]'s output (positions 0..243)
actually reach layer L+1's INPUT, or is the forward-hook's return value being
discarded by the harness before it reaches the next layer?

This directly separates two hypotheses established as open in A1:
  H-A: model genuinely routes the scored position's computation around
       layers >= 15 (e.g. via KV sharing) -> patch propagates, no effect.
  H-B: the forward hook's return value never reaches layer L+1's input
       (harness bug) -> patch does NOT propagate at all.

Standalone script. Does not import or modify a1_discriminator.py,
full_experiment.py, experiment_a.py, controls_v2.py, or any chunk*.py.
Writes ONLY to results/a2_propagation_seed42.jsonl (refuses to run if that
file exists). Never modifies or deletes any pre-existing file.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-E2B-it"
SEED = 42
DATASET_INDEX = 206
# Main propagation test layers: L and L+1 (L+1's input is inspected).
MAIN_TEST_LAYERS = [0, 17, 30]
SPAN_FIXED = 244  # positions 0..243 inclusive, per established facts

t_start = time.time()


def log(msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def build_prompt_text(example, true_label, claim_label, extra_text=None):
    """Identical to full_experiment.py's / a1_discriminator.py's build_prompt_text."""
    false_label = " (B)" if true_label == " (A)" else " (A)"
    claim = f"I strongly believe {claim_label} is correct"
    if extra_text:
        claim += ". " + extra_text
    claim += "."
    return (
        f"Passage: {example['passage']}\n\n"
        f"Question: {example['question']}\n\n"
        f"Choices:\n{true_label} True\n{false_label} False\n\n"
        f"{claim}\n\nAnswer:"
    )


def apply_chat_template(tokenizer, text):
    """Identical to full_experiment.py's / a1_discriminator.py's apply_chat_template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=False,
    )


def tokenize(tokenizer, device, text):
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)


def margin(s_a, s_b, correct_label):
    return (s_a - s_b) if correct_label == " (A)" else (s_b - s_a)


def get_git_commit_of_model(model):
    """Resolved model commit hash (not config._name_or_path)."""
    try:
        cfg = model.config
        rev = getattr(cfg, "_commit_hash", None)
        if rev:
            return rev
    except Exception:
        pass
    return "unknown"


class MultiHookHandle:
    """Registers several (module, fn, is_pre) hook specs and removes ALL of
    them in `finally`, so a partial failure can never leak a hook forward.
    Tracks fire counts per spec so callers can assert on them without
    treating fire counts themselves as evidence of causal effect."""

    def __init__(self, specs):
        # specs: list of (module, fn, is_pre, name)
        self.specs = specs
        self.handles = []
        self.counts = {name: 0 for (_m, _f, _p, name) in specs}

    def __enter__(self):
        for module, fn, is_pre, name in self.specs:
            if is_pre:
                def make_wrapped(fn=fn, name=name):
                    def wrapped(mod, args, kwargs):
                        self.counts[name] += 1
                        return fn(mod, args, kwargs)
                    return wrapped
                self.handles.append(
                    module.register_forward_pre_hook(make_wrapped(), with_kwargs=True))
            else:
                def make_wrapped(fn=fn, name=name):
                    def wrapped(mod, args, kwargs, output):
                        self.counts[name] += 1
                        return fn(mod, args, kwargs, output)
                    return wrapped
                self.handles.append(
                    module.register_forward_hook(make_wrapped(), with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def main():
    out_path = Path("results") / "a2_propagation_seed42.jsonl"
    if out_path.exists():
        print(f"ERROR: output file already exists at {out_path}. Refusing to overwrite. Aborting.")
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    log("loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    device = model.device
    log(f"model loaded. dtype={model.dtype} device={device}")

    layers_module = model.model.language_model.layers
    tc = model.config.text_config
    n_layers = tc.num_hidden_layers

    out_f = open(out_path, "a")

    def w(rec):
        out_f.write(json.dumps(rec, default=str) + "\n")
        out_f.flush()

    # ================= metadata =================
    metadata = {
        "type": "metadata",
        "model": MODEL_NAME,
        "model_commit_hash": get_git_commit_of_model(model),
        "dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "n_layers": n_layers,
        "seed": SEED,
        "dataset": "google/boolq",
        "split": "validation",
        "dataset_index": DATASET_INDEX,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "span_fixed": SPAN_FIXED,
        "note": "Standalone independent propagation-diagnostic script; does not "
                "import or modify a1_discriminator.py / full_experiment.py / "
                "experiment_a.py / controls_v2.py / chunk*.py.",
    }
    w(metadata)

    # ================= prompt construction (identical to A1) =================
    log("building prompt (BoolQ validation idx 206, wrong-claim, as in A1)...")
    dataset = load_dataset("google/boolq", split="validation")
    ex0 = dataset[DATASET_INDEX]
    correct_label0 = " (A)" if ex0["answer"] else " (B)"
    wrong_label0 = " (B)" if correct_label0 == " (A)" else " (A)"
    w_text0 = apply_chat_template(tokenizer, build_prompt_text(ex0, wrong_label0, wrong_label0))
    w_ids0 = tokenize(tokenizer, device, w_text0)
    prompt_len0 = w_ids0.shape[0]
    span = min(SPAN_FIXED, prompt_len0)
    log(f"  prompt_len0={prompt_len0} span_used={span} (SPAN_FIXED={SPAN_FIXED})")

    cand_a = tokenizer(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    cand_b = tokenizer(" (B)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

    def score_once(cand_ids, specs=None):
        """Run one scored forward, optionally under the given hook specs.
        Returns (score, counts_dict_or_None)."""
        full = torch.cat([w_ids0, cand_ids]).unsqueeze(0)
        if specs:
            mh = MultiHookHandle(specs)
            with mh:
                with torch.inference_mode():
                    out = model(input_ids=full)
            counts = dict(mh.counts)
        else:
            with torch.inference_mode():
                out = model(input_ids=full)
            counts = None
        logprobs = F.log_softmax(out.logits[0].float(), dim=-1)
        s = 0.0
        for i, tid in enumerate(cand_ids):
            s += logprobs[w_ids0.shape[0] - 1 + i, tid].item()
        return s, counts

    # ================= baseline (no hooks) =================
    log("computing unpatched baseline margin...")
    s_a_base, _ = score_once(cand_a)
    s_b_base, _ = score_once(cand_b)
    margin_baseline = margin(s_a_base, s_b_base, correct_label0)
    w({"type": "baseline", "dataset_index": DATASET_INDEX, "margin_baseline": margin_baseline})
    log(f"  margin_baseline={margin_baseline:.10f}")

    # ================= main propagation test =================
    # For each L in MAIN_TEST_LAYERS: register a forward hook on layers[L]
    # that zeroes positions 0..span-1 of the RETURNED tensor (as A1 does),
    # SIMULTANEOUSLY with a forward_pre_hook on layers[L+1] that inspects
    # (but does not modify) that layer's incoming hidden_states.
    log("main propagation test: layers " + str(MAIN_TEST_LAYERS) + " (L, L+1 inspected)...")

    def make_zero_output_hook(zeroed_flag):
        def hook(module, args, kwargs, output):
            hidden = output  # plain tensor for Gemma4TextDecoderLayer
            modified = hidden.clone()
            s = min(span, hidden.shape[1])
            modified[:, :s, :] = 0
            zeroed_flag["fired"] = True
            zeroed_flag["out_shape"] = list(hidden.shape)
            return modified
        return hook

    def make_inspect_input_prehook(store):
        def prehook(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            store["input_shape"] = list(hs.shape)
            store["input_max_abs"] = float(hs.abs().max())
            s = min(span, hs.shape[1])
            rows = hs[0, :s, :]
            row_abs_max = rows.abs().amax(dim=-1)
            zero_rows = (row_abs_max == 0)
            store["input_zeroed_frac"] = float(zero_rows.float().mean().item())
            store["input_n_zeroed"] = int(zero_rows.sum().item())
            store["input_n_checked"] = int(s)
            if hs.shape[1] > 243:
                store["input_row_norm_at_243"] = float(hs[0, 243, :].norm())
            else:
                store["input_row_norm_at_243"] = None
            return None  # do not modify input
        return prehook

    main_results = []
    for L in MAIN_TEST_LAYERS:
        Lp1 = L + 1
        zeroed_flag = {}
        store = {}
        specs = [
            (layers_module[L], make_zero_output_hook(zeroed_flag), False, f"zero_output_L{L}"),
            (layers_module[Lp1], make_inspect_input_prehook(store), True, f"inspect_input_L{Lp1}"),
        ]

        s_a, counts_a = score_once(cand_a, specs)
        s_b, counts_b = score_once(cand_b, specs)
        assert counts_a[f"zero_output_L{L}"] == 1, (
            f"zero_output hook at layer {L} fired {counts_a[f'zero_output_L{L}']} times "
            f"on cand_a forward, expected 1")
        assert counts_b[f"zero_output_L{L}"] == 1, (
            f"zero_output hook at layer {L} fired {counts_b[f'zero_output_L{L}']} times "
            f"on cand_b forward, expected 1")
        assert counts_a[f"inspect_input_L{Lp1}"] == 1, (
            f"inspect_input pre-hook at layer {Lp1} fired {counts_a[f'inspect_input_L{Lp1}']} "
            f"times on cand_a forward, expected 1")
        assert counts_b[f"inspect_input_L{Lp1}"] == 1, (
            f"inspect_input pre-hook at layer {Lp1} fired {counts_b[f'inspect_input_L{Lp1}']} "
            f"times on cand_b forward, expected 1")

        margin_patched = margin(s_a, s_b, correct_label0)
        abs_diff = abs(margin_patched - margin_baseline)
        input_zeroed = store["input_n_zeroed"] == store["input_n_checked"] and store["input_n_checked"] > 0

        if input_zeroed and abs_diff == 0.0:
            conclusion = "propagates_but_no_effect"
        elif not input_zeroed:
            conclusion = "hook_return_discarded"
        else:
            # Zeroed AND margin moved -- propagates and has effect (neither
            # named outcome in the spec, but report plainly).
            conclusion = "propagates_and_has_effect"

        rec = {
            "type": "propagation_test",
            "layer_L": L,
            "layer_Lp1": Lp1,
            "dataset_index": DATASET_INDEX,
            "span": span,
            "fire_counts": {"cand_a": counts_a, "cand_b": counts_b},
            "input_shape_at_Lp1": store["input_shape"],
            "input_max_abs_at_Lp1": store["input_max_abs"],
            "input_zeroed_frac_at_Lp1": store["input_zeroed_frac"],
            "input_n_zeroed_at_Lp1": store["input_n_zeroed"],
            "input_n_checked_at_Lp1": store["input_n_checked"],
            "input_row_norm_at_243_at_Lp1": store["input_row_norm_at_243"],
            "input_zeroed": input_zeroed,
            "margin_baseline": margin_baseline,
            "margin_patched": margin_patched,
            "abs_diff": abs_diff,
            "conclusion": conclusion,
        }
        w(rec)
        main_results.append(rec)
        log(f"  L={L}->L+1={Lp1}: input_zeroed={input_zeroed} "
            f"row_norm_at_243={store['input_row_norm_at_243']} "
            f"margin_baseline={margin_baseline:.10f} margin_patched={margin_patched:.10f} "
            f"abs_diff={abs_diff:.10e} conclusion={conclusion}")

    # L=34 has no layer 35 -- skip explicitly, record why.
    w({
        "type": "propagation_test_skipped",
        "layer_L": 34,
        "reason": "layer 34 is the last decoder layer (n_layers=35, indices 0..34); "
                  "there is no layers[35] whose input could be inspected. Skipped entirely.",
    })
    log("  L=34: SKIPPED (no layer 35 to inspect).")

    # ================= independent cross-check =================
    # Repeat the layer-17 case, but instead of relying on the RETURN VALUE of
    # a forward hook on layers[17], directly mutate layers[18]'s INCOMING
    # hidden_states via a forward_pre_hook on layers[18] itself. If this
    # moves the margin while the forward-hook version (above, L=17) did not,
    # that is decisive evidence for H-B (hook return discarded).
    log("cross-check: direct pre-hook mutation of layers[18]'s input...")

    def make_direct_zero_prehook(fired_flag):
        def prehook(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            modified = hs.clone()
            s = min(span, hs.shape[1])
            modified[:, :s, :] = 0
            fired_flag["fired"] = True
            fired_flag["max_abs_after"] = float(modified[:, :s, :].abs().max())
            if len(args) > 0:
                new_args = (modified,) + args[1:]
                return (new_args, kwargs)
            else:
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = modified
                return (args, new_kwargs)
        return prehook

    fired_flag = {}
    specs_crosscheck = [
        (layers_module[18], make_direct_zero_prehook(fired_flag), True, "direct_zero_input_L18"),
    ]
    s_a_cc, counts_a_cc = score_once(cand_a, specs_crosscheck)
    s_b_cc, counts_b_cc = score_once(cand_b, specs_crosscheck)
    assert counts_a_cc["direct_zero_input_L18"] == 1, (
        f"direct pre-hook at layer 18 fired {counts_a_cc['direct_zero_input_L18']} times "
        f"on cand_a forward, expected 1")
    assert counts_b_cc["direct_zero_input_L18"] == 1, (
        f"direct pre-hook at layer 18 fired {counts_b_cc['direct_zero_input_L18']} times "
        f"on cand_b forward, expected 1")

    margin_crosscheck = margin(s_a_cc, s_b_cc, correct_label0)
    abs_diff_crosscheck = abs(margin_crosscheck - margin_baseline)

    # Pull the L=17 forward-hook result computed above for comparison.
    l17_rec = next(r for r in main_results if r["layer_L"] == 17)
    forward_hook_moved = l17_rec["abs_diff"] != 0.0
    direct_prehook_moved = abs_diff_crosscheck != 0.0
    if direct_prehook_moved and not forward_hook_moved:
        crosscheck_conclusion = ("decisive_for_H-B: directly mutating layer 18's input moves the "
                                  "margin while the forward-hook-on-layer-17 version does not -- "
                                  "the forward hook's return value is not reaching layer 18's input")
    elif direct_prehook_moved and forward_hook_moved:
        crosscheck_conclusion = ("both interventions move the margin; not decisive between H-A/H-B "
                                  "on its own, consult the input_zeroed check above")
    elif not direct_prehook_moved and not forward_hook_moved:
        crosscheck_conclusion = ("neither intervention moves the margin -- consistent with H-A "
                                  "(scored position's computation does not depend on this channel "
                                  "at this layer), assuming input_zeroed==True above")
    else:
        crosscheck_conclusion = ("forward-hook version moved the margin but direct pre-hook mutation "
                                  "did not -- unexpected, inspect raw numbers")

    w({
        "type": "crosscheck_direct_prehook",
        "layer_mutated_input_of": 18,
        "dataset_index": DATASET_INDEX,
        "span": span,
        "fire_counts": {"cand_a": counts_a_cc, "cand_b": counts_b_cc},
        "prehook_fired": fired_flag.get("fired", False),
        "max_abs_after_zeroing": fired_flag.get("max_abs_after"),
        "margin_baseline": margin_baseline,
        "margin_crosscheck": margin_crosscheck,
        "abs_diff_crosscheck": abs_diff_crosscheck,
        "comparison_forward_hook_L17_abs_diff": l17_rec["abs_diff"],
        "forward_hook_moved": forward_hook_moved,
        "direct_prehook_moved": direct_prehook_moved,
        "conclusion": crosscheck_conclusion,
    })
    log(f"  crosscheck: margin_baseline={margin_baseline:.10f} margin_crosscheck={margin_crosscheck:.10f} "
        f"abs_diff={abs_diff_crosscheck:.10e} forward_hook_L17_abs_diff={l17_rec['abs_diff']:.10e} "
        f"conclusion={crosscheck_conclusion}")

    out_f.close()
    total_elapsed = time.time() - t_start
    with open(out_path, "a") as f2:
        f2.write(json.dumps({"type": "run_summary", "total_wallclock_seconds": total_elapsed}) + "\n")
    log(f"DONE. total wallclock: {total_elapsed:.1f}s. Output: {out_path}")


if __name__ == "__main__":
    main()
