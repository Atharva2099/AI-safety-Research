"""Independent control battery for the activation-patching correctness audit.

This is a STANDALONE script. It does not import or execute any code from
full_experiment.py / experiment_a.py / chunk*.py. It reimplements model
loading, prompt construction, and scoring from scratch (matching the same
model/revision/prompt format/scoring method) so the checks below are a
genuinely independent verification of the original result, not a rerun of
the same code path (and therefore the same possible bug).

Tests:
  T1  - self-patch identity (all prompt positions, same tensor as source/target)
  T2  - POSITIVE CONTROL: full-sequence patch (prompt+candidate) source==target env
  T2b - prompt-only full patch (candidate positions left unpatched)
  T3  - NEGATIVE CONTROL, corrected: cross-example patch, asserted non-empty slice
  T4  - regression test reproducing the original empty-slice defect
  T5  - single-position sweep at layer 17 for example 206
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-E2B-it"
TOLERANCE = 1e-3

t_start = time.time()


def log(msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def build_prompt_text(example, true_label, claim_label, extra_text=None):
    """Identical to full_experiment.py's build_prompt_text."""
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
    """Identical to full_experiment.py's apply_chat_template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=False,
    )


def tokenize(tokenizer, device, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    return ids


def candidate_logprob(model, prompt_ids, cand_ids):
    """Identical scoring method to full_experiment.py's candidate_logprob."""
    full = torch.cat([prompt_ids, cand_ids]).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=full)
    logprobs = F.log_softmax(out.logits[0], dim=-1)
    s = 0.0
    for i, tid in enumerate(cand_ids):
        s += logprobs[len(prompt_ids) - 1 + i, tid].item()
    return s


def margin(s_a, s_b, correct_label):
    return (s_a - s_b) if correct_label == " (A)" else (s_b - s_a)


class HookHandle:
    """Context manager that registers a hook and asserts it fired, always
    removing the hook in a finally block so a failure can't leak a hook
    into a later test."""

    def __init__(self, module, fn, name):
        self.module = module
        self.fn = fn
        self.name = name
        self.count = 0
        self.handle = None

    def __enter__(self):
        def wrapped(mod, args, kwargs, output):
            self.count += 1
            return self.fn(mod, args, kwargs, output)
        self.handle = self.module.register_forward_hook(wrapped, with_kwargs=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        # Only assert firing if we're not already unwinding an exception.
        if exc_type is None and self.count == 0:
            raise RuntimeError(f"Hook '{self.name}' registered but never fired "
                                f"(silent non-execution) — aborting.")
        return False


def cache_layer_output(model, layers_module, layer, input_ids, positions=None):
    """Run a forward pass, caching the layer's output at `positions`
    (a slice / list of ints), or the FULL sequence if positions is None.
    Returns the cached activation tensor (batch=1, seq, hidden)."""
    cache = {}

    def hook_fn(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if positions is None:
            cache["act"] = hidden.clone()
        else:
            cache["act"] = hidden[:, positions, :].clone()
        return output

    with HookHandle(layers_module[layer], hook_fn, f"cache_layer{layer}") as h:
        with torch.inference_mode():
            _ = model(input_ids=input_ids.unsqueeze(0))
    return cache["act"], h.count


def run_patched_score(model, layers_module, layer, target_ids, cand_ids,
                      patch_act, patch_slice):
    """Run target_ids+cand_ids through the model, patching layer's output
    at patch_slice with patch_act. Returns (score, hook_fire_count)."""

    def patch_hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden.clone()
        modified[:, patch_slice, :] = patch_act
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    full = torch.cat([target_ids, cand_ids]).unsqueeze(0)
    with HookHandle(layers_module[layer], patch_hook, f"patch_layer{layer}") as h:
        with torch.inference_mode():
            out = model(input_ids=full)
    logprobs = F.log_softmax(out.logits[0], dim=-1)
    s = 0.0
    for i, tid in enumerate(cand_ids):
        s += logprobs[len(target_ids) - 1 + i, tid].item()
    return s, h.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        print(f"ERROR: output file already exists at {args.output}. Refusing to overwrite. Aborting.")
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    log("loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    device = model.device
    layers_module = model.model.language_model.layers
    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    hidden_dim = tc.hidden_size
    log(f"model loaded. n_layers={n_layers} hidden_dim={hidden_dim} device={device} dtype={model.dtype}")

    cand_a = tokenizer(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    cand_b = tokenizer(" (B)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

    log("loading dataset google/boolq validation split...")
    dataset = load_dataset("google/boolq", split="validation")
    log(f"dataset loaded, n={len(dataset)}")

    out_f = open(args.output, "a")

    def w(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    metadata = {
        "type": "metadata",
        "model": MODEL_NAME,
        "model_revision": getattr(model.config, "_name_or_path", MODEL_NAME),
        "dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "seed": args.seed,
        "dataset": "google/boolq",
        "split": "validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hook_point": "model.model.language_model.layers[N] output",
        "tolerance": TOLERANCE,
        "note": "Standalone independent reimplementation; does not import full_experiment.py.",
    }
    w(metadata)

    layers_to_test = sorted(set([0, 17, n_layers - 1]))
    dataset_indices = [206, 1144, 3036]
    n_ds = len(dataset)

    # Validate indices are usable.
    usable = {}
    for idx in dataset_indices:
        if idx < 0 or idx >= n_ds:
            log(f"WARNING: dataset index {idx} is OUT OF RANGE (dataset size={n_ds}). Marking unusable.")
            usable[idx] = False
            w({"type": "index_check", "dataset_index": idx, "usable": False,
               "reason": f"index out of range for dataset of size {n_ds}"})
        else:
            usable[idx] = True
            ex = dataset[idx]
            w({"type": "index_check", "dataset_index": idx, "usable": True,
               "question": ex["question"], "answer": bool(ex["answer"])})

    def get_prompts(idx):
        """Return (ex, correct_label, wrong_label, c_ids, w_ids)."""
        ex = dataset[idx]
        correct_label = " (A)" if ex["answer"] else " (B)"
        wrong_label = " (B)" if correct_label == " (A)" else " (A)"
        c_text = apply_chat_template(tokenizer, build_prompt_text(ex, correct_label, correct_label))
        w_text = apply_chat_template(tokenizer, build_prompt_text(ex, wrong_label, wrong_label))
        c_ids = tokenize(tokenizer, device, c_text)
        w_ids = tokenize(tokenizer, device, w_text)
        return ex, correct_label, wrong_label, c_ids, w_ids

    # ================= T1: self-patch identity =================
    log("T1: self-patch identity...")
    idx0 = 206
    if usable.get(idx0):
        ex0, correct_label0, wrong_label0, c_ids0, w_ids0 = get_prompts(idx0)
        target_ids = w_ids0  # wrong-claim prompt

        # unpatched baseline margin for target (wrong-claim prompt), single layout
        s_a_base = candidate_logprob(model, target_ids, cand_a)
        s_b_base = candidate_logprob(model, target_ids, cand_b)
        m_base = margin(s_a_base, s_b_base, correct_label0)

        for layer in layers_to_test:
            n_pos = target_ids.shape[0]
            positions = list(range(n_pos))
            # source_ids == target_ids (same tensor), cache all positions
            patch_act, fire_count_cache = cache_layer_output(
                model, layers_module, layer, target_ids, positions=positions)
            assert fire_count_cache == 1, f"T1 cache hook fired {fire_count_cache} times, expected 1"

            s_a, fc_a = run_patched_score(model, layers_module, layer, target_ids, cand_a,
                                           patch_act, slice(0, n_pos))
            s_b, fc_b = run_patched_score(model, layers_module, layer, target_ids, cand_b,
                                           patch_act, slice(0, n_pos))
            assert fc_a == 1 and fc_b == 1, "T1 patch hook fire count mismatch"
            m_patched = margin(s_a, s_b, correct_label0)
            abs_diff = abs(m_patched - m_base)
            passed = abs_diff < TOLERANCE
            w({
                "test": "T1_self_patch_identity",
                "dataset_index": idx0,
                "layer": layer,
                "positions_patched_count": n_pos,
                "positions_patched_range": [0, n_pos - 1],
                "margin_source": m_base,
                "margin_target": m_base,
                "margin_patched": m_patched,
                "recovery": None,
                "recovery_note": "undefined (denominator zero); comparing margin equality instead",
                "oracle_expected": m_base,
                "observed": m_patched,
                "abs_diff": abs_diff,
                "tolerance": TOLERANCE,
                "passed": passed,
                "hook_fire_counts": {"cache": fire_count_cache, "patch_a": fc_a, "patch_b": fc_b},
            })
            log(f"  T1 layer={layer}: baseline={m_base:.6f} patched={m_patched:.6f} abs_diff={abs_diff:.6e} passed={passed}")
    else:
        w({"test": "T1_self_patch_identity", "dataset_index": idx0, "passed": None,
           "error": "dataset index unusable"})
        log(f"T1 SKIPPED: index {idx0} unusable")

    # ================= T2: positive control (full sequence patch) =================
    log("T2: positive control (full-sequence patch, source env == target env)...")
    idx2 = 206
    if usable.get(idx2):
        ex2, correct_label2, wrong_label2, c_ids2, w_ids2 = get_prompts(idx2)

        len_match = (c_ids2.shape[0] == w_ids2.shape[0])
        if not len_match:
            w({"test": "T2_positive_control", "dataset_index": idx2,
               "aborted": True,
               "reason": f"prompt token lengths differ: correct={c_ids2.shape[0]} wrong={w_ids2.shape[0]}. "
                         f"T2 requires identical length aside from the claim-label token; aborting per spec."})
            log(f"T2 ABORTED: length mismatch correct={c_ids2.shape[0]} wrong={w_ids2.shape[0]}")
        else:
            for cand_name, cand_ids in [("A", cand_a), ("B", cand_b)]:
                for layer in layers_to_test:
                    # source run: correct-claim prompt + candidate. Cache EVERY position
                    # of the FULL sequence (prompt + candidate).
                    source_full = torch.cat([c_ids2, cand_ids])
                    prompt_len = c_ids2.shape[0]
                    full_len = source_full.shape[0]
                    patch_act, fc_cache = cache_layer_output(
                        model, layers_module, layer, source_full, positions=list(range(full_len)))
                    assert fc_cache == 1, f"T2 cache hook fired {fc_cache} times, expected 1"

                    # source margin (unpatched, correct-claim prompt + candidate context)
                    s_a_src = candidate_logprob(model, c_ids2, cand_a)
                    s_b_src = candidate_logprob(model, c_ids2, cand_b)
                    m_source = margin(s_a_src, s_b_src, correct_label2)

                    # target margin baseline (unpatched, wrong-claim prompt)
                    s_a_tgt = candidate_logprob(model, w_ids2, cand_a)
                    s_b_tgt = candidate_logprob(model, w_ids2, cand_b)
                    m_target = margin(s_a_tgt, s_b_tgt, correct_label2)

                    # target run: wrong-claim prompt + SAME candidate, patch EVERY position
                    target_full_len = w_ids2.shape[0] + cand_ids.shape[0]
                    assert target_full_len == full_len, "T2 requires equal full-sequence length"

                    def patch_hook(module, args, kwargs, output, _patch_act=patch_act):
                        hidden = output[0] if isinstance(output, tuple) else output
                        modified = hidden.clone()
                        modified[:, :, :] = _patch_act
                        return (modified, *output[1:]) if isinstance(output, tuple) else modified

                    full_target = torch.cat([w_ids2, cand_ids]).unsqueeze(0)
                    with HookHandle(layers_module[layer], patch_hook, f"T2_patch_layer{layer}") as hh:
                        with torch.inference_mode():
                            out = model(input_ids=full_target)
                    logprobs = F.log_softmax(out.logits[0], dim=-1)
                    # Score BOTH candidates under this fully-patched run's logits at the
                    # candidate position(s) -- but note: since the whole sequence
                    # (prompt+this candidate) is replaced, the logits for the OTHER
                    # candidate aren't directly available from this single run. We
                    # therefore recompute margin using per-candidate full patched runs.
                    s_this = 0.0
                    for i, tid in enumerate(cand_ids):
                        s_this += logprobs[w_ids2.shape[0] - 1 + i, tid].item()

                    # For the margin we need both s_a and s_b from patched runs where
                    # each uses ITS OWN candidate appended (matching how source was cached
                    # per-candidate too). Run the other candidate's full patch as well.
                    other_cand = cand_b if cand_name == "A" else cand_a
                    source_full_other = torch.cat([c_ids2, other_cand])
                    patch_act_other, fc_cache_other = cache_layer_output(
                        model, layers_module, layer, source_full_other,
                        positions=list(range(source_full_other.shape[0])))
                    assert fc_cache_other == 1

                    def patch_hook_other(module, args, kwargs, output, _p=patch_act_other):
                        hidden = output[0] if isinstance(output, tuple) else output
                        modified = hidden.clone()
                        modified[:, :, :] = _p
                        return (modified, *output[1:]) if isinstance(output, tuple) else modified

                    full_target_other = torch.cat([w_ids2, other_cand]).unsqueeze(0)
                    with HookHandle(layers_module[layer], patch_hook_other, f"T2_patch_other_layer{layer}") as hh2:
                        with torch.inference_mode():
                            out_other = model(input_ids=full_target_other)
                    logprobs_other = F.log_softmax(out_other.logits[0], dim=-1)
                    s_other = 0.0
                    for i, tid in enumerate(other_cand):
                        s_other += logprobs_other[w_ids2.shape[0] - 1 + i, tid].item()

                    if cand_name == "A":
                        s_a_patched, s_b_patched = s_this, s_other
                    else:
                        s_a_patched, s_b_patched = s_other, s_this
                    m_patched = margin(s_a_patched, s_b_patched, correct_label2)

                    denom = m_source  # recovery relative to source, oracle: recovery == 1.0
                    recovery = (m_patched / m_source) if abs(m_source) > 1e-9 else None
                    abs_diff = abs(m_patched - m_source)
                    passed = abs_diff < TOLERANCE

                    w({
                        "test": "T2_positive_control",
                        "dataset_index": idx2,
                        "candidate_used_for_cache_and_patch": cand_name,
                        "layer": layer,
                        "positions_patched_count": full_len,
                        "positions_patched_range": [0, full_len - 1],
                        "margin_source": m_source,
                        "margin_target": m_target,
                        "margin_patched": m_patched,
                        "recovery": recovery,
                        "oracle_expected": m_source,
                        "observed": m_patched,
                        "abs_diff": abs_diff,
                        "tolerance": TOLERANCE,
                        "passed": passed,
                        "hook_fire_counts": {"cache_main": fc_cache, "patch_main": hh.count,
                                             "cache_other": fc_cache_other, "patch_other": hh2.count},
                    })
                    log(f"  T2 layer={layer} cand={cand_name}: m_source={m_source:.6f} "
                        f"m_patched={m_patched:.6f} abs_diff={abs_diff:.6e} recovery={recovery} passed={passed}")
                    if not passed:
                        log(f"  *** T2 FAILURE at layer={layer} cand={cand_name}: "
                            f"positive control did not reproduce source margin. "
                            f"abs_diff={abs_diff:.6e} (tolerance={TOLERANCE}) ***")
    else:
        w({"test": "T2_positive_control", "dataset_index": idx2, "passed": None,
           "error": "dataset index unusable"})
        log(f"T2 SKIPPED: index {idx2} unusable")

    # ================= T2b: prompt-only full patch =================
    log("T2b: prompt-only full patch...")
    if usable.get(idx2) and len_match:
        prompt_len = c_ids2.shape[0]
        for cand_name, cand_ids in [("A", cand_a), ("B", cand_b)]:
            for layer in layers_to_test:
                patch_act, fc_cache = cache_layer_output(
                    model, layers_module, layer, c_ids2, positions=list(range(prompt_len)))
                assert fc_cache == 1

                s_a_src = candidate_logprob(model, c_ids2, cand_a)
                s_b_src = candidate_logprob(model, c_ids2, cand_b)
                m_source = margin(s_a_src, s_b_src, correct_label2)

                def patch_hook(module, args, kwargs, output, _p=patch_act, _n=prompt_len):
                    hidden = output[0] if isinstance(output, tuple) else output
                    modified = hidden.clone()
                    modified[:, :_n, :] = _p
                    return (modified, *output[1:]) if isinstance(output, tuple) else modified

                full_target = torch.cat([w_ids2, cand_ids]).unsqueeze(0)
                with HookHandle(layers_module[layer], patch_hook, f"T2b_patch_layer{layer}") as hh:
                    with torch.inference_mode():
                        out = model(input_ids=full_target)
                logprobs = F.log_softmax(out.logits[0], dim=-1)
                s_this = 0.0
                for i, tid in enumerate(cand_ids):
                    s_this += logprobs[w_ids2.shape[0] - 1 + i, tid].item()

                other_cand = cand_b if cand_name == "A" else cand_a
                patch_act_other, fc_cache_other = cache_layer_output(
                    model, layers_module, layer, c_ids2, positions=list(range(prompt_len)))
                assert fc_cache_other == 1

                def patch_hook_other(module, args, kwargs, output, _p=patch_act_other, _n=prompt_len):
                    hidden = output[0] if isinstance(output, tuple) else output
                    modified = hidden.clone()
                    modified[:, :_n, :] = _p
                    return (modified, *output[1:]) if isinstance(output, tuple) else modified

                full_target_other = torch.cat([w_ids2, other_cand]).unsqueeze(0)
                with HookHandle(layers_module[layer], patch_hook_other, f"T2b_patch_other_layer{layer}") as hh2:
                    with torch.inference_mode():
                        out_other = model(input_ids=full_target_other)
                logprobs_other = F.log_softmax(out_other.logits[0], dim=-1)
                s_other = 0.0
                for i, tid in enumerate(other_cand):
                    s_other += logprobs_other[w_ids2.shape[0] - 1 + i, tid].item()

                if cand_name == "A":
                    s_a_patched, s_b_patched = s_this, s_other
                else:
                    s_a_patched, s_b_patched = s_other, s_this
                m_patched = margin(s_a_patched, s_b_patched, correct_label2)
                recovery = (m_patched / m_source) if abs(m_source) > 1e-9 else None
                abs_diff = abs(m_patched - m_source)
                deviation_from_1 = abs((recovery - 1.0)) if recovery is not None else None

                w({
                    "test": "T2b_prompt_only_full_patch",
                    "dataset_index": idx2,
                    "candidate_used_for_cache_and_patch": cand_name,
                    "layer": layer,
                    "positions_patched_count": prompt_len,
                    "positions_patched_range": [0, prompt_len - 1],
                    "margin_source": m_source,
                    "margin_patched": m_patched,
                    "recovery": recovery,
                    "deviation_from_1": deviation_from_1,
                    "abs_diff": abs_diff,
                    "note": "no hard pass/fail asserted; recovery should be near 1.0 but need not be exact",
                    "hook_fire_counts": {"cache_main": fc_cache, "patch_main": hh.count,
                                         "cache_other": fc_cache_other, "patch_other": hh2.count},
                })
                log(f"  T2b layer={layer} cand={cand_name}: recovery={recovery} deviation_from_1={deviation_from_1}")
    else:
        w({"test": "T2b_prompt_only_full_patch", "dataset_index": idx2, "passed": None,
           "error": "dataset index unusable or length mismatch in T2"})

    # ================= T3: negative control, corrected =================
    log("T3: negative control (cross-example patch, corrected)...")
    idx_target, idx_source = 206, 1144
    if usable.get(idx_target) and usable.get(idx_source):
        ex_t, cl_t, wl_t, c_ids_t, w_ids_t = get_prompts(idx_target)
        ex_s, cl_s, wl_s, c_ids_s, w_ids_s = get_prompts(idx_source)
        target_ids = w_ids_t  # example 206's wrong-claim prompt
        source_ids = c_ids_s  # example 1144's correct-claim prompt

        s_a_tb = candidate_logprob(model, target_ids, cand_a)
        s_b_tb = candidate_logprob(model, target_ids, cand_b)
        m_target_baseline = margin(s_a_tb, s_b_tb, cl_t)

        test_positions = [
            ("single_position", [min(5, target_ids.shape[0] - 1, source_ids.shape[0] - 1)]),
            ("all_common_positions", list(range(min(target_ids.shape[0], source_ids.shape[0])))),
        ]

        for pos_name, positions in test_positions:
            for layer in layers_to_test:
                slice_len = len(positions)
                if slice_len == 0:
                    w({"test": "T3_negative_control", "position_mode": pos_name,
                       "layer": layer, "slice_len": 0, "passed": False,
                       "error": "slice would be empty; test setup failure"})
                    log(f"  T3 {pos_name} layer={layer}: EMPTY SLICE - test setup failure")
                    continue

                patch_act, fc_cache = cache_layer_output(
                    model, layers_module, layer, source_ids, positions=positions)
                assert fc_cache == 1

                def patch_hook(module, args, kwargs, output, _p=patch_act, _pos=positions):
                    hidden = output[0] if isinstance(output, tuple) else output
                    modified = hidden.clone()
                    modified[:, _pos, :] = _p
                    return (modified, *output[1:]) if isinstance(output, tuple) else modified

                full_target_a = torch.cat([target_ids, cand_a]).unsqueeze(0)
                with HookHandle(layers_module[layer], patch_hook, f"T3_patch_a_layer{layer}") as hha:
                    with torch.inference_mode():
                        out_a = model(input_ids=full_target_a)
                lp_a = F.log_softmax(out_a.logits[0], dim=-1)
                s_a_patched = 0.0
                for i, tid in enumerate(cand_a):
                    s_a_patched += lp_a[target_ids.shape[0] - 1 + i, tid].item()

                full_target_b = torch.cat([target_ids, cand_b]).unsqueeze(0)
                with HookHandle(layers_module[layer], patch_hook, f"T3_patch_b_layer{layer}") as hhb:
                    with torch.inference_mode():
                        out_b = model(input_ids=full_target_b)
                lp_b = F.log_softmax(out_b.logits[0], dim=-1)
                s_b_patched = 0.0
                for i, tid in enumerate(cand_b):
                    s_b_patched += lp_b[target_ids.shape[0] - 1 + i, tid].item()

                m_patched = margin(s_a_patched, s_b_patched, cl_t)
                abs_diff = abs(m_patched - m_target_baseline)
                passed = abs_diff > TOLERANCE

                w({
                    "test": "T3_negative_control",
                    "position_mode": pos_name,
                    "target_index": idx_target,
                    "source_index": idx_source,
                    "layer": layer,
                    "positions_patched_count": slice_len,
                    "positions_patched_range": [positions[0], positions[-1]],
                    "slice_len": slice_len,
                    "margin_target_baseline": m_target_baseline,
                    "margin_patched": m_patched,
                    "oracle_expected": "differs from baseline by > tolerance",
                    "observed": m_patched,
                    "abs_diff": abs_diff,
                    "tolerance": TOLERANCE,
                    "passed": passed,
                    "hook_fire_counts": {"cache": fc_cache, "patch_a": hha.count, "patch_b": hhb.count},
                })
                log(f"  T3 {pos_name} layer={layer}: slice_len={slice_len} baseline={m_target_baseline:.6f} "
                    f"patched={m_patched:.6f} abs_diff={abs_diff:.6e} passed={passed}")
    else:
        w({"test": "T3_negative_control", "passed": None, "error": "one or both dataset indices unusable"})
        log("T3 SKIPPED: indices unusable")

    # ================= T4: regression test reproducing original defect =================
    log("T4: regression test reproducing the original empty-slice defect...")
    if usable.get(idx_target) and usable.get(idx_source):
        # Reuse same target/source as T3.
        target_ids = w_ids_t
        source_ids = c_ids_s
        # Deliberately pick pos >= source length (mimicking mid_pos from a LONGER
        # target while source is shorter).
        pos = source_ids.shape[0]  # >= len(source) guarantees empty slice
        patch_span = 1
        patch_end = min(pos + patch_span, source_ids.shape[0], target_ids.shape[0])
        slice_len = max(0, patch_end - pos)

        s_a_tb = candidate_logprob(model, target_ids, cand_a)
        s_b_tb = candidate_logprob(model, target_ids, cand_b)
        m_target_baseline = margin(s_a_tb, s_b_tb, cl_t)

        for layer in layers_to_test:
            cache = {}

            def cache_hook(module, args, kwargs, output, _pos=pos, _end=patch_end):
                hidden = output[0] if isinstance(output, tuple) else output
                cache["act"] = hidden[:, _pos:_end, :].clone()
                return output

            with HookHandle(layers_module[layer], cache_hook, f"T4_cache_layer{layer}") as hc:
                with torch.inference_mode():
                    _ = model(input_ids=source_ids.unsqueeze(0))
            cached_act = cache["act"]
            assert cached_act.shape[1] == slice_len, "T4 cached activation shape mismatch with expected slice_len"

            def patch_hook(module, args, kwargs, output, _pos=pos, _end=patch_end, _act=cached_act):
                hidden = output[0] if isinstance(output, tuple) else output
                modified = hidden.clone()
                modified[:, _pos:_end, :] = _act
                return (modified, *output[1:]) if isinstance(output, tuple) else modified

            full_target_a = torch.cat([target_ids, cand_a]).unsqueeze(0)
            with HookHandle(layers_module[layer], patch_hook, f"T4_patch_a_layer{layer}") as hha:
                with torch.inference_mode():
                    out_a = model(input_ids=full_target_a)
            lp_a = F.log_softmax(out_a.logits[0], dim=-1)
            s_a_patched = 0.0
            for i, tid in enumerate(cand_a):
                s_a_patched += lp_a[target_ids.shape[0] - 1 + i, tid].item()

            full_target_b = torch.cat([target_ids, cand_b]).unsqueeze(0)
            with HookHandle(layers_module[layer], patch_hook, f"T4_patch_b_layer{layer}") as hhb:
                with torch.inference_mode():
                    out_b = model(input_ids=full_target_b)
            lp_b = F.log_softmax(out_b.logits[0], dim=-1)
            s_b_patched = 0.0
            for i, tid in enumerate(cand_b):
                s_b_patched += lp_b[target_ids.shape[0] - 1 + i, tid].item()

            m_patched = margin(s_a_patched, s_b_patched, cl_t)
            bit_identical = (s_a_patched == s_a_tb) and (s_b_patched == s_b_tb)
            abs_diff = abs(m_patched - m_target_baseline)
            regression_confirmed = (slice_len == 0) and bit_identical
            w({
                "test": "T4_regression_empty_slice",
                "target_index": idx_target,
                "source_index": idx_source,
                "layer": layer,
                "pos": pos,
                "patch_end": patch_end,
                "slice_len": slice_len,
                "expected_slice_len": 0,
                "margin_target_baseline": m_target_baseline,
                "margin_patched": m_patched,
                "bit_identical_to_baseline": bit_identical,
                "abs_diff": abs_diff,
                "oracle_expected": "slice_len == 0 and margins bit-identical (confirms original bug)",
                "passed": regression_confirmed,
                "hook_fire_counts": {"cache": hc.count, "patch_a": hha.count, "patch_b": hhb.count},
            })
            log(f"  T4 layer={layer}: pos={pos} patch_end={patch_end} slice_len={slice_len} "
                f"bit_identical={bit_identical} regression_confirmed={regression_confirmed}")
    else:
        w({"test": "T4_regression_empty_slice", "passed": None, "error": "indices unusable"})

    # ================= T5: single-position sweep =================
    log("T5: single-position sweep at layer 17, example 206...")
    idx5 = 206
    layer5 = 17
    if usable.get(idx5) and layer5 < n_layers:
        ex5, cl5, wl5, c_ids5, w_ids5 = get_prompts(idx5)
        m5_len = min(c_ids5.shape[0], w_ids5.shape[0])

        s_a_c = candidate_logprob(model, c_ids5, cand_a)
        s_b_c = candidate_logprob(model, c_ids5, cand_b)
        m_correct = margin(s_a_c, s_b_c, cl5)
        s_a_w = candidate_logprob(model, w_ids5, cand_a)
        s_b_w = candidate_logprob(model, w_ids5, cand_b)
        m_wrong = margin(s_a_w, s_b_w, cl5)
        denom = m_correct - m_wrong

        diff_positions = [i for i in range(m5_len) if c_ids5[i] != w_ids5[i]]
        claim_span = (min(diff_positions), max(diff_positions)) if diff_positions else None

        n_sweep = 12
        step = max(1, m5_len // n_sweep)
        sweep_positions = sorted(set(list(range(0, m5_len, step))[:n_sweep]))

        for pos in sweep_positions:
            patch_act, fc_cache = cache_layer_output(
                model, layers_module, layer5, c_ids5, positions=[pos])
            assert fc_cache == 1

            def patch_hook(module, args, kwargs, output, _p=patch_act, _pos=pos):
                hidden = output[0] if isinstance(output, tuple) else output
                modified = hidden.clone()
                modified[:, _pos:_pos+1, :] = _p
                return (modified, *output[1:]) if isinstance(output, tuple) else modified

            full_target_a = torch.cat([w_ids5, cand_a]).unsqueeze(0)
            with HookHandle(layers_module[layer5], patch_hook, f"T5_patch_a_pos{pos}") as hha:
                with torch.inference_mode():
                    out_a = model(input_ids=full_target_a)
            lp_a = F.log_softmax(out_a.logits[0], dim=-1)
            s_a_patched = 0.0
            for i, tid in enumerate(cand_a):
                s_a_patched += lp_a[w_ids5.shape[0] - 1 + i, tid].item()

            full_target_b = torch.cat([w_ids5, cand_b]).unsqueeze(0)
            with HookHandle(layers_module[layer5], patch_hook, f"T5_patch_b_pos{pos}") as hhb:
                with torch.inference_mode():
                    out_b = model(input_ids=full_target_b)
            lp_b = F.log_softmax(out_b.logits[0], dim=-1)
            s_b_patched = 0.0
            for i, tid in enumerate(cand_b):
                s_b_patched += lp_b[w_ids5.shape[0] - 1 + i, tid].item()

            m_patched = margin(s_a_patched, s_b_patched, cl5)
            recovery = (m_patched - m_wrong) / denom if abs(denom) > 1e-6 else None
            in_claim_span = (claim_span is not None and claim_span[0] <= pos <= claim_span[1])

            w({
                "test": "T5_position_sweep",
                "dataset_index": idx5,
                "layer": layer5,
                "position": pos,
                "in_claim_span": in_claim_span,
                "claim_span": list(claim_span) if claim_span else None,
                "margin_correct": m_correct,
                "margin_wrong": m_wrong,
                "margin_patched": m_patched,
                "recovery": recovery,
                "hook_fire_counts": {"cache": fc_cache, "patch_a": hha.count, "patch_b": hhb.count},
            })
            log(f"  T5 pos={pos} in_claim_span={in_claim_span} recovery={recovery}")
    else:
        w({"test": "T5_position_sweep", "passed": None, "error": "index unusable or layer out of range"})

    out_f.close()
    total_elapsed = time.time() - t_start
    w_end = {"type": "run_summary", "total_wallclock_seconds": total_elapsed}
    with open(args.output, "a") as f2:
        f2.write(json.dumps(w_end) + "\n")
    log(f"DONE. total wallclock: {total_elapsed:.1f}s. Output: {args.output}")


if __name__ == "__main__":
    main()
