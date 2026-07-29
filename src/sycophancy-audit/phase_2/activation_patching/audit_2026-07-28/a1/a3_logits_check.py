"""A3 logits check: minimal diagnostic to resolve the contradiction between
"layer 18 input at position 243 is exactly zero after zero-ablating layer 17
output at positions 0..prompt_len-1" and "the scored logit at position 243 is
bit-identical to baseline". This script does not modify or import any prior
script. Writes ONLY to results/a3_logits_check_seed42.jsonl (refuses to run
if that file exists).
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
PATCH_LAYER = 17
INSPECT_LAYER = 34
POS_A = 243

t_start = time.time()


def log(msg):
    elapsed = time.time() - t_start
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def build_prompt_text(example, true_label, claim_label, extra_text=None):
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
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=False,
    )


def tokenize(tokenizer, device, text):
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)


class HookHandle:
    def __init__(self, module, fn, name, with_kwargs=True):
        self.module = module
        self.fn = fn
        self.name = name
        self.count = 0
        self.handle = None
        self.with_kwargs = with_kwargs

    def __enter__(self):
        if self.with_kwargs:
            def wrapped(mod, args, kwargs, output):
                self.count += 1
                return self.fn(mod, args, kwargs, output)
            self.handle = self.module.register_forward_hook(wrapped, with_kwargs=True)
        else:
            def wrapped(mod, args, output):
                self.count += 1
                return self.fn(mod, args, output)
            self.handle = self.module.register_forward_hook(wrapped)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


def top5(row_f32):
    vals, idxs = torch.topk(row_f32, 5)
    return [[int(i), float(v)] for i, v in zip(idxs.tolist(), vals.tolist())]


def main():
    out_path = Path("results") / "a3_logits_check_seed42.jsonl"
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

    out_f = open(out_path, "a")

    def w(rec):
        out_f.write(json.dumps(rec, default=str) + "\n")
        out_f.flush()

    w({
        "type": "metadata",
        "model": MODEL_NAME,
        "dtype": str(model.dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "seed": SEED,
        "dataset_index": DATASET_INDEX,
        "patch_layer": PATCH_LAYER,
        "inspect_layer": INSPECT_LAYER,
        "pos_a": POS_A,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Standalone diagnostic; does not import or modify a1_discriminator.py "
                "or any other existing script.",
    })

    log("building BoolQ idx 206 wrong-claim prompt (same construction as a1)...")
    dataset = load_dataset("google/boolq", split="validation")
    ex0 = dataset[DATASET_INDEX]
    correct_label0 = " (A)" if ex0["answer"] else " (B)"
    wrong_label0 = " (B)" if correct_label0 == " (A)" else " (A)"
    text0 = apply_chat_template(tokenizer, build_prompt_text(ex0, wrong_label0, wrong_label0))
    prompt_ids = tokenize(tokenizer, device, text0)
    prompt_len = prompt_ids.shape[0]

    cand = tokenizer(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    cand_len = cand.shape[0]

    full = torch.cat([prompt_ids, cand]).unsqueeze(0)
    full_len = full.shape[1]
    log(f"prompt_len={prompt_len} full_len={full_len} cand_len={cand_len}")

    if POS_A >= full_len:
        w({"type": "error", "message": f"POS_A={POS_A} out of range for full_len={full_len}"})
        log(f"ERROR: POS_A={POS_A} out of range for full_len={full_len}")
        out_f.close()
        sys.exit(1)

    def run_forward(patched):
        """Runs one forward of `full`. If patched, zero-ablates layer 17 output
        at positions 0..prompt_len-1 (same mechanism as a1_discriminator).
        Also registers an inspection-only hook on layer 34 (no modification)."""
        instr = {"patch_hook_fired": 0, "inspect_hook_fired": 0}
        layer34_row_243 = {}
        layer34_row_last = {}

        def patch_hook(module, args, kwargs, output):
            instr["patch_hook_fired"] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            span = min(prompt_len, hidden.shape[1])
            modified[:, :span, :] = torch.zeros_like(hidden[:, :span, :])
            return (modified, *output[1:]) if isinstance(output, tuple) else modified

        def inspect_hook(module, args, kwargs, output):
            instr["inspect_hook_fired"] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            layer34_row_243["row"] = hidden[0, POS_A, :].detach().float().clone()
            layer34_row_last["row"] = hidden[0, -1, :].detach().float().clone()
            return output

        specs = []
        if patched:
            specs.append(HookHandle(layers_module[PATCH_LAYER], patch_hook, f"patch_layer{PATCH_LAYER}"))
        specs.append(HookHandle(layers_module[INSPECT_LAYER], inspect_hook, f"inspect_layer{INSPECT_LAYER}"))

        try:
            for h in specs:
                h.__enter__()
            with torch.inference_mode():
                out = model(input_ids=full)
        finally:
            for h in specs:
                h.__exit__(None, None, None)

        if patched:
            assert instr["patch_hook_fired"] == 1, f"patch hook fired {instr['patch_hook_fired']} times, expected 1"
        assert instr["inspect_hook_fired"] == 1, f"inspect hook fired {instr['inspect_hook_fired']} times, expected 1"

        logits_shape = tuple(out.logits.shape)
        logits_row_count = out.logits.shape[1]
        row_243 = out.logits[0, POS_A].float()
        row_last = out.logits[0, -1].float()

        return {
            "out_logits": out.logits[0].detach().float().clone(),
            "logits_shape": logits_shape,
            "logits_row_count": logits_row_count,
            "logits_row_count_eq_full_len": logits_row_count == full_len,
            "top5_at_243": top5(row_243),
            "top5_at_last": top5(row_last),
            "logit_row_243_norm": float(row_243.norm()),
            "logit_row_last_norm": float(row_last.norm()),
            "final_layer_row_norm_at_243": float(layer34_row_243["row"].norm()),
            "final_layer_row_norm_at_last": float(layer34_row_last["row"].norm()),
            "layer34_row_243": layer34_row_243["row"],
            "layer34_row_last": layer34_row_last["row"],
        }

    log("running UNPATCHED forward...")
    unpatched = run_forward(patched=False)
    rec_unpatched = {k: v for k, v in unpatched.items() if k not in ("out_logits", "layer34_row_243", "layer34_row_last")}
    rec_unpatched["type"] = "forward_unpatched"
    w(rec_unpatched)
    log(f"  UNPATCHED logits_shape={unpatched['logits_shape']} "
        f"logits_row_count_eq_full_len={unpatched['logits_row_count_eq_full_len']} "
        f"logit_row_243_norm={unpatched['logit_row_243_norm']:.6f} "
        f"final_layer_row_norm_at_243={unpatched['final_layer_row_norm_at_243']:.6f}")

    log("running PATCHED forward (zero-ablate layer 17 output positions 0..prompt_len-1)...")
    patched = run_forward(patched=True)
    rec_patched = {k: v for k, v in patched.items() if k not in ("out_logits", "layer34_row_243", "layer34_row_last")}
    rec_patched["type"] = "forward_patched"
    w(rec_patched)
    log(f"  PATCHED logits_shape={patched['logits_shape']} "
        f"logits_row_count_eq_full_len={patched['logits_row_count_eq_full_len']} "
        f"logit_row_243_norm={patched['logit_row_243_norm']:.6f} "
        f"final_layer_row_norm_at_243={patched['final_layer_row_norm_at_243']:.6f}")

    logits_243_max_abs_diff = float((unpatched["out_logits"][POS_A] - patched["out_logits"][POS_A]).abs().max())
    logits_last_max_abs_diff = float((unpatched["out_logits"][-1] - patched["out_logits"][-1]).abs().max())
    final_layer_243_max_abs_diff = float((unpatched["layer34_row_243"] - patched["layer34_row_243"]).abs().max())

    diffs_per_row = (unpatched["out_logits"] - patched["out_logits"]).abs().amax(dim=-1)
    n_rows_differing = int((diffs_per_row > 0).sum())

    logits_shape_mismatch = (not unpatched["logits_row_count_eq_full_len"]) or (not patched["logits_row_count_eq_full_len"])

    if logits_shape_mismatch:
        verdict = "logits_shape_mismatch"
    elif final_layer_243_max_abs_diff > 0 and logits_243_max_abs_diff == 0:
        verdict = "difference_lost_after_final_layer"
    elif final_layer_243_max_abs_diff == 0:
        verdict = "perturbation_vanishes_in_stack"
    elif logits_243_max_abs_diff > 0:
        verdict = "earlier_null_result_not_reproducible"
    else:
        verdict = "unresolved"

    summary = {
        "type": "diagnostic_summary",
        "logits_shape_unpatched": unpatched["logits_shape"],
        "logits_shape_patched": patched["logits_shape"],
        "logits_row_count_eq_full_len_unpatched": unpatched["logits_row_count_eq_full_len"],
        "logits_row_count_eq_full_len_patched": patched["logits_row_count_eq_full_len"],
        "logits_243_max_abs_diff": logits_243_max_abs_diff,
        "logits_last_max_abs_diff": logits_last_max_abs_diff,
        "final_layer_243_max_abs_diff": final_layer_243_max_abs_diff,
        "n_rows_differing": n_rows_differing,
        "final_layer_row_norm_at_243_unpatched": unpatched["final_layer_row_norm_at_243"],
        "final_layer_row_norm_at_243_patched": patched["final_layer_row_norm_at_243"],
        "final_layer_row_norm_at_last_unpatched": unpatched["final_layer_row_norm_at_last"],
        "final_layer_row_norm_at_last_patched": patched["final_layer_row_norm_at_last"],
        "verdict": verdict,
    }
    w(summary)
    log(f"SUMMARY: {json.dumps(summary)}")

    out_f.close()
    total_elapsed = time.time() - t_start
    with open(out_path, "a") as f2:
        f2.write(json.dumps({"type": "run_summary", "total_wallclock_seconds": total_elapsed}) + "\n")
    log(f"DONE. total wallclock: {total_elapsed:.1f}s. Output: {out_path}")


if __name__ == "__main__":
    main()
