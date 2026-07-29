"""A1 discriminator: does patching layers[N] output[0] control downstream
computation at non-final layers, or is this model using multiple parallel
residual streams (AltUp / per-layer embeddings) that make output[0]-only
patching a structurally incomplete intervention?

Standalone script. Does not import or modify full_experiment.py,
experiment_a.py, controls_v2.py, or any chunk*.py. Writes ONLY to
results/a1_discriminator_seed42.jsonl (refuses to run if that file exists).

Tests:
  T1 (structural_check + config_dump) - inspect layer output structure,
      module signature, and config for AltUp / per-layer-embedding evidence.
  T2 (noise_floor) - repeat identical unpatched forward 5x, derive tolerance.
  T3 (zero-ablation discriminator) - catastrophic zero-patch of output[0]
      across the full prompt span at layers 0, 17, 34.
  T4 (sign-flip cross-check) - same as T3 but hidden * -1.0.
  T5 (complete-state patch) - only if T1 reveals additional stream(s);
      patch ALL tuple elements / the complete state, not just output[0].
"""

import inspect
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
LAYERS_TO_TEST = [0, 17, 34]
DATASET_INDEX = 206

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
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)


def candidate_logprob(model, prompt_ids, cand_ids):
    """Same scoring method as full_experiment.py / controls_v2.py, but with
    the required fp32 upcast fix: bfloat16 log_softmax has ~0.03 resolution
    near logP -5, coarser than any tolerance we care about here."""
    full = torch.cat([prompt_ids, cand_ids]).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=full)
    logprobs = F.log_softmax(out.logits[0].float(), dim=-1)
    s = 0.0
    for i, tid in enumerate(cand_ids):
        s += logprobs[len(prompt_ids) - 1 + i, tid].item()
    return s


def margin(s_a, s_b, correct_label):
    return (s_a - s_b) if correct_label == " (A)" else (s_b - s_a)


class HookHandle:
    """Context manager that registers a hook and removes it in `finally`,
    so a leaked hook can never silently corrupt later numbers."""

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
        return False


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


def full_score(model, prompt_ids, cand_a, cand_b, correct_label):
    s_a = candidate_logprob(model, prompt_ids, cand_a)
    s_b = candidate_logprob(model, prompt_ids, cand_b)
    return margin(s_a, s_b, correct_label)


def main():
    import os
    out_name = os.environ.get("A1_OUTPUT_NAME", "a1_discriminator_seed42.jsonl")
    out_path = Path("results") / out_name
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
    hidden_dim = tc.hidden_size

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
        "hidden_dim": hidden_dim,
        "seed": SEED,
        "dataset": "google/boolq",
        "split": "validation",
        "dataset_index": DATASET_INDEX,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hook_point": "model.model.language_model.layers[N] output",
        "note": "Standalone independent discriminator script; does not import "
                "full_experiment.py / experiment_a.py / controls_v2.py / chunk*.py.",
    }
    w(metadata)

    # ================= Test 1: structural check =================
    log("Test 1: structural check (layers 0, 17, 34)...")
    dataset = load_dataset("google/boolq", split="validation")
    ex0 = dataset[DATASET_INDEX]
    correct_label0 = " (A)" if ex0["answer"] else " (B)"
    wrong_label0 = " (B)" if correct_label0 == " (A)" else " (A)"
    w_text0 = apply_chat_template(tokenizer, build_prompt_text(ex0, wrong_label0, wrong_label0))
    w_ids0 = tokenize(tokenizer, device, w_text0)
    prompt_len0 = w_ids0.shape[0]

    def struct_hook(module, args, kwargs, output, _store):
        _store["output_type"] = str(type(output))
        if isinstance(output, tuple):
            _store["output_len"] = len(output)
            elems = []
            for e in output:
                if torch.is_tensor(e):
                    elems.append({"is_tensor": True, "shape": list(e.shape), "dtype": str(e.dtype)})
                else:
                    elems.append({"is_tensor": False, "type": str(type(e)), "repr": repr(e)[:200]})
            _store["elements"] = elems
        else:
            _store["output_len"] = None
            if torch.is_tensor(output):
                _store["elements"] = [{"is_tensor": True, "shape": list(output.shape), "dtype": str(output.dtype)}]
            else:
                _store["elements"] = [{"is_tensor": False, "type": str(type(output)), "repr": repr(output)[:200]}]
        _store["module_type"] = str(type(module))
        _store["module_class_module"] = type(module).__module__
        try:
            _store["forward_signature"] = str(inspect.signature(module.forward))
        except Exception as e:
            _store["forward_signature"] = f"<error: {e}>"
        _store["child_module_names"] = [n for n, _ in module.named_children()]
        return output

    for layer in LAYERS_TO_TEST:
        store = {}

        def hook_fn(module, args, kwargs, output, _s=store):
            return struct_hook(module, args, kwargs, output, _s)

        with HookHandle(layers_module[layer], hook_fn, f"struct_layer{layer}") as h:
            with torch.inference_mode():
                _ = model(input_ids=w_ids0.unsqueeze(0))
        assert h.count == 1, f"structural hook at layer {layer} fired {h.count} times, expected 1"

        rec = {"type": "structural_check", "layer": layer, "prompt_len": prompt_len0}
        rec.update(store)
        w(rec)
        log(f"  layer={layer}: output_type={store['output_type']} output_len={store.get('output_len')} "
            f"module_type={store['module_type']} children={store['child_module_names']}")

    # config dump: attributes matching relevant substrings
    substrings = ["altup", "alt_up", "laurel", "per_layer", "num_hidden_layers",
                  "hidden_size", "kv_shared", "sliding", "activation_sparsity"]

    def scan_config(cfg):
        found = {}
        for attr in dir(cfg):
            if attr.startswith("_"):
                continue
            low = attr.lower()
            if any(s in low for s in substrings):
                try:
                    val = getattr(cfg, attr)
                except Exception as e:
                    val = f"<error: {e}>"
                if not callable(val):
                    found[attr] = val
        return found

    top_config_matches = scan_config(model.config)
    text_config_matches = scan_config(tc)

    # dump full text_config as dict if small
    try:
        tc_dict = tc.to_dict()
        tc_dump = tc_dict if len(json.dumps(tc_dict, default=str)) < 20000 else {
            "note": "text_config too large to dump in full", "keys": list(tc_dict.keys())
        }
    except Exception as e:
        tc_dump = {"error": str(e)}

    config_record = {
        "type": "config_dump",
        "model_type": type(model).__name__,
        "model_model_type": type(model.model).__name__,
        "language_model_children": [n for n, _ in model.model.language_model.named_children()],
        "top_config_matches": top_config_matches,
        "text_config_matches": text_config_matches,
        "text_config_full": tc_dump,
    }
    w(config_record)
    log(f"  config: top-level matches={list(top_config_matches.keys())}")
    log(f"  config: text_config matches={list(text_config_matches.keys())}")
    log(f"  type(model)={type(model).__name__} type(model.model)={type(model.model).__name__}")
    log(f"  language_model children: {[n for n, _ in model.model.language_model.named_children()]}")

    # Determine whether Test 5 is applicable: multiple tuple elements at any
    # tested layer, or AltUp/per-layer-embedding evidence in config.
    multi_element_layers = []
    for layer in LAYERS_TO_TEST:
        # re-read from what we just wrote (structural_check records)
        pass
    # simpler: recompute from stored structs directly
    struct_records = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "structural_check":
                struct_records.append(r)
    for r in struct_records:
        if r.get("output_len") is not None and r.get("output_len") > 1:
            multi_element_layers.append(r["layer"])

    altup_config_evidence = bool(top_config_matches) or bool(text_config_matches)
    # more precisely: only altup/per_layer related keys count as "additional streams" evidence
    stream_related_keys = [k for k in list(top_config_matches.keys()) + list(text_config_matches.keys())
                           if any(s in k.lower() for s in ["altup", "alt_up", "laurel", "per_layer"])]

    test5_applicable = bool(multi_element_layers) or bool(stream_related_keys)
    log(f"  multi_element_layers={multi_element_layers} stream_related_config_keys={stream_related_keys}")
    log(f"  Test 5 applicable: {test5_applicable}")

    # ================= Test 2: noise floor =================
    log("Test 2: noise floor (5x identical unpatched forward, BoolQ idx 206 wrong-claim)...")
    margins = []
    for i in range(5):
        m = full_score(model, w_ids0, tokenizer(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device),
                        tokenizer(" (B)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device),
                        correct_label0)
        margins.append(m)
        log(f"  run {i+1}/5: margin={m:.10f}")
    spread = max(margins) - min(margins)
    TOLERANCE = max(10 * spread, 1e-6)
    w({"type": "noise_floor", "dataset_index": DATASET_INDEX, "margins": margins,
       "spread": spread, "tolerance": TOLERANCE})
    log(f"  spread={spread:.10e} derived TOLERANCE={TOLERANCE:.10e}")

    cand_a = tokenizer(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
    cand_b = tokenizer(" (B)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)

    margin_baseline = full_score(model, w_ids0, cand_a, cand_b, correct_label0)

    # ================= Test 3: zero-ablation discriminator =================
    log("Test 3: zero-ablation discriminator...")

    def run_patch_test(layer, patch_fn, test_name):
        """patch_fn(hidden) -> modified tensor for positions 0..prompt_len-1."""
        instrumentation = {}

        def patch_hook(module, args, kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden.clone()
            span = min(prompt_len0, hidden.shape[1])
            modified[:, :span, :] = patch_fn(hidden[:, :span, :])
            instrumentation["max_abs_delta"] = float((modified - hidden).abs().max())
            instrumentation["n_elements_changed"] = int((modified != hidden).sum())
            instrumentation["hidden_shape"] = list(hidden.shape)
            instrumentation["output_type"] = str(type(output))
            instrumentation["output_len"] = len(output) if isinstance(output, tuple) else None
            instrumentation["write_landed"] = instrumentation["max_abs_delta"] > 0
            return (modified, *output[1:]) if isinstance(output, tuple) else modified

        def scored(cand_ids):
            full = torch.cat([w_ids0, cand_ids]).unsqueeze(0)
            with HookHandle(layers_module[layer], patch_hook, f"{test_name}_layer{layer}") as h:
                with torch.inference_mode():
                    out = model(input_ids=full)
            assert h.count == 1, f"{test_name} patch hook at layer {layer} fired {h.count} times, expected 1"
            logprobs = F.log_softmax(out.logits[0].float(), dim=-1)
            s = 0.0
            for i, tid in enumerate(cand_ids):
                s += logprobs[w_ids0.shape[0] - 1 + i, tid].item()
            return s

        s_a = scored(cand_a)
        s_b = scored(cand_b)
        margin_patched = margin(s_a, s_b, correct_label0)
        abs_diff = abs(margin_patched - margin_baseline)
        moved = abs_diff > TOLERANCE

        rec = {
            "type": test_name,
            "layer": layer,
            "dataset_index": DATASET_INDEX,
            "prompt_len": prompt_len0,
            "max_abs_delta": instrumentation["max_abs_delta"],
            "n_elements_changed": instrumentation["n_elements_changed"],
            "hidden_shape": instrumentation["hidden_shape"],
            "output_type": instrumentation["output_type"],
            "output_len": instrumentation["output_len"],
            "write_landed": instrumentation["write_landed"],
            "margin_baseline": margin_baseline,
            "margin_patched": margin_patched,
            "abs_diff": abs_diff,
            "tolerance": TOLERANCE,
            "moved": moved,
        }
        if not instrumentation["write_landed"]:
            rec["interpretation"] = "write did not land at all (max_abs_delta == 0)"
        elif not moved:
            rec["interpretation"] = ("write landed but margin did NOT move beyond tolerance: "
                                      "patching output[0] does NOT control downstream computation at this layer")
        else:
            rec["interpretation"] = "write landed and margin moved: mechanism controls downstream computation at this layer"
        w(rec)
        log(f"  {test_name} layer={layer}: max_abs_delta={rec['max_abs_delta']:.4f} "
            f"write_landed={rec['write_landed']} margin_baseline={margin_baseline:.6f} "
            f"margin_patched={margin_patched:.6f} abs_diff={abs_diff:.6e} moved={moved}")
        return rec

    test3_records = {}
    for layer in LAYERS_TO_TEST:
        test3_records[layer] = run_patch_test(
            layer, lambda h: torch.zeros_like(h), "zero_ablation_discriminator")

    # ================= Test 4: sign-flip cross-check =================
    log("Test 4: sign-flip cross-check...")
    test4_records = {}
    for layer in LAYERS_TO_TEST:
        test4_records[layer] = run_patch_test(
            layer, lambda h: h * -1.0, "sign_flip_crosscheck")

    # ================= Test 5: complete-state patch (conditional) =================
    # Test 1's forward-signature inspection showed each decoder layer's forward
    # takes hidden_states AND a SEPARATE positional argument `per_layer_input`
    # (confirmed by direct pre-hook probing: arg0 shape [1,seq,1536] (hidden_states),
    # arg1 shape [1,seq,256] (per_layer_input), at layers 0, 17, and 34 alike).
    # The layer's *output* (what Test 3/4 patch) is a single tensor with no tuple
    # structure -- so the "additional streams" this model actually has are NOT
    # extra elements of output. They are this separate per_layer_input argument,
    # injected into EVERY layer's forward call, that is not derived from the
    # patched hidden_states and is therefore untouched by an output-only hook.
    # Confirmed language_model children: embed_tokens_per_layer / per_layer_model_projection /
    # per_layer_projection_norm compute this per-layer embedding once from input_ids.
    if test5_applicable:
        log("Test 5: complete-state zero-ablation (output AND per_layer_input argument)...")

        def make_per_layer_prehook(span):
            def prehook(module, args, kwargs):
                if len(args) < 2 or not torch.is_tensor(args[1]):
                    return None
                pli = args[1]
                s = min(span, pli.shape[1])
                modified = pli.clone()
                modified[:, :s, ...] = 0
                new_args = list(args)
                new_args[0] = args[0]
                new_args[1] = modified
                return (tuple(new_args), kwargs)
            return prehook

        class MultiHook:
            """Registers several (module, fn, is_pre) hooks and removes ALL of
            them in `finally`, so a partial failure can't leak hooks forward."""

            def __init__(self, specs):
                self.specs = specs
                self.handles = []

            def __enter__(self):
                for module, fn, is_pre in self.specs:
                    if is_pre:
                        self.handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
                    else:
                        self.handles.append(module.register_forward_hook(fn, with_kwargs=True))
                return self

            def __exit__(self, exc_type, exc, tb):
                for h in self.handles:
                    h.remove()
                return False

        def run_complete_state_test(layer, test_name, prehook_layers):
            """Zero output[0]-equivalent hidden_states at `layer` (post-hook, as
            in Test 3) AND zero the per_layer_input argument fed to every layer
            in `prehook_layers` (pre-hook). Reports whether that combined,
            structurally-complete zero-ablation moves the margin."""
            out_instr = {}

            def out_hook(module, args, kwargs, output):
                hidden = output
                span = min(prompt_len0, hidden.shape[1])
                modified = hidden.clone()
                modified[:, :span, :] = torch.zeros_like(hidden[:, :span, :])
                out_instr["max_abs_delta_output"] = float((modified - hidden).abs().max())
                return modified

            pre_fn = make_per_layer_prehook(prompt_len0)
            specs = [(layers_module[layer], out_hook, False)]
            for pl in prehook_layers:
                specs.append((layers_module[pl], pre_fn, True))

            def scored(cand_ids):
                full = torch.cat([w_ids0, cand_ids]).unsqueeze(0)
                with MultiHook(specs):
                    with torch.inference_mode():
                        out = model(input_ids=full)
                logprobs = F.log_softmax(out.logits[0].float(), dim=-1)
                s = 0.0
                for i, tid in enumerate(cand_ids):
                    s += logprobs[w_ids0.shape[0] - 1 + i, tid].item()
                return s

            s_a = scored(cand_a)
            s_b = scored(cand_b)
            margin_patched = margin(s_a, s_b, correct_label0)
            abs_diff = abs(margin_patched - margin_baseline)
            moved = abs_diff > TOLERANCE
            rec = {
                "type": test_name,
                "layer": layer,
                "per_layer_input_zeroed_at_layers": prehook_layers,
                "dataset_index": DATASET_INDEX,
                "max_abs_delta_output": out_instr.get("max_abs_delta_output"),
                "margin_baseline": margin_baseline,
                "margin_patched": margin_patched,
                "abs_diff": abs_diff,
                "tolerance": TOLERANCE,
                "moved": moved,
                "interpretation": ("patching the complete state (hidden_states output AND "
                                   "per_layer_input argument) moves the margin: confirms "
                                   "output-only patching was structurally incomplete") if moved else
                                  ("patching the complete state STILL does not move the margin: "
                                   "the recovery is not (fully) explained by the per_layer_input "
                                   "channel identified here"),
            }
            w(rec)
            log(f"  {test_name} layer={layer} prehook_layers={prehook_layers}: "
                f"max_abs_delta_output={rec['max_abs_delta_output']:.4f} abs_diff={abs_diff:.6e} moved={moved}")

        for layer in LAYERS_TO_TEST:
            # 5a: zero per_layer_input only at this layer (single-layer complete state)
            run_complete_state_test(layer, "complete_state_zero_ablation_single_layer", [layer])
            # 5b: zero per_layer_input at this layer AND every subsequent layer
            # (since per_layer_input is injected fresh at each layer, not carried
            # forward through hidden_states, downstream layers' own untouched
            # per_layer_input could alone explain any residual recovery).
            run_complete_state_test(layer, "complete_state_zero_ablation_layer_and_downstream",
                                     list(range(layer, n_layers)))
    else:
        w({"type": "test5_skipped", "reason":
           "Test 1 found no additional tuple elements beyond output[0] and no "
           "AltUp/per-layer-embedding related config attributes; Test 5 not applicable."})
        log("Test 5: SKIPPED — no multi-element output or AltUp/per-layer-embedding config evidence found.")

    out_f.close()
    total_elapsed = time.time() - t_start
    with open(out_path, "a") as f2:
        f2.write(json.dumps({"type": "run_summary", "total_wallclock_seconds": total_elapsed}) + "\n")
    log(f"DONE. total wallclock: {total_elapsed:.1f}s. Output: {out_path}")


if __name__ == "__main__":
    main()
