# NNsight Sycophancy Audit (learning project)

Fresh reimplementation of the activation-patching result from
`src/sycophancy-audit/phase_2/activation_patching/`, built from scratch with
NNsight as the primary library — not a port or a wrapper around the existing
raw-PyTorch-hooks code.

## Ground rule

Nothing under `src/sycophancy-audit/` is edited, moved, or imported by this
folder. That code stays exactly as-is; see its own `RECOVERY_NOTE.md` for
provenance. This folder is additive and self-contained.

## Goal

Learn NNsight by reproducing the documented `readout_token` / `claim_span`
activation-patching result on `google/gemma-4-E2B-it` (see
`docs/sycophancy-audit-summary.md`, "Extension A: Activation Patching
(Corrected)") using NNsight's tracing API instead of manual
`register_forward_hook` calls. Success is qualitative match to that result,
not just code that runs.

Last updated: 2026-08-12

## Plan

### 1. Learn the basic NNsight model

Use one existing Gemma 4 prompt to learn:

- `LanguageModel` and `model.trace()`
- intermediate layer outputs
- `.save()` and tensor shapes
- writing a small activation intervention

No experiment sweep yet.

### 2. Reproduce one patch

Using the existing v4 experiment as the reference:

- model: `google/gemma-4-E2B-it`
- condition: correct claim → incorrect claim
- first position: `readout_token`
- first scale: one example and one layer

Compare tokenization, logits, margins, and patched activations against the existing PyTorch-hook implementation.

### 3. Reproduce the full small result

Only after the single patch matches:

- add `claim_span`
- sweep the existing selected layers
- run the unrelated-donor negative control
- compare layer-wise recovery and control behavior

A mismatch stops the experiment until it is explained.

### 4. Extend the causal analysis

Only after equivalence is established, move from layer-level patches to:

```text
layer → attention vs. MLP → token position → smaller components
```

Test both necessity (ablation) and sufficiency (injection). Attribution methods are for candidate generation; important results require causal patches.

### 5. Consider SAE analysis later

Revisit `src/sae-gemma-270m/` only if the causal result gives us a concrete feature hypothesis. SAE work is not part of the initial reproduction.

## Research records

For each meaningful run, record:

- hypothesis and exact intervention;
- expected and actual result;
- controls and failures;
- bugs discovered;
- interpretation and alternative explanations.

The existing corrected implementation is at:

```text
src/sycophancy-audit/phase_2/activation_patching/audit_2026-07-28/full_v4/
```
