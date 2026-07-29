# Recovery note (2026-07-28)

## Provenance of the driver code

The code that produced `results/experiment_a_seed42.jsonl` and
`results/full_experiment_seed42.jsonl` (i.e. `full_experiment.py`, `experiment_a.py`,
`chunk1_model_audit.py` through `chunk6_sweep.py`, `chunk6_example_selection.py`, and
`test_activation_patching.py`) existed only on GPU VM `atharva-experiments-l4-b`. That
VM's disk was never a git repository, so this code was never committed anywhere. It was
recovered via SSH/tar from the VM on 2026-07-28 and is added to this directory in this
commit.

`example_selection_seed42.json` and `sweep_discovery_seed42.json` already present in
this directory were compared against the recovered copies and found to be byte-identical;
no separate `.recovered` copy was needed.

## Relationship to `activation_patching.py`

`activation_patching.py`, already present in this directory prior to the 2026-07-28
recovery, is a later, separate consolidation attempt at unifying the driver code above
into a single module. It is not the code that produced the two `results/*.jsonl` files
listed above, has no working CLI, imports `torch` only inside functions, and cannot
currently be run end-to-end.

## `audit_2026-07-28/`

This subdirectory contains new diagnostic scripts written during a 2026-07-28 audit
session of the recovered driver code, organized as:

- `audit_2026-07-28/a1/` — `a1_discriminator.py`, `a2_propagation.py`,
  `a3_logits_check.py`, `nan_diag.py`, and their `.jsonl` outputs.
- `audit_2026-07-28/controls_v2/` — `controls_v2.py` and its `.jsonl` output.
- `audit_2026-07-28/full_v2/` — `full_experiment_v2.py` and its `.jsonl` output.

The audit identified two confirmed defects in the original patching method:

1. **Empty-slice bug**: an indexing error made the shuffled-source negative control a
   silent no-op — the "shuffled" source slice was empty, so the control never actually
   patched anything different from the real condition.
2. **Scoring bug**: the scorer summed logprobs over all 3 tokens of the `" (A)"`/`" (B)"`
   candidate strings instead of only the single discriminating middle token. Combined
   with this model's KV-sharing architecture (layers 15-34 share attention K/V frozen
   from layers <=14), this made patches applied at those layers appear to have zero
   effect on the score, regardless of whether the patch actually changed anything
   upstream.

A further, more serious issue was identified during the audit and has **not** been
fixed: the prompt construction used for patching in `full_experiment_v2.py` does not
clearly implement a true/false sycophancy manipulation. It needs a prompt-construction
fix before results produced by it should be treated as measuring sycophancy. This is
an open issue as of this commit.
