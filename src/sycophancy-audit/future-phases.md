# Future Phases

Last updated: 2026-07-18

These are intentionally deferred until Phases 1 to 3 show a reliable behavioral effect.

## Phase 4: Intervention

Train a small evidence-grounded anti-sycophancy intervention, likely SFT or DPO, and test whether it reduces directional verdict shifts.

The intervention must be evaluated on held-out scenarios, not only the scenarios used for training.

## Phase 5: Trade-Off Audit

Check whether reducing sycophancy causes undesirable behavior:

- Cold or dismissive tone
- Excessive refusal or inability to give practical advice
- Reduced factual accuracy
- Worse performance on ordinary helpfulness prompts

## Later Extensions

- Repeat Phases 1 to 3 on small Qwen and OLMo models under 2B parameters
- Compare base, instruction-tuned, and preference-trained checkpoints where compatible checkpoints exist
- Use activation or SAE analysis only after the behavioral effect is measured reliably
