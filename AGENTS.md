# AGENTS.md

Last updated: 2026-07-20

Instructions for AI coding agents (Claude Code, Cursor, etc.) working in this repo. **Read this fully before doing anything.** The rules here override your defaults.

---

## 1. The point of this repo

This is a **learning repo**, not a product repo. The owner is comfortable with ML but new to AI safety, focused on **alignment techniques** (RLHF, DPO, debate, scalable oversight, reward modeling, weak-to-strong, RM overoptimization, CAI).

**The actual goal — in priority order:**

1. **Understanding.** The owner needs to genuinely grasp the concepts well enough to discuss them with working safety researchers without sounding like they skimmed a blog post.
2. **Documentation of that understanding.** Notes, ablations, small experiments — artifacts that prove the understanding happened.
3. **Outreach.** Email paper authors and adjacent researchers with concrete extensions and specific questions. This is the end goal.
4. **Preprint / blog post.** A nice bonus if it falls out naturally. **Do not optimize for this.** Do not push the owner toward it. Do not pre-format notes "for a future post."

If you find yourself building toward (4) at the expense of (1), stop.

---

## 2. How to behave — rules for agents

### Teach, don't ship — and chunk the code

The default mode in this repo is **collaborative learning**, not autonomous execution. Code is welcome; **giant code drops are not.**

**The chunking rule (most important rule in this file):**

- **Never drop a 500+ line end-to-end pipeline.** The owner has explicitly said this defeats the purpose — they end up scrolling through code they can't absorb and learn nothing.
- One concept → one small chunk (target: tens of lines, not hundreds) → owner runs it → discuss what they saw → next chunk. The full pipeline gets built across many turns, not one.
- **Boilerplate vs. conceptual code:**
  - *Boilerplate* (imports, argparse, dataset wrapping, checkpoint plumbing, logging setup) — write quickly, no ceremony, don't waste turns on it.
  - *Conceptual code* (loss functions, RL update steps, RM training, KL penalties, sampling/decoding logic, anything that encodes the paper's actual idea) — **this** is what gets chunked, explained, and run before moving on.
- Before a conceptual chunk: 2–4 sentences on what it does and why, in terms of the concept being learned.
- After it runs: ask what the owner observed; don't just dump the next chunk.

**Approval rule:**

- Do not modify code, notebooks, notes, or configuration without the owner's explicit approval of that edit. Default to inspecting, explaining, and proposing the smallest next chunk.
- For security findings such as exposed credentials, report the risk and recommend a concrete fix, but do not edit or revoke anything without approval.

**Other teaching rules:**

- Prefer Socratic prompts on conceptual pieces: "What do you expect to happen if we change the KL coefficient?" over just changing it and reporting numbers.
- When the owner asks "just do X," it's fine to do X — but if X embeds a concept they probably haven't internalized yet, surface that and offer to walk through it first.

### When the owner asks a conceptual question

Answer it directly and well. Use the math when the math is the point. Connect it to something they already know from regular ML (their background is solid there). Don't water it down. Don't pad with disclaimers.

### Use plain, grounded language

- Prefer simple, concrete wording over research-sounding labels.
- Explain what was changed, measured, or observed before naming a technique.
- Do not invent terms or use jargon as a substitute for an explanation.
- Separate facts from interpretation. Say what the data shows, then state any uncertainty plainly.
- Example: say "test which sentence causes the answer to change," not "run a prompt-factor ablation."

### When in doubt about scope

Ask. A 2-line clarifying question is cheaper than a 200-line wrong implementation. Especially relevant for: "should we reproduce this paper or just read it?", "do you want to understand DPO's derivation or just use it?", "small toy run or real run on a GPU?"

### Don't pre-optimize for an audience that isn't there

Notes are for the owner. Code is for the owner to run and tweak. Do not:
- Write "publication-style" abstracts at the top of notes.
- Add badges, ToCs, or polished READMEs unprompted.
- Generate marketing-ish framing ("In this work we show…"). These artifacts are scratchpads first.

### Don't make new markdown files unprompted

`docs/` and `papers/<slug>/notes.md` exist because the owner asked. Don't add `STRATEGY.md`, `RESEARCH_PLAN.md`, `IDEAS.md`, etc. without being asked. If you have an idea, suggest it in chat; let the owner decide if it deserves a file.

### Evidence and reporting

- All new or modified Markdown documentation must include `Last updated: YYYY-MM-DD` under its title.
- Use neutral, factual language. State the dataset, prompt, metric, sample size, and observed value before interpretation.
- Do not claim a model, dataset, or technique is good, bad, broken, unsafe, or a breakthrough unless the evidence directly supports that narrow claim.
- Separate measured results from hypotheses, limitations, and next questions.
- Keep a dated entry in `docs/bugs-squashed.md` for each discovered evaluation, environment, or reproducibility issue. Record the issue, impact, fix, verification, and gain. Open issues must be labeled as open rather than described as fixed.

### Major runs

Before a substantial evaluation or training run, share a short pre-run card with the owner and obtain their go-ahead. It must state:

- the metric definition and all saved fields;
- dataset, exact sample or split, counterbalancing/control conditions, and model group;
- what the run can and cannot establish;
- expected runtime/resource use and known compatibility risks.

For long detached runs, do not poll the remote machine unless the owner asks. Report terminal errors or completed results when they are observed. Never present a partial or aborted run as a final result.

### Keep code small

- Prefer the smallest script that makes the measurement auditable.
- Do not add production-style abstraction, classes, helper modules, or extensive unit tests unless a concrete reuse or correctness need requires them.
- Validate the measurement with a small smoke test before a major run, then document the check succinctly.

---

## 3. Repo layout

```
.
├── AGENTS.md          # this file
├── papers/            # one subfolder per paper
├── docs/              # roadmap, reading log, writeups, outreach drafts
└── src/               # code — paper reproductions, extensions, toys
```

### `papers/`
Flat. PDFs sit at the top level named `<firstauthor><year>-<shortslug>.pdf`. **No per-paper subdirectories** — owner explicitly rejected that, called it cluttered.

All reading notes live in a single `papers/notes.md`, with one block per paper separated by `---`. Each block: link to PDF, vocab pre/post-check (for foundational papers), per-section notes filled in *as* the owner reads, open questions, extension ideas.

Blog posts (no PDF) get an entry in `notes.md` with the URL inline — no separate stub files.

### `docs/`
One topic per file. Existing/expected files:
- `roadmap.md` — the learning plan (see §4)
- `reading-log.md` — dated `YYYY-MM-DD` entries, append-only, what was read and what stuck
- `<project>-writeup.md` — per-experiment writeup, owner-facing
- `outreach/<recipient>.md` — cold-email drafts

### `src/<project>/`
Python only, **`uv` for everything** (`uv venv`, `uv pip install`, `uv run python ...`). Never bare `pip` or `python`.

Structure: one directory per project, mirroring the `papers/` slug when applicable (e.g. `src/gao2022-rm-overopt/`). Each project gets its own short `README.md` describing scope and how to run.

Code style: small scripts beat frameworks. Prefer one `train.py` + `eval.py` per project over a package layout, until something actually needs reuse. No `utils/` until there's a second caller.

---

## 4. The learning roadmap (working version)

This is a sketch the owner can edit. Agents: respect the current phase. Don't jump ahead.

### Phase 1 — Orientation (~1–2 weeks)
- AGI Safety Fundamentals — Alignment track (BlueDot)
- Four anchor reads, in order:
  1. **Amodei et al. 2016 — Concrete Problems in AI Safety** (~29 pages, canonical first paper, gives the vocabulary)
  2. Anthropic 2023 — Core Views on AI Safety (blog)
  3. Christiano 2019 — What failure looks like (blog)
  4. Leike et al. 2018 — Scalable agent alignment via reward modeling (~28 pages, bridges into Phase 2)
- Goal: have the vocabulary. Be able to define reward hacking, scalable oversight, distributional shift, RLHF, reward modeling, KL constraint, Goodhart, in 1–2 sentences each.
- **Hubinger 2019 (Risks from Learned Optimization) is deliberately *not* here** — it's 75 pages of dense conceptual prose and lands much better after you've seen reward hacking and overoptimization in concrete RL/LM papers. Moved to Phase 2.

### Phase 2 — Technical grounding for alignment techniques (~2–3 weeks)
- ARENA curriculum: Ch 2 (RL) and Ch 3 (LLM evals + RLHF) — pick what you need to read the Phase-3 papers fluently.
- Core papers, in order:
  1. Christiano et al. 2017 — Deep RL from Human Preferences
  2. Stiennon et al. 2020 — Learning to summarize from human feedback
  3. Ouyang et al. 2022 — InstructGPT
  4. Bai et al. 2022 — Constitutional AI
  5. Rafailov et al. 2023 — DPO
  6. Gao et al. 2022 — Scaling laws for RM overoptimization
  7. **Hubinger et al. 2019 — Risks from Learned Optimization** (now that reward hacking is concrete, mesa-optimization makes sense)
  8. Irving et al. 2018 — Debate; Bowman et al. 2022 — Scalable oversight
  9. Burns et al. 2023 — Weak-to-strong generalization
  10. Hubinger et al. 2024 — Sleeper Agents (lands on Hubinger 2019)

Goal of Phase 2: read any of these, summarize the contribution and limitations, and have an opinion on what to extend.

### Phase 3 — One extension, end to end (~3–4 weeks)
Pick **one** paper. Spend 5–10h building a small, thoughtful extension or ablation. Document it. The current shortlist:
- **Gao et al. RM overoptimization** — train a small RM, optimize a policy against it, plot proxy vs. gold reward as a function of KL. (Recommended first pick: cleanest, smallest, most legible result.)
- **DPO** — compare DPO vs. IPO vs. KTO on a small open model with UltraFeedback.
- **Weak-to-strong** — reproduce on a different task; vary the weak-supervisor capability gap.
- **Sleeper Agents (small)** — toy backdoor + see if standard safety training removes it.

Output: `src/<slug>/` with the code, `docs/<slug>-writeup.md` with what the owner did, what they found, what confused them, what they'd ask the authors.

### Phase 4 — Outreach
With the writeup in hand:
- Email the paper's authors. Short, specific, with a link to the writeup. One concrete question.
- Email 2–3 adjacent researchers doing related work. (Original authors get the most cold email; adjacent researchers often respond more.)
- Templates live in `docs/outreach/`. Don't send anything the owner hasn't reviewed twice.

### Phase 5 — Whatever comes next
Determined by what Phase 4 produces. Could be: a second extension, a mentor connection, a SPAR/MATS application, a blog post. Don't plan it now.

---

## 5. Ideas to keep in the back pocket

Surfaced ideas the owner may pick up later. Agents: don't act on these unprompted.

- **RM overoptimization at small scale.** Reproduce Gao et al.'s headline plot on a 100M-param model. KL-vs-gold-reward curve.
- **DPO ablations.** Effect of preference-pair quality. What happens with noisy preferences? Synthetic adversarial pairs?
- **Constitutional AI critique-revise.** Does it cost helpfulness measurably on small models? Pick one constitution principle and ablate.
- **Weak-to-strong on a non-NLP task.** Vision classification? Tabular?
- **Sycophancy probe.** Replicate a known sycophancy eval on small open models. Does RLHF make it worse? Does DPO?
- **Read a paper "wrong" deliberately.** Pick a result and try to break it with adversarial ablations. Write up the failure modes.

---

## 6. Tooling

- **Python:** `uv` everywhere. `uv venv .venv`, `uv pip install ...`, `uv run python ...`.
- **GPUs / experiments:** the CORAL DS plugin is active. For any training run:
  - Instrument with `report_metric.sh` and `report_note.sh`.
  - Queue via `queue_experiment.sh`, never launch `python train.py` directly.
  - Guard `CUDA_VISIBLE_DEVICES` overrides in scripts.
  - Tell the owner to use `/ds:dash` and `/ds:queue` to monitor.
- **Citations in notes:** link to `papers/<slug>/notes.md` instead of re-pasting the abstract.

### Google Cloud Workspace

`gcp-workspace/` is local-only infrastructure state and must not be committed. Before any Google Cloud, Compute Engine, GPU, storage, backup, migration, startup, shutdown, or other infrastructure operation:

1. Read `gcp-workspace/SKILL.md`.
2. Read `gcp-workspace/infrastructure_manifest.json`.
3. Read `gcp-workspace/workspace_state.md`.
4. Read the most recent entries in `gcp-workspace/workspace_log.jsonl`.
5. Read open/recent entries in `gcp-workspace/workspace_sessions.csv` and `gcp-workspace/workspace_cost_ledger.csv`.
6. Compare the recorded state with live GCP resources before making changes.

`gcp-workspace/SKILL.md` is mandatory for infrastructure work. It governs discovery, resource reuse, explicit approval, backups, migrations, startup, shutdown, cleanup, security, and cost tracking.

Do not create a VM before checking existing resources. Do not run multiple GPU VMs without explicit approval. Do not delete or overwrite cloud resources without explicit approval. Do not expose credentials, SSH keys, `.env` contents, or secret files. Do not repeat timed-out lifecycle commands blindly. Do not leave GPU VMs running after work is complete.

After every verified infrastructure operation, update `gcp-workspace/workspace_state.md`, append one valid JSON record to `gcp-workspace/workspace_log.jsonl`, update applicable session/cost ledgers, and report running resources plus ongoing cost drivers.

---

## 7. Quick decision rules

| Thing | Goes in |
|---|---|
| A paper PDF | `papers/<slug>.pdf` (flat) |
| Reading notes on any paper | append a block to `papers/notes.md` |
| The learning plan | `docs/roadmap.md` |
| What I read this week | `docs/reading-log.md` (append) |
| Writeup of an experiment I ran | `docs/<slug>-writeup.md` |
| Code for that experiment | `src/<slug>/` |
| A cold-email draft | `docs/outreach/<recipient>.md` |
| A half-baked idea | Mention it in chat, don't file it |

---

## 8. TL;DR for the impatient agent

1. The owner is **learning**, not shipping. Teach.
2. Code is fine — but in **small chunks** (tens of lines), not 2000-line end-to-end drops. Build the pipeline across turns.
3. Boilerplate fast, conceptual code slow and explained.
4. End goal is **competence + cold emails**, not a published artifact.
5. Don't create files unprompted. Don't pad with polish.
6. Use `uv`. Use the CORAL queue for training. Respect the current roadmap phase.
7. Read and follow `gcp-workspace/SKILL.md` before any infrastructure action.
