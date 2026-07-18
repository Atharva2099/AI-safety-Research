# Reading notes

Last updated: 2026-07-18

One block per paper, in reading order. Separator: `---`.

Each block: PDF link, vocab pre-check (for foundational papers), per-section notes filled in *as* you read, open questions, extension ideas. Don't pre-fill — empty fields are fine until you've read it.

Convention:
- Status: ☐ not started · ◐ in progress · ☑ done
- Open questions: specific enough that a researcher could answer them. "I don't get §4" doesn't count.

---

## Phase 1 — Orientation

Start here. Goal: vocabulary + threat models, not technique mastery.

---

### Amodei et al. 2016 — Concrete Problems in AI Safety ☐

[PDF](amodei2016-concrete-problems.pdf) · arXiv 1606.06565 · ~29 pages

The canonical first safety paper. Survey-style, 5 concrete failure modes with toy examples.

**Vocab pre-check** — write what you think these mean cold.
- Reward hacking:
- Negative side effects:
- Scalable oversight:
- Safe exploration:
- Distributional shift:

**Per-problem notes** — one paragraph each, the clearest toy example + why it's hard.
- Avoiding negative side effects:
- Reward hacking:
- Scalable oversight:
- Safe exploration:
- Robustness to distributional shift:

**Vocab post-check** — rewrite the ones you got wrong. Keep the originals.

**Open questions:**
-

**Which of these 5 feels most relevant to "alignment techniques" (RLHF/DPO/etc.) and why:**

---

### Anthropic 2023 — Core Views on AI Safety ☐

Blog. [anthropic.com/news/core-views-on-ai-safety](https://www.anthropic.com/news/core-views-on-ai-safety)

- The optimistic / intermediate / pessimistic framing in my words:
- Which alignment techniques they emphasize and why:
- What this tells me about the "alignment techniques" research agenda I'm aiming at:
- Open questions:

---

### Christiano 2019 — What failure looks like ☐

Blog. [Alignment Forum](https://www.alignmentforum.org/posts/HBxe6wdjxK239zajf/what-failure-looks-like)

- "Going out with a whimper" in my words:
- "Going out with a bang" in my words:
- Which story do I find more compelling and why:
- Connects to Amodei 2016's reward-hacking section how:
- Open questions:

---

### Leike et al. 2018 — Scalable agent alignment via reward modeling ☐

[PDF](leike2018-reward-modeling.pdf) · arXiv 1811.07871 · ~28 pages

DeepMind's pre-RLHF framing of the reward-modeling agenda. Bridges Phase 1 → Phase 2.

- Reward modeling in my words (1 paragraph):
- "Recursive reward modeling" — what is it and why might it scale?
- Open problems they list — which 2 feel most important to me:
- Open questions:
- Connects to Christiano 2017 (next paper) how:

---

## Phase 2 — Core papers

In order. Each one builds on the previous.

---

### Christiano et al. 2017 — Deep RL from Human Preferences ☐

[PDF](christiano2017-deep-rl-human-preferences.pdf) · arXiv 1706.03741

- Setup (env, agent, RM, preference collection) in 3 sentences:
- How is the reward model trained from pairwise preferences?
- Why use a reward model at all instead of direct human reward?
- Key results / what surprised me:
- Open questions:
- Extension ideas:

---

### Stiennon et al. 2020 — Learning to summarize from human feedback ☐

[PDF](stiennon2020-summarize-hf.pdf) · arXiv 2009.01325

- What changes vs. Christiano 2017 when the domain becomes language:
- KL penalty against the reference model — why?
- Where does the model beat human reference summaries?
- Open questions:
- Extension ideas:

---

### Ouyang et al. 2022 — InstructGPT ☐

[PDF](ouyang2022-instructgpt.pdf) · arXiv 2203.02155

- SFT → RM → PPO pipeline in my words:
- Alignment tax — what is it, what causes it, how do they mitigate?
- Failure modes they document:
- Open questions:
- Extension ideas:

---

### Bai et al. 2022 — Constitutional AI ☐

[PDF](bai2022-constitutional-ai.pdf) · arXiv 2212.08073

- Critique-revise loop in my words:
- RLAIF: how does the model substitute for human labelers?
- Helpfulness vs. harmlessness tradeoff — what do they find?
- Open questions:
- Extension ideas:

---

### Rafailov et al. 2023 — DPO ☐

[PDF](rafailov2023-dpo.pdf) · arXiv 2305.18290

- The derivation in my words: from RLHF objective → closed-form policy → DPO loss.
- Why no explicit reward model? Where did it go?
- When does DPO empirically beat / lose to PPO?
- Open questions:
- Extension ideas:

---

### Gao et al. 2022 — Scaling laws for RM overoptimization ☐

[PDF](gao2022-rm-overopt.pdf) · arXiv 2210.10760

- Proxy reward vs. gold reward as a function of KL — the headline plot in my words:
- BoN vs. RL overoptimization curves — how do they differ?
- Implications for picking KL coefficients in practice:
- Open questions:
- Extension ideas:

---

### Hubinger et al. 2019 — Risks from Learned Optimization ☐

[PDF](hubinger2019-risks-learned-optimization.pdf) · arXiv 1906.01820 · ~75 pages

Placed here (not Phase 1) because deceptive alignment is easier to grasp after you've seen reward hacking and overoptimization in concrete papers. Structured as 4 mini-papers — read one at a time.

**Vocab pre-check** — write what you think these mean cold.
- Base optimizer:
- Mesa-optimizer:
- Base objective:
- Mesa-objective:
- Inner alignment:
- Outer alignment:
- Pseudo-alignment:
- Deceptive alignment:

**Part 1 — Mesa-optimization (§1–2)**
- One-paragraph definition in my own words:
- Clearest example they give:
- Why this is different from "the policy is good at the task":
- Surprised me:
- Confused me:

**Part 2 — Conditions for mesa-optimization (§3)**
- Factors they argue make it more likely:
- Strongest / weakest:
- A real ML setup where these conditions plausibly hold:
- Confused me:

**Part 3 — The inner alignment problem (§4)**
- Why "high reward on training distribution" isn't enough:
- Pseudo-alignment taxonomy in my words:
- ML example where a learned policy has a proxy objective:
- Confused me:

**Part 4 — Deceptive alignment (§5)**
- The story in ≤6 sentences:
- What has to be true about the mesa-optimizer for deception to be instrumentally rational?
- Why is this hard to detect from training behavior alone?
- Confused me:

**Vocab post-check** — rewrite the ones you got wrong.

**Open questions:**
-

**Extension ideas:**
-

---

### Irving et al. 2018 — AI Safety via Debate ☐

[PDF](irving2018-debate.pdf) · arXiv 1805.00899

- The debate game setup in my words:
- Why might debate scale supervision beyond what a single human can verify?
- Failure modes the authors flag:
- Open questions:
- Extension ideas:

---

### Bowman et al. 2022 — Measuring Progress on Scalable Oversight ☐

[PDF](bowman2022-scalable-oversight.pdf) · arXiv 2211.03540

- "Sandwiching" methodology in my words:
- What do their experiments find?
- Connects to debate (Irving 2018) how:
- Open questions:
- Extension ideas:

---

### Burns et al. 2023 — Weak-to-strong generalization ☐

[PDF](burns2023-weak-to-strong.pdf) · arXiv 2312.09390

- The setup (weak supervisor → strong student) in my words:
- What is the "weak-to-strong gap" and how is it measured?
- Why is this a (toy) model of superalignment?
- Open questions:
- Extension ideas:

---

### Hubinger et al. 2024 — Sleeper Agents ☐

[PDF](hubinger2024-sleeper-agents.pdf) · arXiv 2401.05566

- The backdoor construction in my words:
- Which safety-training methods do/don't remove the backdoor?
- Connects to Hubinger 2019 (deceptive alignment) how:
- Open questions:
- Extension ideas:
