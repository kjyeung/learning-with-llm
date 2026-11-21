# Tutor state - Looped Transformers (Socratic tutor)

To resume: start a session, run `/socratic-tutor`, and say "resume from PROGRESS.md".
Update this file whenever a level changes or a notebook is finished.

Last updated: 2026-11-21 (end of session 1)

## Goal
Learn transformers from scratch by implementing each idea, then turn the tiny GPT into a
looped transformer. Target level 4/5 per subskill. Companion resource: Karpathy,
"Let's build GPT: from scratch, in code, spelled out" (Zero to Hero series), plus nanoGPT.

## Skill map (0 unaware . 1 heard of it . 2 can follow along . 3 can do with help .
## 4 can do independently . 5 can teach it / edge cases)

    A1 next-token prediction + loss  [######....]  3/5
    A2 tokens, embeddings, positions [####......]  2/5  <- current focus
    A3 self-attention                [##........]  1/5
    A4 multi-head attention          [..........]  0/5  untested
    A5 transformer block             [####......]  2/5
    A6 training loop                 [..........]  0/5  untested
    B7 weight-tied looping           [####......]  2/5
    B8 input injection               [..........]  0/5  untested
    B9 adaptive depth / halting      [..........]  0/5  untested
    B10 latent reasoning / fixed pts [..........]  0/5  untested

## Covered
- Diagnostic done (5 questions).
- A1: model outputs logits -> softmax -> probabilities; loss = -log(p_correct), averaged.
  Uniform baseline for vocab V is ln(V) (65 chars -> 4.17).
- 01_loss.ipynb: hand-computed cross-entropy matches F.cross_entropy; all-zeros -> 4.1744.

## Misconceptions / watch points
- A3: thinks attention "aggregates scores to generate a token" - doesn't yet know scores
  weight a sum of value vectors. Revisit.
- B7: said looping cuts params "to about 1/12" - ignores embedding params (in GPT-2 small,
  ~38M of 124M are embeddings). Said motivation is memory savings only; hasn't
  mentioned decoupling compute/depth from params. Probe later.
- Used log10 instead of ln once (fixed). Called -log "negative logistic" (fixed).
- Hidden-notebook-state bug (used undefined `x`); learned Restart -> Run All check.
- New to PyTorch: needs short API cheatsheets, not derivation.

## Setup
- `.venv/` (uv) with torch 2.15 (MPS available), numpy, ipykernel. Work in notebooks,
  one per topic: NN_topic.ipynb with a goals cell at the top.

## Covered (cont.)
- A1 init: logits = sigma*randn -> bigger sigma, bigger starting loss; sigma->0 -> uniform, ln(V).
  Symmetry problem (all-zero weights -> identical units forever). Recipe: small random hidden
  weights, output layer scaled ~0.01. Slip to watch: said "sum of probs" meaning sum of exp(logits).
- A1 -> 3/5 on 2026-11-21. Needs unprompted recall later for 4/5.

- 02_tokenizer.ipynb done: char vocab (sorted set), stoi/itos, encode/decode, round trip True,
  corpus as long tensor (1,115,394). Gaps: no len(vocab)==65 check, encode("hello") never run.
- A2: raw IDs impose false order/closeness -> one-hot (0-based index; once said 45 for ID 46) ->
  one_hot @ W (65,d) = row lookup (first said "column", fixed) = nn.Embedding. Explained similar
  contexts -> similar logits -> gradient pulls rows together. Recalled sinusoidal PE from paper,
  added to token embedding.
- A2 -> 2/5 on 2026-11-21.
- Open edge case for later (level 5): can two different embedding rows give identical logits?
  (yes, difference in null space of W_out)

## Misconception in progress
- Learned positional embedding: user proposed a (65 vocab x 8 positions) table, one param per
  (char, position). Correct: table of shape (block_size, d) indexed by position only, added to
  token embedding. Do NOT give the answer; continue from the pending question.

## Pending question (ask this first on resume)
For the `l` at position 3, does your (65 x 8) table return a single number or a vector? How would
you add that to a d-dimensional embedding? Does the table need to know the char is `l`, given the
token embedding already carries that?
After A2: build embeddings (token + position) in a notebook 03, then A3 self-attention.
