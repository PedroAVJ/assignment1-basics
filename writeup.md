# CS336 Assignment 1 — Writeup

Answers to the written problems from `cs336_assignment1_basics.pdf`.
Code problems are checked by the test suite (`uv run pytest`) and are listed here
only as a progress tracker.

Legend: ✍️ = written answer goes here · 💻 = code, verified by tests · 📈 = experiment/training output

---

## §2 Tokenization

### ✍️ `unicode1` — Understanding Unicode (1 point)

**(a)** What Unicode character does `chr(0)` return? *(one sentence)*

> _Your answer:_

**(b)** How does this character's string representation (`__repr__()`) differ from its printed representation? *(one sentence)*

> _Your answer:_

**(c)** What happens when this character occurs in text? *(one sentence)*

> _Your answer:_

### ✍️ `unicode2` — Unicode Encodings (3 points)

**(a)** Why prefer training on UTF-8 encoded bytes rather than UTF-16 or UTF-32? *(one–two sentences)*

> _Your answer:_

**(b)** Give an input byte string for which the provided `decode_utf8_bytes_to_str_wrong` produces incorrect output, and explain why. *(example + explanation)*

> _Your answer:_

**(c)** Give a two-byte sequence that does not decode to any Unicode character(s). *(example + one sentence)*

> _Your answer:_

### 💻 `train_bpe` — BPE Tokenizer Training (15 points)

- [ ] Passes tests

### ✍️📈 `train_bpe_tinystories` — BPE Training on TinyStories (2 points)

**(a)** Training time, memory, longest token — does it make sense? *(one–two sentences)*

> _Your answer:_

**(b)** Profile: what part of the tokenizer training takes the most time? *(one–two sentences)*

> _Your answer:_

### ✍️📈 `train_bpe_expts_owt` — BPE Training on OpenWebText (2 points)

**(a)** Longest token in the OWT vocab — does it make sense? *(one–two sentences)*

> _Your answer:_

**(b)** Compare the TinyStories and OWT tokenizers. *(one–two sentences)*

> _Your answer:_

### 💻 `tokenizer` — Implementing the tokenizer (15 points)

- [ ] Passes tests

### ✍️📈 `tokenizer_experiments` — Experiments with tokenizers (4 points)

**(a)** Compression ratio (bytes/token) of each tokenizer on sample documents. *(one–two sentences)*

> _Your answer:_

**(b)** What happens when you tokenize OWT with the TinyStories tokenizer? *(one–two sentences)*

> _Your answer:_

**(c)** Estimate your tokenizer's throughput; how long to tokenize the Pile? *(one–two sentences)*

> _Your answer:_

**(d)** Why is `uint16` an appropriate dtype for the token ID arrays? *(one–two sentences)*

> _Your answer:_

---

## §3 Transformer LM

### 💻 `linear` — Linear module (1 point)

- [ ] Passes tests

### 💻 `embedding` — Embedding module (1 point)

- [ ] Passes tests

### 💻 `rmsnorm` — RMSNorm (1 point)

- [ ] Passes tests

### 💻 `positionwise_feedforward` — SwiGLU feed-forward (2 points)

- [ ] Passes tests

### 💻 `rope` — RoPE (2 points)

- [ ] Passes tests

### 💻 `softmax` — Softmax (1 point)

- [ ] Passes tests

### 💻 `scaled_dot_product_attention` — Scaled dot-product attention (5 points)

- [ ] Passes tests

### 💻 `multihead_self_attention` — Causal multi-head self-attention (5 points)

- [ ] Passes tests

### 💻 `transformer_block` — Transformer block (3 points)

- [ ] Passes tests

### 💻 `transformer_lm` — Transformer LM (3 points)

- [ ] Passes tests

### ✍️ `transformer_accounting` — Resource accounting (5 points)

**(a)** GPT-2 XL parameter count and memory to load it. *(one–two sentences)*

> _Your answer:_

**(b)** Matrix multiplies and total FLOPs for one forward pass. *(one–two sentences)*

> _Your answer:_

**(c)** FLOPs breakdown by component for GPT-2 small, medium, large, and XL.

> _Your answer:_

**(d)** GPT-2 XL with context length 16,384: how do FLOPs change? *(one–two sentences)*

> _Your answer:_

---

## §4 Training

### 💻 `cross_entropy` — Cross-entropy loss (1 point)

- [ ] Passes tests

### ✍️📈 `learning_rate_tuning` — Tuning the learning rate (1 point)

Behaviors observed when varying the learning rate in the toy SGD example. *(one–two sentences)*

> _Your answer:_

### 💻 `adamw` — AdamW optimizer (2 points)

- [ ] Passes tests

### ✍️ `adamw_accounting` — Resource accounting for AdamW (2 points)

**(a)** Peak memory: algebraic expressions for parameters, activations, gradients, optimizer state.

> _Your answer:_

**(b)** Memory as `a · batch_size + b`; max batch size in 80 GB.

> _Your answer:_

**(c)** FLOPs for one AdamW step. *(expression + justification)*

> _Your answer:_

**(d)** Training time for the given scenario. *(number + justification)*

> _Your answer:_

### 💻 `learning_rate_schedule` — Cosine schedule with warmup (1 point)

- [ ] Passes tests

### 💻 `gradient_clipping` — Gradient clipping (1 point)

- [ ] Passes tests

### 💻 `data_loading` — Data loading (2 points)

- [ ] Passes tests

### 💻 `training_together` — Training loop script (4 points)

- [ ] Script written and working

---

## §5–7 Generation, Experiments, Leaderboard

### 💻 `decoding` — Decoding / sampling (3 points)

- [ ] Implemented

### 📈 `experiment_log` — Experiment logging (3 points)

> _Link/path to your experiment log:_

### 📈 `learning_rate` — Tune the learning rate (3 points)

Learning curves for multiple LRs; explain the hyperparameter search. Target: validation loss ≤ 1.45 on TinyStories. Include at least one divergent run.

> _Your answer / curve links:_

### 📈 `batch_size_experiment` — Batch size variations (1 point)

Learning curves for different batch sizes + a few sentences of discussion.

> _Your answer:_

### 📈 `generate` — Generate text (1 point)

Text dump of ≥256 tokens + brief comment on fluency.

> _Your output:_

### 📈 `layer_norm_ablation` — Remove RMSNorm (1 point)

Learning curves with/without RMSNorm + commentary.

> _Your answer:_

### 📈 `pre_norm_ablation` — Post-norm vs pre-norm (1 point)

Learning curve comparison.

> _Your answer:_

### 📈 `no_pos_emb` — NoPE vs RoPE (1 point)

Learning curve comparison.

> _Your answer:_

### 📈 `swiglu_ablation` — SwiGLU vs SiLU (1 point)

Findings discussion.

> _Your answer:_

### 📈 `main_experiment` — Experiment on OWT (2 points)

OWT learning curve, comparison to TinyStories losses, generated text sample.

> _Your answer:_

### 📈 `leaderboard` — Leaderboard (6 points)

Final validation loss, learning curve with wall-clock x-axis < 45 min, description of modifications. Baseline to beat: 5.0 loss.

> _Your answer:_
