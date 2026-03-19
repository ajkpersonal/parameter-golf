This record captures the planned `Optimization Baseline`.

This is a planning artifact for `codex/ajk-test`, not a scored submission yet.

Trainer changes in this planned snapshot:
- start from the current repository `train_gpt.py` and the official SP-1024 baseline setup
- keep the published `fineweb10B_sp1024` dataset and tokenizer unchanged
- keep the `10` minute wallclock cap on `8xH100`
- disable periodic validation during training so the budget is spent on updates, then run the normal full final validation once
- keep the same export path and final evaluation path as the official baseline
- change the model to moderate depth recurrence
- add a late-phase compression-aware objective matched to the final int8+zlib roundtrip target

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_UNIQUE_LAYERS=6 NUM_RECURRENCE=2 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Validation during training: `VAL_LOSS_EVERY=0`
- Planned compression-aware settings: `COMPRESSION_AWARE_MODE=shadow_int8_kl COMPRESSION_AWARE_START_FRAC=0.60 COMPRESSION_AWARE_EVERY=4 COMPRESSION_AWARE_KL_WEIGHT=0.10`

Planned command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=ajk_simple_compression_aware_baseline \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=6 \
NUM_RECURRENCE=2 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=0 \
COMPRESSION_AWARE_MODE=shadow_int8_kl \
COMPRESSION_AWARE_START_FRAC=0.60 \
COMPRESSION_AWARE_EVERY=4 \
COMPRESSION_AWARE_KL_WEIGHT=0.10 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

Design rationale:
- The official naive baseline is the control: `9` unique blocks at `512` width, score `final_int8_zlib_roundtrip_exact val_bpb:1.22436570`, and total artifact `15,863,489` bytes.
- This optimization baseline is intended to be the first practical improvement path, not the purest ablation. The main idea is to trade unique parameters for repeated computation: fewer stored blocks, similar per-step compute, smaller artifact.
- Moderate recurrence is deliberately conservative here: `6` unique blocks looped `2x` gives `12` effective layers without collapsing to a single repeatedly used block.
- The first objective is not to beat the baseline immediately on score. The first objective is to preserve roughly baseline-quality post-roundtrip behavior while materially reducing artifact size.
- If that works, the second step is straightforward: spend the saved artifact budget on width or other capacity.
- Compression-aware training is included from the start because recurrence is exactly the kind of change that could improve pre-quant behavior but lose some of that gain after the int8+zlib roundtrip.

Planned implementation note:
- `COMPRESSION_AWARE_MODE=shadow_int8_kl` means the model will periodically run a quantized-shadow forward that mimics the final per-row int8 export target, then add a KL term from the normal logits to the quantized-shadow logits.
- These environment variables are part of the planned baseline and are not present in the repository yet.
- `NUM_UNIQUE_LAYERS` and `NUM_RECURRENCE` are also planned additions and are not present in the repository yet.

Pseudocode plan:
```text
Args:
  num_unique_layers = 6
  num_recurrence = 2
  compression_aware_mode = shadow_int8_kl
  compression_aware_start_frac = 0.60
  compression_aware_every = 4
  compression_aware_kl_weight = 0.10

Model construction:
  build tok_emb, final_norm, lm_head exactly as today
  build blocks = [Block(...) for _ in range(num_unique_layers)]
  build skip_weights for num_unique_layers
  effective_layers = num_unique_layers * num_recurrence

Forward logits without quantized shadow:
  x = tok_emb(input_ids)
  x = rms_norm(x)
  x0 = x
  skips = []
  for i in range(num_unique_layers):
      x = blocks[i](x, x0)
      skips.append(x)
  for i in range(num_unique_layers):
      x = x + skip_weights[i] * skips.pop()
      x = blocks[i](x, x0)
  x = final_norm(x)
  logits = tied or untied output projection

Forward logits with quantized shadow:
  same control flow as above
  but any tensor that would be quantized by the final exporter uses
  fake_quantize_like_export(weight) before its matmul or embedding lookup
  small passthrough tensors stay unchanged, matching export behavior

Training step:
  float_logits = model.forward_logits(x, use_export_quant=False)
  ce_loss = cross_entropy(float_logits, targets)
  loss = ce_loss

  active = elapsed_wallclock_frac >= compression_aware_start_frac
           and step % compression_aware_every == 0
  if active:
      teacher_probs = softmax(float_logits.detach())
      shadow_logits = model.forward_logits(x, use_export_quant=True)
      kl_loss = kl_div(log_softmax(shadow_logits), teacher_probs)
      loss = ce_loss + compression_aware_kl_weight * kl_loss

  backward(loss)
  optimizer_step()

Fake export quantization helper:
  if tensor is non-floating: return tensor unchanged
  if tensor.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL: return tensor unchanged
  q, s = quantize_float_tensor(tensor)
  deq = dequantize(q, s) in original dtype
  return tensor + (deq - tensor).detach()
```

Success criteria for the first full run:
- stay near the official naive baseline on the exact final roundtrip metric
- materially reduce total submission size versus the official naive baseline
- stay under `16,000,000` total submission bytes
- finish within the `600` second wallclock cap on `8xH100`
- preserve stable training behavior and a normal final full validation pass

Deliberately excluded from baseline v1:
- wider model dimensions
- SwiGLU or other MLP changes
- tokenizer changes
- factorized embeddings
- MQA or more aggressive attention compression
- hybrid linear/recurrent attention
- mixed-precision or sub-8-bit export changes
- multi-token prediction
- test-time adaptation or test-time training

Appendix: next versions to try after baseline v1
- spend any recovered artifact budget on width, likely `MODEL_DIM=576` or `640`
- pure compression-aware training on the original `9x512` baseline as a control
- baseline plus gated MLP replacement (`SwiGLU`) if recurrence creates size headroom
- baseline plus mild outlier regularization during the compression-aware phase
- baseline plus a slightly smarter family-aware exporter once the training-side effect is isolated

Included files:
- `README.md` (this planning record)
- `submission.json` (planned metadata with pending metrics)
