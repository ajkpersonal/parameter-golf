# Parameter Golf Handoff

Date: 2026-03-20  
Workspace: `/workspace/parameter-golf`  
Primary local branch head: `40593b3`

## Purpose

This document is a self-contained technical handoff for the OpenAI Parameter Golf challenge. It is meant to let a new reader understand:

- what the challenge is
- what the real objective and constraints are
- what code and infrastructure exist in this workspace
- what experiments have already been run
- what worked, what failed, and what seems most promising next
- which ideas from strong external submissions are worth reimplementing

The goal is that someone could read only this document and then offer high-quality suggestions or continue the work effectively.

## Current status

This handoff contains a lot of historical experiment detail, but the active mainline is now much narrower.

Current live priorities:

- dense `9x512 KV4` remains the main architecture lane
- the next serious retrain sequence should be:
  - first, apply the improved integrated exporter to the saved strong `seq2048` `MLP2` checkpoint
  - then run two integrated retrains on the same `seq2048` lane:
    - `MLP_MULT=2`
    - `MLP_MULT=3`
- the integrated dense stack to compare is:
  - `TRAIN_SEQ_LEN=2048`
  - bulk `int6` on `.mlp.`, `.attn.c_q.`, `.attn.c_v.`, `.attn.proj.`
  - `zstd` outer compression
  - `fp16` passthrough for `tok_emb.weight`
  - selective late-`K` preservation on `blocks.7/8.attn.c_k.weight`
  - grouped int8 for the remaining `c_k`
  - PR-style optimizer settings around `0.02/0.02/0.03`, Muon `0.99`, and `clip=0.3`
- evaluation is still important, but it is no longer the mainline research question until the training-side base is competitive

Implementation status:

- the integrated dense lane is now wired directly into [train_gpt.py](/workspace/parameter-golf/train_gpt.py)
- new serializer controls in the main trainer:
  - `LOWBIT_BITS`
  - `LOWBIT_NAME_PATTERNS`
  - `INT8_KEEP_FLOAT_NAME_PATTERNS`
  - `INT8_GROUP_OVERRIDES`
  - `SERIAL_COMPRESSOR`
- smoke-tested end to end:
  - [integrated_lane_smoke_a.txt](/workspace/parameter-golf/logs/integrated_lane_smoke_a.txt)
  - final exact roundtrip line: `final_mixed_int6_zstd_roundtrip_exact val_bpb:1.96079433`
  - artifact files now written by the trainer:
    - [final_model.quant.ptz](/workspace/parameter-golf/final_model.quant.ptz)
    - [final_model.int8.ptz](/workspace/parameter-golf/final_model.int8.ptz) as a legacy alias for local helper scripts
- prefilled launcher for the new lane:
  - [run_dense_integrated_lane.sh](/workspace/parameter-golf/scripts/run_dense_integrated_lane.sh)
- [checkpoint_frontier_sweep.py](/workspace/parameter-golf/scripts/checkpoint_frontier_sweep.py) now reuses the integrated serializer path from [train_gpt.py](/workspace/parameter-golf/train_gpt.py) instead of its older standalone low-bit approximation
- first bounded exporter-transfer check on the saved seq2048 winner:
  - [seq2048_export_transfer_smoke.csv](/workspace/parameter-golf/logs/seq2048_export_transfer_smoke.csv)
  - variant: `int6_zstd_core_fp16_embed_latek`
  - evaluator: `stream_sliding 2048/256`
  - bounded result on first `20` docs: `1.37096764`
  - artifact size: `12,956,566` bytes

What is now effectively archived:

- recurrence as the primary branch
- basis/codebook/sidecar moonshots
- compression-aware training on the current codebase
- doc-isolated evaluation as a pillar
- blind local `40m` confirmations on the single-shard proxy without schedule retuning

Two checkpoints remain worth remembering:

- best measured score so far:
  - dense `seq2048` `12m` checkpoint with `stream_sliding 2048/256`
  - `1.30203861`
  - `15,853,594` bytes
- best byte-headroom signal so far:
  - dense `MLP2` checkpoint with `int6 + zstd + fp16 tied embedding`
  - `1.30709551`
  - `12,336,039` bytes
  - this is the clearest proof that low-bit export can fund a larger dense retrain

## 1. The task from scratch

OpenAI Parameter Golf asks participants to train the best language model that:

- fits under a total submission artifact size of `16,000,000` bytes
- runs in under `10 minutes` on `8x H100` GPUs for training and evaluation
- is evaluated on FineWeb validation compression in tokenizer-agnostic `bits per byte` (`val_bpb`)

The main public repo entry point is [README.md](/workspace/parameter-golf/README.md).

Important practical interpretation:

- lower `val_bpb` is better
- the counted artifact is `code bytes + compressed model bytes`
- evaluation is allowed to do nontrivial test-time computation, as long as it still fits the time limit
- submissions should live in `records/`, but core repo code can be used for local iteration

At the time of writing, the public README still lists the naive baseline as the official leaderboard entry:

- `Naive Baseline`
- `val_bpb = 1.2244`
- `9 layers`, `512 dim`, `1024 vocab`, tied embeddings, `4 KV heads`

That means our local work should be judged against two references:

1. the current public baseline around `1.2244`
2. the much better record-style PRs that report around `1.1605`

## 2. What matters technically

This challenge is not just “make loss lower.”

There are at least four interacting bottlenecks:

1. Capacity under a hard artifact-size budget  
   A model can be larger in effective capacity than its stored parameter count if it uses better compression, weight sharing, test-time compute, or smarter evaluation.

2. Training throughput under a hard wallclock cap  
   Bigger or fancier models are not automatically better if they train too slowly and therefore take fewer optimizer steps.

3. Export quality  
   Some architectures train fine but degrade badly when compressed.

4. Evaluation method  
   The challenge allows aggressive evaluation tricks so long as they fit the time budget and artifact budget. Sliding-window evaluation is one example of this.

The strongest public record-style submissions lean heavily on this full stack:

- better low-bit export than plain int8
- selective precision for sensitive tensors
- more capacity where bytes matter most
- evaluation methods that improve BPB without breaking the runtime budget

## 3. Local environment and constraints

Current local setup:

- `2x NVIDIA H100 80GB HBM3`
- repo-local Python environment in `.venv`
- `torch 2.10.0+cu128`
- local dataset/tokenizer under:
  - `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024`
  - `/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

Important caveat:

- local training currently uses only `1` training shard for quick iteration
- local “full” runs used `40m` wallclock as a rough stand-in for the real `10m on 8x H100`
- this is directionally useful, but not an exact simulation of leaderboard conditions

## 4. Current code in this workspace

Main trainer:

- [train_gpt.py](/workspace/parameter-golf/train_gpt.py)

Archived experimental trainers kept only as references:

- [train_gpt_basis.py](/workspace/parameter-golf/train_gpt_basis.py)
- [train_gpt_codebook.py](/workspace/parameter-golf/train_gpt_codebook.py)
- [train_gpt_sidecar.py](/workspace/parameter-golf/train_gpt_sidecar.py)

Runners added locally:

- [run_overnight_moonshots.sh](/workspace/parameter-golf/scripts/run_overnight_moonshots.sh)
- [run_proxy_screen.sh](/workspace/parameter-golf/scripts/run_proxy_screen.sh)
- [run_dense_calibration.sh](/workspace/parameter-golf/scripts/run_dense_calibration.sh)
- [run_dense_integrated_lane.sh](/workspace/parameter-golf/scripts/run_dense_integrated_lane.sh)
- [checkpoint_frontier_sweep.py](/workspace/parameter-golf/scripts/checkpoint_frontier_sweep.py)

Important local fix that already happened:

- Dense/non-recurrent DDP in `train_gpt.py` had a bug where the second half of the network reused the wrong block indices and left later blocks without gradients.
- That bug was fixed locally in:
  - [train_gpt.py](/workspace/parameter-golf/train_gpt.py)
  - [train_gpt_basis.py](/workspace/parameter-golf/train_gpt_basis.py)
  - [train_gpt_codebook.py](/workspace/parameter-golf/train_gpt_codebook.py)
  - [train_gpt_sidecar.py](/workspace/parameter-golf/train_gpt_sidecar.py)

Current export path in the main trainer:

- default behavior remains the clean baseline path: int8 + `zlib`
- the trainer now also supports an integrated dense record-style lane via env flags:
  - selective bulk `int6`
  - grouped int8 overrides
  - explicit fp16 passthrough for sensitive tensors
  - `zstd` outer compression
  - full roundtrip validation on the configured artifact, not just post-hoc sweep-only export

This closes one of the biggest earlier gaps: the first `MLP3` tests were being judged with a sweep-only low-bit path rather than a real integrated training/export lane.

Current intended dense mainline config:

- `NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`
- `GRAD_CLIP_NORM=0.3`
- `WARMDOWN_ITERS=3000`
- `QK_GAIN_INIT=1.7`
- `LOWBIT_BITS=6`
- `LOWBIT_NAME_PATTERNS=.mlp.,.attn.c_q.,.attn.c_v.,.attn.proj.`
- `INT8_KEEP_FLOAT_NAME_PATTERNS=tok_emb.weight,blocks.7.attn.c_k.weight,blocks.8.attn.c_k.weight`
- `INT8_GROUP_OVERRIDES=.attn.c_k.:64`
- `SERIAL_COMPRESSOR=zstd`

One extra environment note:

- `zstandard` is installed in the repo-local `.venv`, so local `zstd` export experiments are now possible.

## 5. What we tried before the overnight batch

### 5.1 Optimized recurrent baseline

We tested the most recent local optimized baseline based on recurrence and compression-aware training.

Log:

- [opt_baseline_2gpu_40m_20260319.txt](/workspace/parameter-golf/logs/opt_baseline_2gpu_40m_20260319.txt)

Result:

- `val_bpb = 1.36603550`
- `bytes_total = 10,783,674`

Interpretation:

- very small artifact
- did not beat the public naive baseline
- recurrence preserved size headroom but quality was not competitive enough by itself

## 6. Overnight full-run batch

We then ran a 12-job overnight batch covering:

- dense controls
- recurrent controls
- basis-sharing experiments
- codebook-style compression experiments
- sidecar experiments

Source:

- [overnight_mar19_fix1_summary.csv](/workspace/parameter-golf/logs/overnight_mar19_fix1_summary.csv)

### 6.1 Full results

| run | trainer | val_bpb | bytes_total | comment |
|---|---|---:|---:|---|
| `dense_control_9x512_kv4_m2` | `train_gpt.py` | `1.37892873` | `15,893,764` | local dense control |
| `recur_control_6u2r_512_kv4_m2` | `train_gpt.py` | `1.36929268` | `10,783,610` | main recurrent control |
| `dense_bulk_squeeze_9x640_kv2_m1` | `train_gpt.py` | `1.34181490` | `15,968,537` | best current-branch full run |
| `recur_bulk_squeeze_6u2r_768_kv2_m1` | `train_gpt.py` | `2.57477576` | `9,649,594` | catastrophic recurrent bulk squeeze |
| `basis_mlp_only_dense` | `train_gpt_basis.py` | `1.43249706` | `7,672,701` | not competitive |
| `basis_mlp_plus_delta_recur` | `train_gpt_basis.py` | `1.46353494` | `5,458,804` | not competitive |
| `basis_full_block_recur` | `train_gpt_basis.py` | `2.72571532` | `1,537,138` | much worse |
| `codebook_export_only_recur` | `train_gpt_codebook.py` | `6.14896959` | `5,464,281` | unusable in current form |
| `codebook_shadow_recur` | `train_gpt_codebook.py` | `10.00194411` | `5,313,443` | unusable |
| `codebook_shadow_selective_precision_recur` | `train_gpt_codebook.py` | `11.12727461` | `5,309,778` | unusable |
| `dense_sidecar_4gram_bias` | `train_gpt_sidecar.py` | `1.37783300` | `21,875,584` | over cap |
| `recur_sidecar_4gram_bias` | `train_gpt_sidecar.py` | `2.19396057` | `15,825,611` | under cap but poor quality |

### 6.2 Main lessons from the overnight batch

1. The best current-branch result is not a moonshot.  
   The winner was a denser, wider model with `KV2` and `MLP_MULT=1`, not basis/codebook/sidecar tricks.

2. Recurrence itself is not dead.  
   The simple recurrent control beat the simple dense control, but it was under budget by roughly `5 MB`.

3. Aggressive recurrent bulk squeeze is bad.  
   `6u2r_768_kv2_m1` collapsed badly, which suggested that cutting recurrence MLP capacity too hard was the wrong move.

4. Basis/codebook/sidecar families are not ready.  
   These ideas may still matter later, but their local implementations were not remotely competitive enough to justify near-term focus.

5. Compression-aware training has not earned its complexity.  
   We do not have evidence that it helps enough on this branch to be worth the training slowdown and added noise.

## 7. Short proxy screen

To avoid committing every idea to a `40m` local run, we built a proxy runner.

Runner:

- [run_proxy_screen.sh](/workspace/parameter-golf/scripts/run_proxy_screen.sh)

Results:

- [proxy_mar19_a_summary.csv](/workspace/parameter-golf/logs/proxy_mar19_a_summary.csv)

Proxy setup:

- `MAX_WALLCLOCK_SECONDS=720`
- `WARMUP_STEPS=4`
- `VAL_LOSS_EVERY=0`
- `COMPRESSION_AWARE_MODE=off`
- final exact roundtrip eval preserved

### 7.1 Proxy results

| run | val_bpb | bytes_total | takeaway |
|---|---:|---:|---|
| `dense_anchor_9x640_kv2_m1` | `1.36148663` | `15,950,523` | dense anchor |
| `recur_anchor_6u2r_512_kv4_m2` | `1.35839009` | `10,725,244` | recurrent anchor |
| `recur_wide_6u2r_576_kv4_m2` | `1.35118640` | `13,140,902` | clear improvement |
| `recur_wide_6u2r_608_kv4_m2` | `1.34907255` | `14,264,252` | slightly better still |
| `recur_wide_6u2r_640_kv2_m2` | `1.40163487` | `14,893,918` | bad |
| `recur_unique_7u2r_576_kv4_m2` | `1.34791587` | `14,873,796` | best short-run recurrent config |

### 7.2 What the proxy taught us

1. The recurrent branch improves when it spends more of the byte budget.  
   That was the most important answer from the proxy screen.

2. `7u2r_576_kv4_m2` is the strongest current-branch recurrent candidate.  
   `6u2r_608_kv4_m2` is a very close second.

3. `KV2` is a bad lever for the recurrent line in this regime.  
   It worked in the dense winner but did not transfer to recurrence.

4. The proxy is useful for pruning, not for final judgment.  
   Larger/slower models get fewer steps in a fixed `12m` budget, so small margins should still be confirmed with full runs.

## 8. What we learned about LR sweeps

We discussed whether to run `30s` experiments just to kill bad learning rates.

Conclusion:

- `30s` runs are fine only as a coarse instability filter
- they are not reliable for BPB-based ranking
- they are biased by startup cost and throughput differences

Recommended experiment ladder:

1. `2-3m` train-loss-only sweeps for LR/momentum/warmup sanity
2. `10-12m` proxy runs with the real exporter and exact final roundtrip eval
3. `40m` confirmation runs for only the best candidates

## 9. Strong external submissions worth learning from

The user pointed to three upstream PRs. We fetched and inspected them locally, but the point here is not to copy them wholesale. The point is to identify the ideas worth reimplementing.

External references:

- Long Context Seq2048 v2: `records/track_10min_16mb/2026-03-18_LongContextSeq2048`
- PR 88: https://github.com/openai/parameter-golf/pull/88
- PR 99: https://github.com/openai/parameter-golf/pull/99
- PR 102: https://github.com/openai/parameter-golf/pull/102

### 9.0 Long Context Seq2048 v2

Record folder:

- `records/track_10min_16mb/2026-03-18_LongContextSeq2048`

Reported result:

- `val_bpb = 1.20576485`
- `15.87 MB`

Core ideas:

- same basic `9x512, kv4, mlp_mult=2` dense recipe
- `TRAIN_SEQ_LEN=2048`
- same int8 + `zlib` style export family, not an exotic serializer
- slightly reduced learning rates:
  - `TIED_EMBED_LR=0.04`
  - `MATRIX_LR=0.032`
  - `SCALAR_LR=0.032`

Why it matters:

- it is a strong quality gain from runtime/memory budget, not artifact-byte tricks
- it shows long context can be very valuable even without int6 or sliding-window evaluation
- it is one of the cleanest ideas to combine with a small-under-cap model

Important caveat:

- this idea does not spend artifact bytes; it spends training/eval time and GPU memory
- that makes it compatible with a small model, but not automatically compatible with a slow model
- recurrence plus seq2048 is plausible, but must be tested carefully because both increase runtime pressure

### 9.1 PR 88

Record folder:

- `records/track_10min_16mb/2026-03-19_Int6_MLP3x_MTP_SlideEval`

Reported result:

- `val_bpb = 1.1605`
- `15.28 MB`

Core ideas:

- int6 quantization
- `zstd` instead of `zlib`
- `3x` MLP
- long-context training at `4096`
- MTP auxiliary head
- fp16 tied embedding passthrough
- sliding-window evaluation
- tuned optimizer schedule

Judgment:

- strongest raw package
- also the heaviest and most complex to reimplement
- best viewed as a rich source of ideas rather than the first thing to reproduce feature-for-feature

### 9.2 PR 99

Record folder:

- `records/track_10min_16mb/2026-03-19_Seed2025_Top2K_Stride64`

Reported result:

- `val_bpb = 1.16050360`
- `15.84 MB`

Core ideas:

- int6 mixed quantization
- `zstd`
- `MLP_MULT=3`
- fp16 tied embedding
- selective late-layer `c_k` fp16 passthrough
- grouped int8 on remaining `c_k`
- sliding-window eval at `stride=64`

Judgment:

- best practical idea bundle to replicate first
- strong result without the extra complexity of MTP or `4096` training length
- especially aligned with our own intuition that selective precision is valuable

### 9.3 PR 102

Record folder:

- `records/track_10min_16mb/2026-03-19_Int6_MLP3x_TunedLR_SmearGate_SlidingWindow`

Reported result:

- `val_bpb = 1.1618`
- `15.14 MB`

Core ideas:

- int6 mixed quantization
- `zstd`
- `MLP_MULT=3`
- tuned optimizer schedule
- fp16 tied embedding
- SmearGate
- sliding-window eval

Judgment:

- very strong
- cleaner than PR 88
- likely the best donor for optimizer settings and cheap architectural tweaks

## 10. Best ideas to replicate without directly pulling a PR

If we want to stay in the spirit of “do not pull any PR directly,” these are the best ideas to reimplement on our own.

### 10.1 High-priority ideas

1. Mixed int6 export for the big matrices  
   Current local mainline still uses int8+zlib. The record-style lanes win by freeing several MB of artifact budget. That extra room is then spent on real capacity.

2. `zstd` instead of `zlib` for low-bit artifacts  
   Especially helpful when the low-bit representation leaves compressible structure in the serialized bytes.

3. Wider MLP (`MLP_MULT=3`) on the dense baseline  
   This is the single most consistent “spend bytes where quality matters” idea in the strong public lanes.

4. Sliding-window evaluation  
   This is clearly part of the winning stack. It directly improves BPB at evaluation time.

5. Selective fp16 passthrough for the most sensitive large tensors  
   Tied embedding is the cleanest starting point. Late-layer `K` matrices are a strong next candidate.

6. Tuned optimizer dynamics  
   Lower LR, higher Muon momentum, and longer warmdown appear consistently in the stronger lanes.

7. Longer training context (`TRAIN_SEQ_LEN=2048`)  
   The seq2048 record shows that context length alone can buy meaningful quality while leaving the artifact format unchanged.

### 10.2 Medium-priority ideas

1. SmearGate  
   Cheap, simple, and plausible enough to try after the quantization/eval stack is improved.

2. Grouped quantization overrides for selected tensors  
   Especially on attention `K` or other sensitive matrices.

3. More aggressive sequence-length experiments  
   Long-context training clearly helped in the seq2048 lane, but it is more expensive to reason about locally and must be balanced against slower models.

### 10.3 Lower-priority ideas for now

1. MTP  
   Interesting, but more moving parts and harder to debug.

2. Recurrence + aggressive export changes together  
   Recurrence is alive, but it is already a different variable. Mixing it immediately with a brand new int6/zstd stack makes attribution much harder.

3. Basis/codebook/sidecar ideas  
   Good long-term research bets, bad short-term optimization bets based on current evidence.

## 11. What seems most promising now

There are really two credible paths forward.

### Path A: Dense int6/zstd mainline

This is the most likely path to near-term leaderboard relevance.

Suggested stack:

- dense `9x512`-style base
- `MLP_MULT=3`
- possibly `TRAIN_SEQ_LEN=2048` once the plain int6/zstd path is stable
- int6 mixed export
- `zstd`
- fp16 tied embedding
- sliding-window eval
- tuned LR / momentum / warmdown
- later selective fp16 on late-layer `K`
- maybe later SmearGate

Why this is promising:

- it is directly aligned with the strongest public results
- it does not require believing in a novel architecture thesis first
- the ideas are modular and easy to ablate

### Path B: Recurrent cap-filling branch

This is a real research branch, but probably not the mainline until the dense int6/zstd path is understood.

Best current candidates:

- `7u2r_576_kv4_m2`
- `6u2r_608_kv4_m2`

Why this is still interesting:

- recurrence was the only structural compression trick that did not collapse
- it clearly benefits from using more of the available byte budget
- it may be able to borrow seq2048-style gains because that improvement spends runtime/memory rather than artifact bytes
- it might combine well with stronger export methods later

Why it is probably second priority:

- even our best recurrent proxies are still far behind the strong `~1.16` public record lanes
- the dense int6/zstd stack seems like lower-risk leverage

## 12. Recommended experiment order

The best mainline is:

- dense, not recurrent
- modular, not kitchen-sink
- focused on proving evaluation and export gains separately before expensive retrains

The best order is:

1. use a clean dense checkpoint as an eval/export lab
2. run a seq2048 dense calibration branch
3. build the serious dense int6/zstd + MLP3 mainline
4. keep recurrence only as a side hedge

### 12.1 Checkpoint frontier sweeps first

Before building a fresh trainer branch, train one clean dense checkpoint and sweep evaluation/export ideas on that same checkpoint.

Recommended checkpoint:

- dense `9x512`
- `kv4`
- `mlp_mult=2`
- tuned schedule
- no recurrence
- no compression-aware training

For that one checkpoint, always measure:

- pre-quant BPB
- post-quant BPB
- post-slide BPB
- total bytes
- eval time

Sweep family A, evaluation:

- flat evaluation
- sliding-window evaluation with stride in `{1024, 512, 256, 128, 64}`

Sweep family B, exporter/precision:

- int8 + zlib
- int8 + zlib + fp16 tied embedding
- int6 + zstd on large matrix groups
- same low-bit format plus one promoted candidate group such as late `c_k`

Why this is the best first step:

- it separates evaluation wins from training wins
- it separates exporter wins from architecture wins
- it lets us rank selective-precision ideas by BPB gained per added MB
- it avoids retraining full models just to test stride or serializer choices

Current implementation status:

- This step is now operational.
- A clean dense calibration checkpoint has been trained and saved.
- The frontier sweep harness was initially wrong because it evaluated an uncompiled model, while the trainer's reported metrics come from compiled CUDA graphs.
- The harness has now been fixed to use the trainer-matching compiled eval path for both flat eval and sliding-window logits scoring.

Concrete validation after the fix:

- trainer stop-time flat eval on the calibration checkpoint: `1.3368`
- trainer final roundtrip exact eval: `1.33820201`
- reduced fixed sweep flat prequant eval: `1.33597771`
- reduced fixed sweep flat int8 roundtrip eval: `1.33819382`
- reduced fixed sweep sliding stride `512` prequant eval: `1.30782535`
- full fixed frontier sweep: [dense_calib_9x512_tuned_12m_a_frontier.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier.csv)

Interpretation:

- the sweep infrastructure is now decision-grade
- sliding-window eval is confirmed to be a high-ROI lever in this codebase too
- the full stride/export frontier is now available and gives a concrete first ranking

Key full-sweep takeaways:

- sliding-window eval is worth about `0.028` to `0.031` BPB on this dense calibration checkpoint, depending on exporter
- stride improvements saturate quickly: `512 -> 256` helps a bit, `256 -> 128` helps only slightly, and `128 -> 64` is very small
- best quality in the sweep was `int8 + zlib + fp16 tied embedding + stride 64` at `1.30525147`, but that artifact is over cap at `16,209,041` bytes
- best under-cap result in the sweep was plain `int8 + zlib + stride 64` at `1.30688143` and `15,889,674` bytes
- `int6 + zstd` on the current core tensor group buys large byte headroom, but on this checkpoint it costs a small amount of BPB relative to plain int8
- `int6 + zstd + fp16 tied embedding + stride 64` reached `1.30709551` at only `12,336,039` bytes, which means low-bit export still looks promising as a budget-funding mechanism for a larger retrain
- a targeted late-`K` test on the last three `c_k` weights did not help: [dense_calib_9x512_tuned_12m_a_frontier_latek3.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier_latek3.csv) came back slightly worse than the same low-bit branch without late-`K` promotion

### 12.2 Use seq2048 as a calibration branch

The seq2048 record is best treated as a control lane, not as the first mainline.

Suggested lane:

- dense `9x512 kv4 mlp2`
- tied embeddings
- `TRAIN_SEQ_LEN=2048`
- tuned LR family
- Muon momentum around `0.99`
- warmdown around `3000`

Purpose:

- verify our harness reproduces a known-good directional gain
- test long-context training in our codebase before combining it with new exporter complexity

Current completed seq2048 results:

- `12m` tuned dense seq2048 calibration:
  - [dense_calib_seq2048_tuned_12m_a.txt](/workspace/parameter-golf/logs/dense_calib_seq2048_tuned_12m_a.txt)
  - raw checkpoint: [dense_calib_seq2048_tuned_12m_a.pt](/workspace/parameter-golf/artifacts/dense_calib_seq2048_tuned_12m_a.pt)
  - quant artifact: [dense_calib_seq2048_tuned_12m_a.int8.ptz](/workspace/parameter-golf/artifacts/dense_calib_seq2048_tuned_12m_a.int8.ptz)
  - stopped at `step 3682`
  - built-in flat exact roundtrip metric: `1.32118384`
  - total int8+zlib size: `15,853,615` bytes

- Full-val evaluator shortlist on that `12m` seq2048 checkpoint:
  - [dense_eval_seq2048_fullval_shortlist.csv](/workspace/parameter-golf/logs/dense_eval_seq2048_fullval_shortlist.csv)
  - best exported result was:
    - `int8 + zlib`
    - `stream_sliding`
    - `eval_seq_len=2048`
    - `stride=256`
    - `val_bpb=1.30203861`
    - `artifact_bytes=15,853,594`
  - this is the best measured score of the current research stage

- `40m` seq2048 local confirmation on the same single-shard proxy failed:
  - [final_stage_seq2048_40m_b.txt](/workspace/parameter-golf/logs/final_stage_seq2048_40m_b.txt)
  - raw checkpoint: [final_stage_seq2048_40m_b.pt](/workspace/parameter-golf/artifacts/final_stage_seq2048_40m_b.pt)
  - quant artifact: [final_stage_seq2048_40m_b.int8.ptz](/workspace/parameter-golf/artifacts/final_stage_seq2048_40m_b.int8.ptz)
  - stopped at `step 12318`
  - built-in flat exact roundtrip metric degraded badly to `1.36896822`
  - total int8+zlib size stayed under cap at `15,896,755` bytes

Interpretation:

- seq2048 is a real positive lever
- but longer local training on the current `1`-shard proxy is not automatically better
- the best seq2048 checkpoint so far is the `12m` one, not the `40m` one
- this means the next seq2048 work should be schedule tuning or trajectory selection, not “just train longer”

### 12.3 Serious dense mainline after that

Once the evaluator and exporter are understood, the real mainline should be:

- dense `9x512 kv4`
- `TRAIN_SEQ_LEN=1024` at first
- tuned Muon schedule
- fp16 tied embedding
- int6 mixed export
- `zstd`
- then `MLP_MULT=3`
- then sliding-window eval
- then maybe selective late-layer `K` precision

The key framing is:

- int6 is mainly an enabler
- the real prize is spending the saved bytes on MLP capacity
- selective precision is a refinement, not the foundation

### 12.4 Only then try refinements

Worth trying later, in order:

- selective late-layer `K` promotion
- smaller slide stride if eval budget permits
- SmearGate
- seq2048 on top of the widened/export-improved dense lane

Not worth mixing in immediately:

- recurrence
- MTP
- basis/codebook/sidecar
- validation-set-training ideas

One important exception to preserve:

- the low-bit exporter on the strong dense `MLP2` checkpoint remains one of the best current leads even though the first `MLP3` branch failed
- specifically, `int6 + zstd + fp16 tied embedding + stride 64` reached `1.30709551` at only `12,336,039` bytes
- that is worse than the best current seq2048 score, but it preserves more than `3.5 MB` of extra artifact headroom
- so it remains one of the cleanest budget-funding candidates for a future larger retrain

### 12.5 First dense MLP3 calibration

We have now run the first direct dense `MLP_MULT=3` calibration on the tuned dense lane.

Run:

- [dense_calib_9x512_mlp3_tuned_12m_a.txt](/workspace/parameter-golf/logs/dense_calib_9x512_mlp3_tuned_12m_a.txt)
- raw checkpoint: [dense_calib_9x512_mlp3_tuned_12m_a.pt](/workspace/parameter-golf/artifacts/dense_calib_9x512_mlp3_tuned_12m_a.pt)
- quant artifact: [dense_calib_9x512_mlp3_tuned_12m_a.int8.ptz](/workspace/parameter-golf/artifacts/dense_calib_9x512_mlp3_tuned_12m_a.int8.ptz)

Result:

- params: `21,778,504`
- local cap: `720s`
- stopped at `step 4001`
- stop-time flat eval: `val_bpb 1.3434`
- final int8 zlib roundtrip exact: `1.34540988`
- total int8 zlib artifact size: `20,164,010` bytes

Interpretation:

- On plain int8 zlib export, the first `MLP3` run is worse than the tuned dense `MLP2` calibration.
- It is also far over the artifact cap on the default exporter.
- This does not kill the idea yet, because the intended thesis is “use lower-bit export to fund `MLP3`,” not “use `MLP3` with the old exporter.”
- The right next measurement is therefore a reduced frontier sweep on the saved `MLP3` checkpoint with the corrected evaluator.

Current live status:

- Reduced `MLP3` frontier sweep:
  - [dense_calib_9x512_mlp3_tuned_12m_a_frontier64.csv](/workspace/parameter-golf/logs/dense_calib_9x512_mlp3_tuned_12m_a_frontier64.csv)
- One first attempt failed only because the sweep was launched with the default `MLP_MULT=2` shape.
- That was corrected by rerunning with `--mlp-mult 3`.

Reduced frontier result:

- prequant sliding stride `64`: `1.31346864`
- int8 zlib sliding stride `64`: `1.31547150` at `20,163,992` bytes
- int6 zstd core sliding stride `64`: `1.31914374` at `15,321,608` bytes
- int6 zstd core + fp16 tied embedding sliding stride `64`: `1.31770234` at `15,461,089` bytes

Conclusion:

- `MLP3` was not rescued by the current low-bit export stack.
- Even its best under-cap low-bit variant is materially worse than the tuned dense `MLP2` frontier.
- That means the immediate mainline should return to the stronger dense `MLP2` branch rather than keep pushing this first `MLP3` attempt.

## 13. What not to spend time on right now

- current codebook branch
- current basis-sharing branch
- current sidecar branch
- recurrent `kv2` variants in the current setup
- compression-aware training on the current branch
- global fp16/fp32 export instead of selective precision

The problem with these ideas is not that they are theoretically impossible. The problem is that the evidence we have today says they are not the best use of the next few days of iteration.

## 14. Concrete next-step plan

### Immediate next work

1. Dense calibration checkpoint: completed.
   - run id: `dense_calib_9x512_tuned_12m_a`
   - trainer log: [dense_calib_9x512_tuned_12m_a.txt](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a.txt)
   - raw checkpoint: [dense_calib_9x512_tuned_12m_a.pt](/workspace/parameter-golf/artifacts/dense_calib_9x512_tuned_12m_a.pt)
   - quant artifact: [dense_calib_9x512_tuned_12m_a.int8.ptz](/workspace/parameter-golf/artifacts/dense_calib_9x512_tuned_12m_a.int8.ptz)
   - exact metric: `1.33820201`
   - size: `15,889,688` bytes

2. Checkpoint-only frontier sweeps: completed and validated.
   - script: [checkpoint_frontier_sweep.py](/workspace/parameter-golf/scripts/checkpoint_frontier_sweep.py)
   - first output CSV: [dense_calib_9x512_tuned_12m_a_frontier.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier.csv)
   - current status: fixed and now trainer-matching
   - root cause: the original sweep evaluated an uncompiled model instead of the trainer's compiled eval path
   - reduced validation sweep: [dense_calib_9x512_tuned_12m_a_frontier_fixcheck.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier_fixcheck.csv)
   - full fixed frontier sweep: [dense_calib_9x512_tuned_12m_a_frontier.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier.csv)
   - reduced fixed sweep results:
     - prequant flat: `1.33597771`
     - prequant sliding stride `512`: `1.30782535`
     - int8 zlib flat: `1.33819382`
     - int8 zlib sliding stride `512`: `1.31000055`
   - full fixed sweep highlights:
     - prequant sliding stride `64`: `1.30469691`
     - best under-cap exported result: int8 zlib sliding stride `64` = `1.30688143` at `15,889,674` bytes
     - best overall exported result: int8 zlib + fp16 tied embedding + sliding stride `64` = `1.30525147`, but over cap at `16,209,041` bytes
     - most budget-efficient low-bit variant so far: int6 zstd core + fp16 tied embedding + sliding stride `64` = `1.30709551` at `12,336,039` bytes

3. Immediate next work:
   - stop the current `MLP3` branch and return to the stronger dense `MLP2` lane

4. Immediate follow-up experiments from the frontier:
   - seq2048 calibration: completed and successful at `12m`
   - seq2048 `40m` confirmation: completed and negative on the current single-shard proxy
   - broaden the export sweep on the `MLP2` checkpoint beyond the current core int6 pattern if we want to keep pushing the byte-headroom lane
   - do not spend more near-term time on this first `MLP3` branch unless a substantially different exporter is ready

5. Latest seq2048 conclusion:
   - the `12m` seq2048 checkpoint is currently the best training-side result
   - the best measured evaluator on that checkpoint is `stream_sliding 2048/256`
   - the `40m` local seq2048 run should not be treated as the new base because it clearly degraded

6. Only after schedule retuning or trajectory selection, build the next dense mainline with:
   - int6 mixed export
   - `zstd`
   - fp16 tied embedding
   - `MLP_MULT=3`
   - sliding-window eval

7. Add a `2-3m` LR/momentum or warmdown prefilter runner.

8. Confirm only the top candidates with `10-12m` proxy runs first.
   Do not assume `40m` local runs are better until the longer-horizon schedule is retuned.

### If we want to preserve the recurrent line in parallel

Run full `40m` confirmations for:

- `7u2r_576_kv4_m2`
- `6u2r_608_kv4_m2`

Then, if either is still alive, test one seq2048 variant before adding any new exporter complexity.

and do not spend more time on recurrent `kv2`.

## 15. Focus Shift: Evaluation Mainline

At this point, evaluation should be the mainline.

Why:

- The challenge explicitly allows aggressive evaluation methods within a separate `10 minute` evaluation budget and allows evaluation at any sequence length.
- Our local dense checkpoint already proved that evaluation is not just cleanup. On [dense_calib_9x512_tuned_12m_a_frontier.csv](/workspace/parameter-golf/logs/dense_calib_9x512_tuned_12m_a_frontier.csv), plain flat int8 scoring was `1.33819382`, while the same exported model with sliding-window stride `64` reached `1.30688143`, a gain of about `0.0313` BPB with zero retraining.
- The first selective-precision probe was negative and the first `MLP3` branch was not rescued by the current exporter, so evaluation is now the highest-ROI lever that transfers across checkpoints we already have.
- Public evidence points the same way:
  - PR `#77` shows `doc-isolated` scoring alone improved `1.2278 -> 1.2168`, then sliding-window improved that further to `1.1941`, while LoRA TTT itself only added a small extra gain to `1.1910`.
  - PR `#102` reports sliding-window stride `64` as about a `0.03` BPB gain and still finishes eval in about `64s` on `8xH100`.
  - PR `#114` gets to `1.1574` with `240s` of sliding-window eval, which means the public frontier is still not saturating the full eval budget.

So the current order of operations should be:

1. Make the evaluator itself the experiment surface on the strong dense `MLP2` checkpoint.
2. Exhaust the obvious evaluation frontier before starting another retrain.
3. Only after the evaluator is strong, decide whether the next retrain should be `seq2048`, improved low-bit export, or both.

Concrete next evaluator tasks:

1. Add `doc-isolated` evaluation to [checkpoint_frontier_sweep.py](/workspace/parameter-golf/scripts/checkpoint_frontier_sweep.py).
   Use BOS or validation document boundaries so context never crosses documents.

2. Sweep `EVAL_SEQ_LEN x EVAL_STRIDE` on the dense `MLP2` checkpoint.
   First matrix: `seq_len in {1024, 1536, 2048}` crossed with `stride in {256, 128, 64}`.

3. Keep export variants limited while doing this.
   Use only the strongest current under-cap exports:
   - `int8 + zlib`
   - `int6 + zstd core + fp16 tied embedding`

4. If doc-isolated sliding works well, build one adaptive second-pass evaluator.
   Start with:
   - pass 1: cheap doc-isolated sliding, e.g. `1024/256`
   - pass 2: rescore only the hardest documents or chunks at `2048/64`

5. Only then test one no-gradient cache-style sidecar.
   For example, a per-document cache or n-gram interpolation during scoring. This should come before any full LoRA-TTT implementation.

Stop/go rules:

- Keep any evaluation change that buys at least about `0.005` BPB while staying comfortably within the `10 minute` evaluation budget.
- Prefer evaluator upgrades that transfer across checkpoints over checkpoint-specific training hacks.
- Do not launch another fresh architecture branch until the best dense evaluator is known.

Status update after doing those steps:

- [checkpoint_frontier_sweep.py](/workspace/parameter-golf/scripts/checkpoint_frontier_sweep.py) now supports:
  - BOS-split document isolation
  - arbitrary `EVAL_SEQ_LEN`
  - bounded research sweeps via `--max-docs` and `--max-val-tokens`
  - full exported-model scoring on the same evaluator settings

- First bounded evaluator sanity on `50` docs:
  - [\_tmp_eval_sanity.csv](/workspace/parameter-golf/logs/_tmp_eval_sanity.csv)
  - `stream_flat 1024`: `1.38289849`
  - `doc_flat 1024`: `1.36369156`
  - `doc_sliding 1024/256`: `1.34448464`
  - This confirmed that the doc-aware path was live and directionally sensible.

- `500`-doc dense `MLP2` prequant frontier:
  - [dense_eval_mlp2_prequant_docs500.csv](/workspace/parameter-golf/logs/dense_eval_mlp2_prequant_docs500.csv)
  - Best row was `stream_sliding 1024/256` at `1.31281773`
  - `doc_sliding 1024/256` was essentially tied but slightly worse at `1.31343754`
  - `2048` flat/doc-flat improved over `1024` flat/doc-flat, but `2048` sliding was not competitive on this `1024`-trained checkpoint

- `2000`-doc dense `MLP2` prequant confirmation:
  - [dense_eval_mlp2_prequant_docs2000.csv](/workspace/parameter-golf/logs/dense_eval_mlp2_prequant_docs2000.csv)
  - `stream_sliding 1024/256`: `1.31464953`
  - `doc_sliding 1024/256`: `1.31496118`
  - `doc_flat 2048`: `1.32639351`
  - `doc_sliding 2048/256`: `1.32377867`
  - This narrowed the realistic full-val shortlist to:
    - `stream_sliding 1024/256`
    - `doc_sliding 1024/256`
    - `doc_flat 2048`
    - `doc_sliding 2048/256`

- `2000`-doc quantized shortlist:
  - [dense_eval_mlp2_quant_docs2000_shortlist.csv](/workspace/parameter-golf/logs/dense_eval_mlp2_quant_docs2000_shortlist.csv)
  - On the exported `int8 + zlib` artifact, the ranking shifted toward the simpler evaluator:
    - `stream_sliding 1024/256`: `1.31732518`
    - `doc_sliding 1024/256`: `1.31751521`
    - `doc_sliding 2048/256`: `1.32660634`
    - `doc_flat 2048`: `1.32894754`

- Full-val confirmation on the current strong dense checkpoint:
  - [dense_eval_mlp2_fullval_shortlist.csv](/workspace/parameter-golf/logs/dense_eval_mlp2_fullval_shortlist.csv)
  - Prequant:
    - `stream_sliding 1024/256`: `1.30545675`
    - `doc_sliding 1024/256`: `1.30557165`
    - `doc_sliding 2048/256`: `1.31307652`
    - `doc_flat 2048`: `1.31518570`
  - Exported `int8 + zlib`:
    - `stream_sliding 1024/256`: `1.30763732`
    - `doc_sliding 1024/256`: `1.30773999`
    - `doc_sliding 2048/256`: `1.31523387`
    - `doc_flat 2048`: `1.31737605`

Current evaluator conclusion:

- On the current best dense `1024`-trained checkpoint, the best full-val exported evaluator is `stream_sliding` with `eval_seq_len=1024` and `stride=256`.
- Doc isolation is not a clear win on this checkpoint; it is nearly tied at `1024/256`, but not better.
- Longer-context evaluation at `2048` did not pay off on this checkpoint once the model was exported.
- This means the public evaluator lessons did transfer only partially here: sliding-window absolutely matters, but the best local evaluator is simpler than the strongest doc-isolated public lane.

Latest seq2048 update after evaluator completion:

- On the seq2048-trained `12m` checkpoint, the evaluator ranking changed materially.
- [dense_eval_seq2048_fullval_shortlist.csv](/workspace/parameter-golf/logs/dense_eval_seq2048_fullval_shortlist.csv) shows that the best exported evaluator there is `stream_sliding 2048/256` at `1.30203861`.
- So the best evaluator is checkpoint-dependent:
  - dense `1024`-trained checkpoint: `stream_sliding 1024/256`
  - dense `2048`-trained checkpoint: `stream_sliding 2048/256`

Latest training-side failure mode we found:

- The trainer originally reread the single local train shard from disk on every wrap, which caused periodic large stalls in seq2048 local runs.
- This is now fixed by caching train shards in RAM through `CACHE_TRAIN_SHARDS` in [train_gpt.py](/workspace/parameter-golf/train_gpt.py).
- That fix improved throughput stability, but it did not change the underlying conclusion: the `40m` seq2048 run still validated much worse than the `12m` seq2048 run.
- The likely reasons are:
  - the effective LR schedule changed because warmdown is coupled to wallclock in [train_gpt.py](/workspace/parameter-golf/train_gpt.py#L1005)
  - the local proxy still trains on only one deterministic shard, so longer runs can overfit the local stream

Current best candidates, ranked by what they teach us:

1. Best current measured score:
   - `12m` dense seq2048 checkpoint
   - evaluator `stream_sliding 2048/256`
   - `1.30203861`
   - `15,853,594` bytes

2. Best current budget-headroom candidate:
   - dense `MLP2` checkpoint with `int6 + zstd + fp16 tied embedding + stride 64`
   - `1.30709551`
   - `12,336,039` bytes
   - this is the strongest current path if the goal is to fund a larger retrain with minimal artifact pressure

3. Best simple under-cap dense baseline-style evaluator on the `1024` lane:
   - dense `MLP2` checkpoint with `int8 + zlib + stream_sliding 1024/256`
   - `1.30763732`
   - `15,889,674` bytes

## 16. Short version

The current branch learned three things that still matter:

- sliding-window evaluation is real, but it is not enough to rescue an uncompetitive training lane
- the strongest byte-funding signal is still our low-bit dense `MLP2` result at `12.3MB`
- the first failed `MLP3` attempt does not invalidate the public dense SOTA recipe, because it was not yet a true integrated `int6 + zstd + selective-precision + long-context` lane

The mainline is now:

- dense `9x512 KV4`
- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=2048`
- integrated `int6 + zstd` export in the trainer
- `fp16 tok_emb`
- selective late-`K` preservation and grouped int8 for the remaining `K`
- PR-style optimizer settings

What is no longer the focus:

- recurrence as the primary route
- basis/codebook/sidecar work
- doc-isolated eval as the next big bet
- blind long local runs on the one-shard proxy

The immediate next work after this handoff update is:

1. run the full-val exporter-transfer measurement on the saved `12m` seq2048 winner under the improved stack
2. run two integrated `12m` seq2048 retrains with the same improved stack:
   - `MLP_MULT=2`
   - `MLP_MULT=3`
3. if one of those is directionally good, repeat at `20m`
4. do not jump straight back to `40m` on the one-shard proxy until the schedule is retuned to the longer horizon
5. only after the base model is stronger, return to scaling test-time compute toward the full `600s` budget
