# 2026-03-19 Research Summary

## Current branch

- Workspace branch: `40593b3` plus local experimental trainers and runners.
- Current main trainer: `train_gpt.py`
- Local proxy/full-run environment: `2x H100`, using `40m` local runs as a rough stand-in for `10m` on `8x H100`.

## Full 40-minute results on current branch

Source: [overnight_mar19_fix1_summary.csv](/workspace/parameter-golf/logs/overnight_mar19_fix1_summary.csv)

Best full-run result on the current branch:

- `dense_bulk_squeeze_9x640_kv2_m1`
- `val_bpb = 1.34181490`
- `bytes_total = 15,968,537`

Best recurrent control on the current branch:

- `recur_control_6u2r_512_kv4_m2`
- `val_bpb = 1.36929268`
- `bytes_total = 10,783,610`

Main takeaways from the overnight batch:

- Plain recurrence is alive, but the baseline `6u2r_512_kv4_m2` is massively under budget.
- Aggressive recurrent bulk squeeze (`kv2 + mlp_mult=1`) is bad in this regime.
- Basis/codebook/sidecar moonshots were not competitive in their current form.
- Compression-aware training did not show enough value to justify itself for hillclimbing.

## 12-minute proxy screen

Source: [proxy_mar19_a_summary.csv](/workspace/parameter-golf/logs/proxy_mar19_a_summary.csv)

Proxy settings:

- `MAX_WALLCLOCK_SECONDS=720`
- `WARMUP_STEPS=4`
- `VAL_LOSS_EVERY=0`
- `COMPRESSION_AWARE_MODE=off`
- final exact roundtrip eval kept on

Results:

| run | val_bpb | bytes_total |
|---|---:|---:|
| `dense_anchor_9x640_kv2_m1` | `1.36148663` | `15,950,523` |
| `recur_anchor_6u2r_512_kv4_m2` | `1.35839009` | `10,725,244` |
| `recur_wide_6u2r_576_kv4_m2` | `1.35118640` | `13,140,902` |
| `recur_wide_6u2r_608_kv4_m2` | `1.34907255` | `14,264,252` |
| `recur_wide_6u2r_640_kv2_m2` | `1.40163487` | `14,893,918` |
| `recur_unique_7u2r_576_kv4_m2` | `1.34791587` | `14,873,796` |

Main takeaways from the proxy batch:

- The recurrent branch improves materially when it spends more of the byte budget.
- Best short-run recurrent config was `7u2r_576_kv4_m2`.
- Close second was `6u2r_608_kv4_m2`.
- `kv2` is not a good width-funding lever for recurrence in this setup.
- The 12-minute proxy is useful for pruning, but close results should still be confirmed with full runs.

## What 30-second LR sweeps are good for

Thirty-second runs are useful only as a coarse prefilter.

Good use:

- eliminate obviously unstable learning rates
- catch clearly underpowered learning rates that barely move loss
- sanity check warmup or momentum changes

Bad use:

- choosing final winners by BPB
- comparing exporter-sensitive changes
- judging close calls between viable configs

Reason:

- at `30s`, warmup and startup overhead dominate
- larger models get fewer steps, so ranking gets biased by throughput
- final exact eval is too noisy relative to the amount of training

Recommended screening ladder:

1. `2-3m` LR prefilter on train loss only
2. `10-12m` proxy runs with exact final roundtrip eval
3. `40m` confirmation runs for the top few configs

## PR review

Fetched and inspected:

- PR 88: `upstream/pr-88`
- PR 99: `upstream/pr-99`
- PR 102: `upstream/pr-102`

### PR 88

Record folder:

- `records/track_10min_16mb/2026-03-19_Int6_MLP3x_MTP_SlideEval`

Headline:

- `val_bpb = 1.1605`
- `15.28 MB`

Core ideas:

- int6 quantization + `zstd`
- `MLP_HIDDEN=1536` / `3x` MLP
- `TRAIN_SEQ_LEN=4096`
- MTP auxiliary head
- fp16 tied embedding export
- sliding-window eval
- tuned optimizer dynamics

Assessment:

- strongest raw package of the three
- also the most complex starting point
- long-context training and MTP add a lot of moving parts

### PR 99

Record folder:

- `records/track_10min_16mb/2026-03-19_Seed2025_Top2K_Stride64`

Headline:

- `val_bpb = 1.16050360`
- `15.84 MB`

Core ideas:

- int6 mixed quantization + `zstd`
- `MLP_MULT=3`
- fp16 tied embedding
- fp16 late-layer `c_k` passthrough on blocks 7 and 8
- grouped int8 on remaining `c_k`
- sliding-window eval with `stride=64`

Assessment:

- best practical starting point
- very strong result
- simpler than PR 88
- already structured around selective precision, which lines up with our own intuition

### PR 102

Record folder:

- `records/track_10min_16mb/2026-03-19_Int6_MLP3x_TunedLR_SmearGate_SlidingWindow`

Headline:

- `val_bpb = 1.1618`
- `15.14 MB`

Core ideas:

- int6 mixed quantization + `zstd`
- `MLP_MULT=3`
- tuned LR / momentum / warmdown
- fp16 tied embedding
- SmearGate
- sliding-window eval

Assessment:

- excellent second starting point
- cleaner than PR 88
- likely easier to adapt than the MTP lane
- SmearGate looks cheap enough to try on top of PR 99

## Recommendation

If the goal is to move to a better base immediately:

1. Start from PR 99.
2. Treat PR 102 as the first donor for ideas, especially optimizer settings and SmearGate.
3. Treat PR 88 as a richer but more complex lane, worth mining only after the simpler base is running.

If staying on the current branch for one more round:

1. Full-run `7u2r_576_kv4_m2`
2. Full-run `6u2r_608_kv4_m2`
3. Stop spending time on recurrent `kv2`

## Next concrete steps

- Write a small LR prefilter runner for `2-3m` runs, not `30s` runs.
- Bring PR 99 into a local branch and reproduce its exact record.
- Port one idea at a time from PR 102 onto the PR 99 base:
  - optimizer schedule
  - SmearGate
- Only after that, revisit selective precision experiments on top of the stronger int6/zstd base.
