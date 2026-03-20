#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import io
import math
import os
import sys
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
import zstandard

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import train_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint-only eval/export frontier sweep.")
    parser.add_argument("--checkpoint", required=True, help="Path to a raw model checkpoint (.pt)")
    parser.add_argument("--summary-out", default="", help="Optional CSV output path")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--num-unique-layers", type=int, default=0)
    parser.add_argument("--num-recurrence", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=524288)
    parser.add_argument("--eval-seq-lens", default="", help="Comma-separated eval seq lens; default=train_seq_len")
    parser.add_argument("--strides", default="1024,512,256,128,64")
    parser.add_argument(
        "--modes",
        default="stream_flat,stream_sliding,doc_flat,doc_sliding",
        help="Comma-separated modes: stream_flat,stream_sliding,doc_flat,doc_sliding",
    )
    parser.add_argument(
        "--variant-names",
        default="",
        help="Optional comma-separated subset of variant names to run (e.g. prequant,int8_zlib)",
    )
    parser.add_argument("--late-k-patterns", default="", help="Comma-separated additional fp16 keep-float patterns")
    parser.add_argument("--max-docs", type=int, default=0, help="If >0, only evaluate the first N documents")
    parser.add_argument("--max-val-tokens", type=int, default=0, help="If >0, only evaluate the first N tokens")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> train_gpt.GPT:
    model = train_gpt.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        num_recurrence=args.num_recurrence,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        compression_aware_kl_weight=0.0,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, train_gpt.CastedLinear):
            module.float()
    train_gpt.restore_low_dim_params_to_fp32(model)
    return model


def configure_cuda_runtime() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)


def compile_eval_functions(base_model: train_gpt.GPT) -> tuple[torch.nn.Module, callable]:
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    return compiled_model, compiled_logits


def warm_eval_functions(
    compiled_model: torch.nn.Module,
    compiled_logits: callable,
    val_tokens: torch.Tensor,
    device: torch.device,
    seq_len: int,
    val_batch_size: int,
) -> None:
    warm_batch_seqs = max(1, val_batch_size // seq_len)
    desired = warm_batch_seqs * seq_len + 1
    warm_tokens = val_tokens[:desired]
    usable = ((warm_tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Need at least {seq_len + 1} validation tokens for warmup; got {int(warm_tokens.numel())}")
    warm_tokens = warm_tokens[: usable + 1].to(device=device, dtype=torch.int64, non_blocking=True)
    x = warm_tokens[:-1].reshape(-1, seq_len)
    y = warm_tokens[1:].reshape(-1, seq_len)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _ = compiled_model(x, y)
            _ = compiled_logits(x)
    torch.cuda.synchronize()


def parse_patterns(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_int_list(text: str, fallback: int) -> list[int]:
    if not text.strip():
        return [fallback]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


def quantize_int6_per_row(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8).contiguous()
        return q, scale.contiguous()
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / float(scale.item())), -32, 31).to(torch.int8).contiguous()
    return q, scale


def keep_float_tensor(
    name: str,
    t: torch.Tensor,
    keep_fp16_patterns: tuple[str, ...],
    passthrough_orig_dtypes: dict[str, str],
) -> torch.Tensor:
    if matches_any(name, train_gpt.CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=torch.float16).contiguous()
    if matches_any(name, keep_fp16_patterns):
        return t.to(dtype=torch.float16).contiguous()
    return t


def quantize_state_dict_variant(
    state_dict: dict[str, torch.Tensor],
    keep_fp16_patterns: tuple[str, ...],
    int6_patterns: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, torch.Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += train_gpt.tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["payload_bytes"] += train_gpt.tensor_nbytes(t)
            continue

        if (
            t.numel() <= train_gpt.INT8_KEEP_FLOAT_MAX_NUMEL
            or matches_any(name, keep_fp16_patterns)
            or matches_any(name, train_gpt.CONTROL_TENSOR_NAME_PATTERNS)
        ):
            kept = keep_float_tensor(name, t, keep_fp16_patterns, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += train_gpt.tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if t.ndim == 2 and matches_any(name, int6_patterns):
            q, s = quantize_int6_per_row(t)
            qmeta[name] = {"scheme": "per_row", "bits": 6}
        else:
            q, s = train_gpt.quantize_float_tensor(t)
            qmeta[name] = {"scheme": "per_row" if s.ndim > 0 else "per_tensor", "bits": 8}

        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += train_gpt.tensor_nbytes(q) + train_gpt.tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "mixed_lowbit_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_variant(obj: dict[str, object]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def compress_obj(obj: dict[str, object], compressor: str) -> tuple[bytes, int]:
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    if compressor == "zlib":
        blob = zlib.compress(raw, level=9)
    elif compressor == "zstd":
        blob = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        raise ValueError(f"Unsupported compressor: {compressor}")
    return blob, len(raw)


def load_validation_tokens_full(pattern: str) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return torch.cat([train_gpt.load_data_shard(file) for file in files]).contiguous()


def split_doc_ranges(val_tokens: torch.Tensor, bos_id: int) -> list[tuple[int, int]]:
    bos_positions = (val_tokens == bos_id).nonzero(as_tuple=False).flatten().tolist()
    if not bos_positions:
        return [(0, int(val_tokens.numel()))]
    if bos_positions[0] != 0:
        bos_positions.insert(0, 0)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(bos_positions):
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else int(val_tokens.numel())
        if end - start >= 2:
            ranges.append((start, end))
    return ranges


def build_windows(
    ranges: list[tuple[int, int]],
    seq_len: int,
    stride: int | None,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for start, end in ranges:
        doc_len = end - start
        if doc_len < 2:
            continue
        if stride is None:
            local_starts = range(0, doc_len - 1, seq_len)
            next_target = 1
        else:
            local_starts = range(0, doc_len - 1, stride)
            next_target = 1
        for local_start in local_starts:
            actual_len = min(seq_len, doc_len - local_start - 1)
            if actual_len <= 0:
                break
            if stride is None:
                score_from = 0
                score_len = actual_len
            else:
                window_target_start = local_start + 1
                window_target_end = local_start + actual_len
                score_target_start = max(next_target, window_target_start)
                if score_target_start > window_target_end:
                    continue
                score_from = score_target_start - window_target_start
                score_len = window_target_end - score_target_start + 1
                next_target = window_target_end + 1
            windows.append((start + local_start, actual_len, score_from, score_len))
    return windows


def eval_windows(
    model: train_gpt.GPT,
    logits_fn: callable,
    seq_len: int,
    val_batch_size: int,
    val_tokens: torch.Tensor,
    windows: list[tuple[int, int, int, int]],
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    vocab_size: int,
    pad_id: int,
) -> tuple[float, float]:
    eval_batch_windows = max(1, val_batch_size // seq_len)
    val_nll_sum = 0.0
    val_token_count = 0
    val_byte_count = 0.0
    model.eval()

    with torch.inference_mode():
        for batch_off in range(0, len(windows), eval_batch_windows):
            batch = windows[batch_off:batch_off + eval_batch_windows]
            bs = len(batch)
            fixed_bs = eval_batch_windows
            x_cpu = torch.full((fixed_bs, seq_len), fill_value=pad_id, dtype=torch.int64)
            y_cpu = torch.full((fixed_bs, seq_len), fill_value=pad_id, dtype=torch.int64)
            score_mask_cpu = torch.zeros((fixed_bs, seq_len), dtype=torch.bool)

            for row_idx, (token_start, actual_len, score_from, score_len) in enumerate(batch):
                chunk = val_tokens[token_start: token_start + actual_len + 1].to(dtype=torch.int64)
                x_cpu[row_idx, :actual_len] = chunk[:-1]
                y_cpu[row_idx, :actual_len] = chunk[1:]
                score_mask_cpu[row_idx, score_from: score_from + score_len] = True

            x_batch = x_cpu.to(device=device, non_blocking=True)
            y_batch = y_cpu.to(device=device, non_blocking=True)
            score_mask = score_mask_cpu.to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = logits_fn(x_batch).view(fixed_bs, seq_len, vocab_size)

            per_token_nll = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1),
                reduction="none",
            ).view(fixed_bs, seq_len)
            val_nll_sum += float(per_token_nll.masked_select(score_mask).sum().item())
            val_token_count += int(score_mask.sum().item())

            prev_ids = x_batch.masked_select(score_mask)
            tgt_ids = y_batch.masked_select(score_mask)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += float(token_bytes.float().sum().item())

    val_loss = val_nll_sum / val_token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = val_token_count / val_byte_count
    model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def run_eval_mode(
    mode_name: str,
    seq_len: int,
    stride: int | None,
    model: train_gpt.GPT,
    logits_fn: callable,
    val_batch_size: int,
    val_tokens: torch.Tensor,
    stream_ranges: list[tuple[int, int]],
    doc_ranges: list[tuple[int, int]],
    device: torch.device,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    vocab_size: int,
    pad_id: int,
) -> tuple[float, float]:
    if mode_name == "stream_flat":
        windows = build_windows(stream_ranges, seq_len=seq_len, stride=None)
    elif mode_name == "stream_sliding":
        if stride is None:
            raise ValueError("stream_sliding requires stride")
        windows = build_windows(stream_ranges, seq_len=seq_len, stride=stride)
    elif mode_name == "doc_flat":
        windows = build_windows(doc_ranges, seq_len=seq_len, stride=None)
    elif mode_name == "doc_sliding":
        if stride is None:
            raise ValueError("doc_sliding requires stride")
        windows = build_windows(doc_ranges, seq_len=seq_len, stride=stride)
    else:
        raise ValueError(f"Unknown mode: {mode_name}")

    return eval_windows(
        model=model,
        logits_fn=logits_fn,
        seq_len=seq_len,
        val_batch_size=val_batch_size,
        val_tokens=val_tokens,
        windows=windows,
        device=device,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_cuda_runtime()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    summary_path = Path(args.summary_out) if args.summary_out else Path("logs") / f"{checkpoint_path.stem}_frontier.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab size {int(sp.vocab_size())}")

    val_tokens = load_validation_tokens_full(os.path.join(args.data_path, "fineweb_val_*.bin"))
    if args.max_val_tokens > 0:
        val_tokens = val_tokens[: args.max_val_tokens].contiguous()
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_gpt.build_sentencepiece_luts(sp, args.vocab_size, device)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    bos_id = int(sp.bos_id())
    pad_id = int(sp.pad_id())
    doc_ranges = split_doc_ranges(val_tokens, bos_id=bos_id)
    if args.max_docs > 0:
        doc_ranges = doc_ranges[: args.max_docs]
        if doc_ranges:
            val_tokens = val_tokens[: doc_ranges[-1][1]].contiguous()
        doc_ranges = split_doc_ranges(val_tokens, bos_id=bos_id)
    stream_ranges = [(0, int(val_tokens.numel()))]
    print(f"Loaded validation tokens={int(val_tokens.numel())} docs={len(doc_ranges)}")

    base_model = build_model(args, device)
    base_model.load_state_dict(state_dict, strict=True)
    model, compiled_logits = compile_eval_functions(base_model)
    eval_seq_lens = parse_int_list(args.eval_seq_lens, args.train_seq_len)
    modes = parse_patterns(args.modes)
    variant_name_filter = set(parse_patterns(args.variant_names))
    rows: list[dict[str, object]] = []
    strides = [int(piece) for piece in args.strides.split(",") if piece.strip()]
    code_bytes = Path("train_gpt.py").stat().st_size

    late_k_patterns = parse_patterns(args.late_k_patterns)
    variants = [
        {"name": "int8_zlib", "compressor": "zlib", "keep_fp16": tuple(), "int6": tuple()},
        {"name": "int8_zlib_fp16_embed", "compressor": "zlib", "keep_fp16": ("tok_emb.weight",), "int6": tuple()},
        {
            "name": "int6_zstd_core",
            "compressor": "zstd",
            "keep_fp16": tuple(),
            "int6": (".mlp.", ".attn.c_q.", ".attn.c_v.", ".attn.proj."),
        },
        {
            "name": "int6_zstd_core_fp16_embed",
            "compressor": "zstd",
            "keep_fp16": ("tok_emb.weight",),
            "int6": (".mlp.", ".attn.c_q.", ".attn.c_v.", ".attn.proj."),
        },
    ]
    if late_k_patterns:
        variants.append(
            {
                "name": "int6_zstd_core_fp16_embed_latek",
                "compressor": "zstd",
                "keep_fp16": ("tok_emb.weight",) + late_k_patterns,
                "int6": (".mlp.", ".attn.c_q.", ".attn.c_v.", ".attn.proj."),
            }
        )

    all_variants = [{"name": "prequant", "compressor": "none", "keep_fp16": tuple(), "int6": tuple()}] + variants
    if variant_name_filter:
        all_variants = [variant for variant in all_variants if variant["name"] in variant_name_filter]

    for variant in all_variants:
        if variant["name"] == "prequant":
            model_bytes = "NA"
            artifact_bytes = "NA"
        else:
            qobj, stats = quantize_state_dict_variant(state_dict, variant["keep_fp16"], variant["int6"])
            blob, raw_bytes = compress_obj(qobj, variant["compressor"])
            model_bytes = len(blob)
            artifact_bytes = model_bytes + code_bytes
            dequantized_state = (
                dequantize_state_dict_variant(torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu"))
                if variant["compressor"] == "zlib"
                else dequantize_state_dict_variant(torch.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(blob)), map_location="cpu"))
            )
            base_model.load_state_dict(dequantized_state, strict=True)

        for eval_seq_len in eval_seq_lens:
            warm_eval_functions(model, compiled_logits, val_tokens, device, eval_seq_len, args.val_batch_size)

            for mode_name in modes:
                if mode_name.endswith("_flat"):
                    t_eval = time.perf_counter()
                    eval_loss, eval_bpb = run_eval_mode(
                        mode_name=mode_name,
                        seq_len=eval_seq_len,
                        stride=None,
                        model=model,
                        logits_fn=compiled_logits,
                        val_batch_size=args.val_batch_size,
                        val_tokens=val_tokens,
                        stream_ranges=stream_ranges,
                        doc_ranges=doc_ranges,
                        device=device,
                        base_bytes_lut=base_bytes_lut,
                        has_leading_space_lut=has_leading_space_lut,
                        is_boundary_token_lut=is_boundary_token_lut,
                        vocab_size=args.vocab_size,
                        pad_id=pad_id,
                    )
                    rows.append(
                        {
                            "variant": variant["name"],
                            "mode": mode_name,
                            "eval_seq_len": eval_seq_len,
                            "stride": eval_seq_len,
                            "val_loss": f"{eval_loss:.8f}",
                            "val_bpb": f"{eval_bpb:.8f}",
                            "model_bytes": model_bytes,
                            "artifact_bytes": artifact_bytes,
                            "compressor": variant["compressor"],
                            "eval_ms": f"{1000.0 * (time.perf_counter() - t_eval):.0f}",
                        }
                    )
                    print(
                        f"{variant['name']:>28} {mode_name:>14} eval_seq={eval_seq_len:<4} stride={eval_seq_len:<4} "
                        f"val_bpb={rows[-1]['val_bpb']} artifact_bytes={artifact_bytes}"
                    )
                    continue

                for stride in strides:
                    if stride >= eval_seq_len:
                        continue
                    t_eval = time.perf_counter()
                    eval_loss, eval_bpb = run_eval_mode(
                        mode_name=mode_name,
                        seq_len=eval_seq_len,
                        stride=stride,
                        model=model,
                        logits_fn=compiled_logits,
                        val_batch_size=args.val_batch_size,
                        val_tokens=val_tokens,
                        stream_ranges=stream_ranges,
                        doc_ranges=doc_ranges,
                        device=device,
                        base_bytes_lut=base_bytes_lut,
                        has_leading_space_lut=has_leading_space_lut,
                        is_boundary_token_lut=is_boundary_token_lut,
                        vocab_size=args.vocab_size,
                        pad_id=pad_id,
                    )
                    rows.append(
                        {
                            "variant": variant["name"],
                            "mode": mode_name,
                            "eval_seq_len": eval_seq_len,
                            "stride": stride,
                            "val_loss": f"{eval_loss:.8f}",
                            "val_bpb": f"{eval_bpb:.8f}",
                            "model_bytes": model_bytes,
                            "artifact_bytes": artifact_bytes,
                            "compressor": variant["compressor"],
                            "eval_ms": f"{1000.0 * (time.perf_counter() - t_eval):.0f}",
                        }
                    )
                    print(
                        f"{variant['name']:>28} {mode_name:>14} eval_seq={eval_seq_len:<4} stride={stride:<4} "
                        f"val_bpb={rows[-1]['val_bpb']} artifact_bytes={artifact_bytes}"
                    )

        base_model.load_state_dict(state_dict, strict=True)

    fieldnames = ["variant", "mode", "eval_seq_len", "stride", "val_loss", "val_bpb", "model_bytes", "artifact_bytes", "compressor", "eval_ms"]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']:>28}  {row['mode']:>14}  eval_seq={row['eval_seq_len']:<4}  stride={row['stride']:<4}  "
            f"val_bpb={row['val_bpb']}  model_bytes={row['model_bytes']}  artifact_bytes={row['artifact_bytes']}"
        )
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
