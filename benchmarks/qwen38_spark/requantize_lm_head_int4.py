#!/usr/bin/env python3
"""Repack an int8 GPTQ lm_head shard into 4-bit GPTQ (group 128, full-range
symmetric), the layout vLLM's GPTQ-Marlin serves with `"+:.*lm_head$": {"bits": 4}`.

The head is memory-bound at decode: it is read once per draft position plus once
per verification, so halving its bytes halves that share of the step. Input is the
int8 head produced by tools/quantize_lm_head_int8.py; it is dequantized (group-128
symmetric, error ~0.2%) and requantized to 4 bits.

Layout produced (matches auto_round:auto_gptq 4-bit, v1 zero storage):
  lm_head.qweight  int32 [in/8, out]      8 nibbles per int32, low nibble first
  lm_head.qzeros   int32 [groups, out/8]  all nibbles 7 (zp=8, v1 stores zp-1)
  lm_head.scales   f16   [groups, out]

Usage:
  requantize_lm_head_int4.py <in_shard.safetensors> <out_shard.safetensors>
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import safe_open, save_file

GROUP = 128
OUT_CHUNK = 8192


def unpack_int8(qweight: torch.Tensor) -> torch.Tensor:
    """int32 [in/4, out] -> uint8 [in, out], 4 little-endian bytes per int32."""
    packed = qweight.contiguous().view(torch.uint8)  # [in/4, out*4]
    rows, _ = qweight.shape
    out_f = qweight.shape[1]
    return packed.reshape(rows, out_f, 4).permute(0, 2, 1).reshape(rows * 4, out_f)


def requantize(qweight: torch.Tensor, scales: torch.Tensor):
    in_f, out_f = qweight.shape[0] * 4, qweight.shape[1]
    assert in_f % GROUP == 0 and out_f % 8 == 0
    groups = in_f // GROUP
    q8 = unpack_int8(qweight)

    new_scales = torch.empty((groups, out_f), dtype=torch.float16)
    error = weight_abs = 0.0
    new_qweight = torch.empty((in_f // 8, out_f), dtype=torch.int32)

    for start in range(0, out_f, OUT_CHUNK):
        stop = min(start + OUT_CHUNK, out_f)
        # Dequantize this slice of the vocabulary back to float32.
        chunk = q8[:, start:stop].to(torch.float32).sub_(128.0)
        chunk = chunk.reshape(groups, GROUP, stop - start)
        chunk.mul_(scales[:, start:stop].to(torch.float32).unsqueeze(1))

        # Full-range symmetric: the largest-magnitude element of each
        # (group, out) maps to the -8 slot, so the scale carries its opposite
        # sign. Ties resolve to the positive element, as in the int8 tool.
        wmin = chunk.amin(dim=1)
        wmax = chunk.amax(dim=1)
        picked = torch.where(-wmin > wmax, wmin, wmax)
        scale = (picked / -8).to(torch.float16)
        scale[scale == 0] = torch.finfo(torch.float16).tiny
        new_scales[:, start:stop] = scale

        scale_f32 = scale.to(torch.float32).unsqueeze(1)
        q4 = torch.clamp(torch.round(chunk / scale_f32) + 8, 0, 15)
        error += float((q4.sub(8).mul_(scale_f32) - chunk).abs().sum())
        weight_abs += float(chunk.abs().sum())

        q4 = q4.reshape(in_f, stop - start).reshape(in_f // 8, 8, stop - start)
        nibbles = q4.numpy().astype("uint32")
        packed = nibbles[:, 0, :].copy()
        for nibble in range(1, 8):
            packed |= nibbles[:, nibble, :] << (4 * nibble)
        new_qweight[:, start:stop] = torch.from_numpy(packed.view("int32"))
        del chunk, q4, nibbles, packed

    print(f"requantization mean absolute error: {error / weight_abs:.4%} of |w|")
    qzeros = torch.full((groups, out_f // 8), 0x77777777, dtype=torch.int32)
    return new_qweight, qzeros, new_scales


def main() -> int:
    in_path, out_path = sys.argv[1], sys.argv[2]
    tensors = {}
    with safe_open(in_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for key in keys:
            tensors[key] = f.get_tensor(key)
    missing = {"lm_head.qweight", "lm_head.scales"} - set(tensors)
    if missing:
        sys.exit(f"missing {missing} in {in_path}; expected an int8 GPTQ head")

    qweight, scales = tensors["lm_head.qweight"], tensors["lm_head.scales"]
    print(f"int8 head qweight {tuple(qweight.shape)} scales {tuple(scales.shape)}")
    new_qweight, new_qzeros, new_scales = requantize(qweight, scales)
    print(
        f"int4 head qweight {tuple(new_qweight.shape)} scales {tuple(new_scales.shape)}"
    )

    tensors["lm_head.qweight"] = new_qweight
    tensors["lm_head.qzeros"] = new_qzeros
    tensors["lm_head.scales"] = new_scales
    save_file(tensors, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
