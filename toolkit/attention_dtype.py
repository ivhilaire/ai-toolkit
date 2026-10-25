# Attention in fp16 or fp32 for bf16 training on GPUs older than sm80 (Turing: RTX 20xx).
#
# PyTorch has no flash or memory-efficient attention kernel for bf16 below sm80,
# so bf16 scaled_dot_product_attention falls back to the math path, which builds
# the full score matrix (8 GB peak per call for Flux at 1024 px: OOM on 22 GB).
# The memory-efficient kernel does run in fp16 and fp32 there, so only attention
# changes dtype; everything else stays bf16. Measured on a 2080 Ti (Flux size,
# 4608 tokens, fwd+bwd): bf16 math 148 ms / 8 GB, fp16 38 ms / 0.5 GB,
# fp32 189 ms / 0.3 GB; fp16 and fp32 errors vs fp32 math equal bf16 math's.
#
# float32: no range risk in the forward pass.
# float16: more mantissa bits than bf16, range is the only risk (Flux and
# Qwen-Image RMS-normalise q and k):
#  - forward: q/k/v that don't fit fp16 use the original bf16 call
#  - backward: the incoming gradient is divided by its max before the fp16 cast
#    (no underflow of small gradients) and rescaled after
# Both: non-finite gradients are recomputed with the original bf16 call (the fp32
# efficient kernel's backward gives NaN for inputs around 1e5, fine at 1e4).
# Every fallback is counted and printed, so nothing degrades silently.
import torch
import torch.nn.functional as F

_orig_sdpa = None
_dtype = None
calls = 0
fallbacks = 0
_FP16_MAX = torch.finfo(torch.float16).max


def _report_fallback(where):
    global fallbacks
    fallbacks += 1
    if fallbacks <= 5 or fallbacks % 100 == 0:
        print(f"attention_dtype: {where} out of range, used bf16 math attention "
              f"(fallbacks {fallbacks} of {calls} calls)")


def _to_fp16(x):
    x16 = x.to(torch.float16)
    return x16 if torch.isfinite(x16).all() else None


def _mask_to(mask, dtype):
    if mask is None or mask.dtype == torch.bool:
        return mask
    if dtype == torch.float16:
        # additive mask: very negative bf16 values must not become -inf (a fully
        # masked row would give NaN instead of 0)
        mask = mask.clamp(min=-_FP16_MAX)
    return mask.to(dtype)


def _sdpa_with_grads(q, k, v, grad_out, mask, kwargs):
    with torch.enable_grad():
        q, k, v = (x.detach().requires_grad_() for x in (q, k, v))
        out = _orig_sdpa(q, k, v, attn_mask=mask, **kwargs)
        return torch.autograd.grad(out, (q, k, v), grad_out)


class _CastAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, qc, kc, vc, mask, kwargs):
        # only the cast copies are kept: bf16 -> fp16/fp32 -> bf16 is exact for finite values
        ctx.save_for_backward(qc, kc, vc)
        ctx.mask = mask
        ctx.kwargs = kwargs
        out = _orig_sdpa(qc, kc, vc, attn_mask=_mask_to(mask, qc.dtype), **kwargs)
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        qc, kc, vc = ctx.saved_tensors
        dtype = qc.dtype
        scale = grad_out.abs().amax().float() if dtype == torch.float16 else torch.ones((), device=grad_out.device)
        if scale > 0 and torch.isfinite(scale):
            gc = (grad_out.float() / scale).to(dtype)
            grads = _sdpa_with_grads(qc, kc, vc, gc, _mask_to(ctx.mask, dtype), ctx.kwargs)
            grads = [(g.float() * scale).to(grad_out.dtype) for g in grads]
            if all(torch.isfinite(g).all() for g in grads):
                return (*grads, None, None, None, None, None)
        _report_fallback("backward")
        q, k, v = (x.to(grad_out.dtype) for x in (qc, kc, vc))
        grads = _sdpa_with_grads(q, k, v, grad_out, ctx.mask, ctx.kwargs)
        return (*grads, None, None, None, None, None)


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    kwargs = dict(dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if query.dtype != torch.bfloat16 or not query.is_cuda or dropout_p != 0.0:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)
    global calls
    calls += 1
    if _dtype == torch.float16:
        qc, kc, vc = _to_fp16(query), _to_fp16(key), _to_fp16(value)
        if qc is None or kc is None or vc is None:
            _report_fallback("forward")
            return _orig_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)
    else:
        qc, kc, vc = query.float(), key.float(), value.float()
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return _CastAttention.apply(query, key, value, qc, kc, vc, attn_mask, kwargs)
    out = _orig_sdpa(qc, kc, vc, attn_mask=_mask_to(attn_mask, _dtype), **kwargs)
    return out.to(query.dtype)


def install(dtype):
    """Patch F.scaled_dot_product_attention; dtype "float16" or "float32". No-op on sm80+."""
    global _orig_sdpa, _dtype
    dtypes = {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
    if dtype not in dtypes:
        raise ValueError(f"attention_dtype must be float16 or float32, got {dtype!r}")
    if _orig_sdpa is not None or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        print(f"attention_dtype: sm{major}{minor} has bf16 attention kernels, not needed")
        return False
    _dtype = dtypes[dtype]
    _orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa
    print(f"attention_dtype: on (sm{major}{minor}): bf16 attention runs in {dtype}")
    return True
