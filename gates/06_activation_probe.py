#!/usr/bin/env python3
"""Gate 6 (model side) — per-layer activation range probe.

gfx906 has no BF16. The design runs FP16 activations: FP16 overflows at 65,504 and an FP16 RMSNorm's
x^2 overflows at |x| >= 256. This records, per decoder layer, max|residual| entering the layer
(input_layernorm's input), max|residual| after attention (post_attention_layernorm's input — usually the
largest), max|attention output|, max|MLP/MoE output| and the final norm's input, over a prompt + greedy
generation, and flags every site that would overflow. Those sites must be FP32 in the design
(residual, norm input, accumulation).

Needs the whole model in GPU memory: GLM-5.3 is ~1.5 TB in bf16 (8 x MI300X/MI325X), or use the FP8
checkpoint on 8 x H200 with --dtype auto (weights stay FP8; activations are what is measured).
The script refuses to run if accelerate offloaded any module to CPU/disk.

  python3 gates/06_activation_probe.py --model zai-org/GLM-5.3 --prompt-file prompt.txt [--dtype auto|bf16|fp16|fp32]

Works for any HF causal LM whose decoder layers live at model.model.layers (GLM-5.x, DeepSeek-V3.x,
Llama, Qwen) or model.model.language_model.layers (VL wrappers). Use long, real agentic prompts:
massive activations are token-dependent.
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--prompt-file', required=True)
ap.add_argument('--dtype', default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
ap.add_argument('--max-new', type=int, default=64)
ap.add_argument('--out', default='results/gate6_activations.json')
a = ap.parse_args()

dt = {'auto': 'auto', 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[a.dtype]
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map='auto', trust_remote_code=True).eval()
except TypeError:  # transformers < 4.56
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt, device_map='auto', trust_remote_code=True).eval()

offloaded = {n: d for n, d in getattr(model, 'hf_device_map', {}).items() if str(d) in ('cpu', 'disk')}
if offloaded:
    raise SystemExit(f'{len(offloaded)} modules offloaded to CPU/disk (not enough GPU memory), e.g. {list(offloaded)[:3]}; '
                     f'use a node with enough GPU memory (bf16: ~1.6 TB) or the FP8 checkpoint with --dtype auto')

def find_layers(m):
    for path in ('model.layers', 'model.language_model.layers', 'language_model.layers', 'layers'):
        obj = m
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        owner = m
        for part in path.split('.')[:-1]:
            owner = getattr(owner, part)
        return owner, obj
    raise SystemExit(f'cannot find decoder layers on {type(m).__name__}')

owner, layers = find_layers(model)
stats = {}

def rec(name, t):
    if not torch.is_tensor(t) or t.numel() == 0:
        return
    t = t.detach()
    m = max(float(t.max()), -float(t.min()))  # no fp32/abs temporaries: a 100k-token prefill would OOM otherwise
    stats[name] = max(stats.get(name, 0.0), m)

def first_hidden(args, kwargs):
    return args[0] if args else kwargs.get('hidden_states')

def pre_hook(name):
    def f(mod, args, kwargs):
        rec(name, first_hidden(args, kwargs))
    return f

def post_hook(name):
    def f(mod, args, out):
        rec(name, out[0] if isinstance(out, tuple) else out)
    return f

handles = []
for i, l in enumerate(layers):
    handles.append(l.register_forward_pre_hook(pre_hook(f'L{i:02d}.residual_in'), with_kwargs=True))
    pan = getattr(l, 'post_attention_layernorm', None)
    if pan is not None:
        handles.append(pan.register_forward_pre_hook(pre_hook(f'L{i:02d}.residual_mid'), with_kwargs=True))
    for what in ('self_attn', 'mlp'):
        sub = getattr(l, what, None)
        if sub is not None:
            handles.append(sub.register_forward_hook(post_hook(f'L{i:02d}.{what}')))
final_norm = getattr(owner, 'norm', None)
if final_norm is not None:
    handles.append(final_norm.register_forward_pre_hook(pre_hook('final.norm_in'), with_kwargs=True))

text = open(a.prompt_file).read()
ids = tok(text, return_tensors='pt').to(model.device)
with torch.no_grad():
    model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)

FP16_MAX, SQ_MAX = 65504.0, 256.0
bad = 0
for k in sorted(stats):
    v = stats[k]
    is_norm_in = k.endswith('residual_in') or k.endswith('residual_mid') or k.endswith('norm_in')
    flag = 'OVERFLOW_FP16' if v > FP16_MAX else ('RMSNORM_X2_OVERFLOW_FP16' if v > SQ_MAX and is_norm_in else '')
    bad += bool(flag)
    print(f'{k:24s} max|x|={v:12.2f} {flag}')
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump({'model': a.model, 'dtype': a.dtype, 'prompt_tokens': int(ids['input_ids'].shape[1]), 'stats': stats},
          open(a.out, 'w'), indent=1)
print(f'\n{bad} site(s) exceed FP16-safe ranges -> those sites need FP32 in the design' if bad
      else '\nno FP16-unsafe activation on this prompt (try longer / harder prompts)')
