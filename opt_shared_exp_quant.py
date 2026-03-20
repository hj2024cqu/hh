

import os, sys, time, json, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_encode import (
    encode_weight_matrix_fast, decode_exp_meta, decode_packed_value,
    BASE_OFFSET, BASE_BITS, BASE_MASK, PAYLOAD_BITS,
)

SPARSE_DECODE_D0 = [0,0,0,1,1,2, 1,2,3,2,3,3]
SPARSE_DECODE_D1 = [1,2,3,2,3,3, 0,0,0,1,1,2]


def get_opt(model_path):
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model_path = os.path.abspath(model_path)
    if not os.path.exists(os.path.join(model_path, 'config.json')):
        parent = os.path.dirname(model_path)
        avail = os.listdir(parent) if os.path.exists(parent) else ['parent not found']
        raise FileNotFoundError(f"No config.json in {model_path}\nAvailable: {avail}")
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers,
                               name=name + '.' + name1 if name != '' else name1))
    return res


@torch.no_grad()
def verify_encoding_fast(W_original, compressed):
    rows, cols = compressed['shape']
    num_groups = compressed['num_groups']
    block_size = compressed['block_size']
    num_blocks = compressed['num_blocks']

    D0 = compressed['D0_packed']
    D1 = compressed['D1_packed']
    sp_packed = compressed['sparse_idx_packed']
    meta = compressed['exp_meta']

    W_decoded = torch.zeros(rows, cols, dtype=torch.float32)

    for g in range(num_groups):
        byte_idx = g // 2
        sp_bytes = sp_packed[:, byte_idx]
        codes = ((sp_bytes >> 4) & 0xF) if g % 2 == 0 else (sp_bytes & 0xF)

        for b in range(num_blocks):
            rs = b * block_size
            re = min(rs + block_size, rows)
            meta_val = meta[b, g].item()
            # ★ 使用 fast_encode 里统一的 decode_exp_meta
            d0_exps, d1_exps = decode_exp_meta(meta_val)

            d0_bytes = D0[rs:re, g]
            d1_bytes = D1[rs:re, g]
            block_codes = codes[rs:re]

            for local_r in range(re - rs):
                r = rs + local_r
                code = block_codes[local_r].item()
                if code > 11:
                    continue
                col0 = g * 4 + SPARSE_DECODE_D0[code]
                col1 = g * 4 + SPARSE_DECODE_D1[code]
                # ★ 使用 fast_encode 里统一的 decode_packed_value
                W_decoded[r, col0] = decode_packed_value(d0_bytes[local_r].item(), d0_exps)
                W_decoded[r, col1] = decode_packed_value(d1_bytes[local_r].item(), d1_exps)

    W_orig = W_original.float().cpu()
    nz_orig = W_orig != 0
    nz_dec = W_decoded != 0

    pos_match = (nz_orig & nz_dec).sum().item()
    pos_total = nz_orig.sum().item()
    extra_nz = (nz_dec & ~nz_orig).sum().item()

    result = {
        'position_match_rate': pos_match / max(pos_total, 1),
        'nonzero_orig': pos_total,
        'nonzero_decoded': nz_dec.sum().item(),
        'extra_nonzero': extra_nz,
        'max_rel_error': 0.0, 'mean_rel_error': 0.0,
        'max_abs_error': 0.0, 'mean_abs_error': 0.0,
    }
    mask = nz_orig & nz_dec
    if mask.any():
        errors = torch.abs(W_orig[mask] - W_decoded[mask])
        rel_errors = errors / torch.abs(W_orig[mask]).clamp(min=1e-38)
        result['max_abs_error'] = errors.max().item()
        result['mean_abs_error'] = errors.mean().item()
        result['max_rel_error'] = rel_errors.max().item()
        result['mean_rel_error'] = rel_errors.mean().item()
    return result


def compute_memory_stats(compressed):
    rows, cols = compressed['shape']
    num_groups = compressed['num_groups']
    num_blocks = compressed['num_blocks']
    d0 = compressed['D0_packed'].numel()
    d1 = compressed['D1_packed'].numel()
    sp = compressed['sparse_idx_packed'].numel()
    mt = num_blocks * num_groups * 4
    total = d0 + d1 + sp + mt
    orig = rows * cols * 2
    return {
        'D0_packed_bytes': d0, 'D1_packed_bytes': d1,
        'sparse_idx_bytes': sp, 'exp_meta_bytes': mt,
        'total_compressed_bytes': total,
        'original_fp16_bytes': orig,
        'compression_vs_fp16': orig / total,
        'bits_per_nonzero': total * 8 / (rows * num_groups * 2),
    }


@torch.no_grad()
def encode_model(model, args, dev):
    print('\n' + '=' * 70)
    print('Encoding model (FAST vectorized v3)')
    print('=' * 70)
    print(f'  Block size: {args.block_size}')
    print(f'  Mantissa bits: {args.mantissa_bits}')

    layers = model.model.decoder.layers
    all_compressed = {}
    total_orig = total_comp = 0
    total_mode_d0 = [0,0,0,0]
    total_mode_d1 = [0,0,0,0]
    total_swaps = total_pairs = total_1_4 = 0
    verify_count = 0

    for layer_idx in range(len(layers)):
        print(f'\n{"="*60}')
        print(f'Layer {layer_idx}')
        print(f'{"="*60}')

        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)

        for name, module in subset.items():
            W = module.weight.data.float().to(dev)
            rows, cols = W.shape
            if cols % 4 != 0:
                print(f'  {name}: skipped (cols={cols})')
                continue

            W_grouped = W.reshape(rows, -1, 4)
            nz = (W_grouped != 0).sum(dim=2)
            sr = (nz <= 2).float().mean().item()
            if sr < 0.90:
                print(f'  {name}: skipped (sparse={sr:.2%})')
                continue

            print(f'\n  {name} [{rows}x{cols}]')
            tick = time.time()

            compressed, enc_stats = encode_weight_matrix_fast(
                W, block_size=args.block_size, mantissa_bits=args.mantissa_bits
            )
            elapsed = time.time() - tick

            mem = compute_memory_stats(compressed)
            total_orig += mem['original_fp16_bytes']
            total_comp += mem['total_compressed_bytes']
            for i in range(4):
                total_mode_d0[i] += enc_stats['mode_counts'][i]
                total_mode_d1[i] += enc_stats['mode_counts_d1'][i]
            total_swaps += enc_stats['swap_count']
            total_pairs += enc_stats['total_pairs']
            total_1_4 += enc_stats.get('total_1_4', 0)

            mn = ['cont', '2+2', '3+1', 'gen']
            tb0, tb1 = sum(enc_stats['mode_counts']), sum(enc_stats['mode_counts_d1'])
            d0s = ', '.join(f"{mn[i]}:{enc_stats['mode_counts'][i]/max(tb0,1):.1%}" for i in range(4))
            d1s = ', '.join(f"{mn[i]}:{enc_stats['mode_counts_d1'][i]/max(tb1,1):.1%}" for i in range(4))

            print(f'    Time: {elapsed:.1f}s')
            print(f'    Compression: {mem["compression_vs_fp16"]:.2f}x vs FP16, '
                  f'{mem["bits_per_nonzero"]:.1f} bits/value')
            print(f'    Pairs: 2:4={enc_stats["total_pairs"]}, 1:4={enc_stats.get("total_1_4",0)}')
            print(f'    Swap: {enc_stats["swap_count"]}/{enc_stats["total_pairs"]} '
                  f'({enc_stats["swap_count"]/max(enc_stats["total_pairs"],1):.2%})')
            print(f'    D0 modes: {d0s}')
            print(f'    D1 modes: {d1s}')
            print(f'    Memory: total={mem["total_compressed_bytes"]//1024}KB')

            if verify_count < 3 and not args.skip_verify:
                print(f'    Verifying...')
                vt = time.time()
                vr = verify_encoding_fast(W.cpu(), compressed)
                ve = time.time() - vt
                print(f'    Verify ({ve:.1f}s): '
                      f'pos_match={vr["position_match_rate"]:.4%}, '
                      f'extra_nz={vr["extra_nonzero"]}, '
                      f'max_rel_err={vr["max_rel_error"]:.4f}, '
                      f'mean_rel_err={vr["mean_rel_error"]:.6f}')
                verify_count += 1

            key = f'layer{layer_idx}.{name}'
            all_compressed[key] = compressed
            if module.bias is not None:
                all_compressed[key + '.bias'] = module.bias.data.cpu()

        layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache()

    print('\n' + '=' * 70)
    print('Global Statistics')
    print('=' * 70)
    td0, td1 = sum(total_mode_d0), sum(total_mode_d1)
    mn = ['continuous', '2+2 split', '3+1 outlier', 'general']
    print(f'Total compression: {total_orig/1024/1024:.1f}MB -> '
          f'{total_comp/1024/1024:.1f}MB ({total_orig/max(total_comp,1):.2f}x)')
    print(f'Total pairs: 2:4={total_pairs}, 1:4={total_1_4}')
    print(f'Total swap: {total_swaps}/{total_pairs} ({total_swaps/max(total_pairs,1):.2%})')
    for slot, counts, total in [('D0', total_mode_d0, td0), ('D1', total_mode_d1, td1)]:
        print(f'\n{slot} mode distribution:')
        for i in range(4):
            print(f'  {mn[i]}: {counts[i]}/{total} ({counts[i]/max(total,1):.2%})')
    return all_compressed


def save_compressed(all_compressed, save_path, args):
    os.makedirs(save_path, exist_ok=True)
    meta_info = {
        'format': 'shared_exp_2_4_sparse_v3',
        'block_size': args.block_size,
        'mantissa_bits': args.mantissa_bits,
        'base_offset': BASE_OFFSET,
        'base_bits': 6,
        'payload_bits': PAYLOAD_BITS,
        'layers': {}
    }
    for key, data in all_compressed.items():
        layer_path = os.path.join(save_path, f'{key}.pt')
        torch.save(data, layer_path)
        if isinstance(data, dict):
            meta_info['layers'][key] = {
                'shape': list(data['shape']), 'file': f'{key}.pt', 'type': 'compressed_weight',
            }
        else:
            meta_info['layers'][key] = {
                'shape': list(data.shape), 'file': f'{key}.pt', 'type': 'tensor',
            }
    with open(os.path.join(save_path, 'compressed_meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f'\nSaved to {save_path} ({len(all_compressed)} files)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--mantissa_bits', type=int, default=4)
    parser.add_argument('--skip_verify', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    DEV = torch.device(args.device)
    print('=' * 70)
    print('Shared Exponent 2:4 Sparse Encoder v3')
    print('=' * 70)
    print(f'Model: {args.model}')
    print(f'Block size: {args.block_size}, Mantissa bits: {args.mantissa_bits}')
    print(f'Base encoding: {BASE_BITS} bits, offset {BASE_OFFSET}, range [{-BASE_OFFSET}, {(1<<BASE_BITS)-1-BASE_OFFSET}]')

    print('\nLoading model...')
    model = get_opt(args.model)
    model.eval()

    tick = time.time()
    all_compressed = encode_model(model, args, DEV)
    print(f'\nTotal encoding time: {time.time()-tick:.1f}s')

    save_compressed(all_compressed, args.save, args)

    total_size = sum(
        v.numel() * v.element_size()
        for d in all_compressed.values() if isinstance(d, dict)
        for v in d.values() if isinstance(v, torch.Tensor)
    )
    print(f'Total compressed size: {total_size/1024/1024:.1f} MB')
    print('Done.')