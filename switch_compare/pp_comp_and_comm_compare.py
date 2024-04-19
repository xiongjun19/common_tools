# coding=utf8

def main(bs, seq, spec_FLOPS, recomp,  layers=1, hid_dim=12288, heads=96, pp_bandwidth=20, tp=8):
    ffn_dim = 4 * hid_dim
    flops = calc_flops(bs, seq, hid_dim, ffn_dim, layers, heads, recomp)
    comp_time = flops / (spec_FLOPS * 1e12) / tp
    comm_time =  _calc_pp_msg(bs, seq, hid_dim, pp_bandwidth)
    print(f"comp_time is {comp_time}, comm_time is: {comm_time}")


def calc_flops(bs, seq, hid_dim, ffn_dim, layers, heads, recomp):
    one_layer_flops = _calc_one_layer_flops(bs, seq, hid_dim, ffn_dim, recomp)
    tot_flops = layers * one_layer_flops['tot']
    return tot_flops


def _calc_pp_msg(bs, seq, hid_dim, pp_bandwidth):
    comm_size = 2 * bs * seq * hid_dim
    print(comm_size / (1024 **2))
    comm_time = comm_size / (1024 ** 3) / pp_bandwidth
    return comm_time

def _calc_one_layer_flops(bs, seq,  hid_dim, ffn_dim, recomp_str):
    h = hid_dim
    fwd_att_flops = 4 * bs * seq * (h ** 2) * 2 + 2 * bs * (seq ** 2) * h * 2
    bwd_att_flops = 2 * fwd_att_flops
    fwd_mlp_flops = 2 * bs * seq * h * ffn_dim * 2
    bwd_mlp_flops = 2 * fwd_mlp_flops
    if recomp_str == 'recomp':
        bwd_att_flops += fwd_att_flops
        bwd_mlp_flops += fwd_mlp_flops
    elif recomp_str == 'selective':
        bwd_att_flops += 2 * bs * (seq ** 2) * h * 2
    tot = fwd_att_flops + bwd_att_flops + fwd_mlp_flops + bwd_mlp_flops
    return {
        'tot': tot, 'fwd_att': fwd_att_flops,
        'bwd_att': bwd_att_flops, 'fwd_mlp': fwd_mlp_flops,
        'bwd_mlp': bwd_mlp_flops,
    }


if __name__ == '__main__':
    main(1, 4096, 192, None, 1, 12288, 96, 20)
