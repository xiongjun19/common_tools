# coding=utf8


def calc_gpt_params(layer_num, hid_dim, vocab_size=50000, ffn_mid=None):
    att_param = 4 * (hid_dim ** 2) * layer_num
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    ffn_param = 2 * hid_dim * ffn_mid * layer_num
    lm_head_param = hid_dim * vocab_size
    res = att_param + ffn_param +  2 * lm_head_param
    return res


def calc_gpt_moe_params(layer_num, hid_dim, moe_layer, experts, vocab_size=50000, ffn_mid=None):
    att_param = 4 * (hid_dim ** 2) * layer_num
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    ffn_param = 2 * hid_dim * ffn_mid * (layer_num - moe_layer)
    moe_param = 2 * hid_dim * ffn_mid * experts * moe_layer
    lm_head_param = hid_dim * vocab_size
    res = att_param + ffn_param + lm_head_param + moe_param
    return res


def calc_transformer_param(enc_num, dec_num, hid_dim, vocab_size=50000, ffn_mid=None):
    enc_params = calc_gpt_params(enc_num, hid_dim, vocab_size, ffn_mid)
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    att_param = 4 * (hid_dim ** 2) * dec_num * 2
    ffn_param = 2 * hid_dim * ffn_mid * dec_num
    dec_param = att_param + ffn_param
    res = enc_params + dec_param
    return res


def calc_transformer_param_moe(enc_num, dec_num, hid_dim,  moe_enc_num, moe_dec_num, experts,  vocab_size=50000, ffn_mid=None):
    # import ipdb; ipdb.set_trace()
    enc_params = calc_gpt_moe_params(enc_num, hid_dim, moe_enc_num, experts,  vocab_size, ffn_mid)
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    att_param = 4 * (hid_dim ** 2) * dec_num * 2
    ffn_param = 2 * hid_dim * ffn_mid * (dec_num - moe_dec_num)
    moe_param = 2 * hid_dim * ffn_mid * experts * moe_dec_num
    dec_param = att_param + ffn_param + moe_param
    route_param = hid_dim * hid_dim* (moe_enc_num + moe_dec_num)
    res = enc_params + dec_param + route_param
    return res


def _calc_t5_parm_imp(dec_num, hid_dim, ffn_mid, is_dec):
    att_num = 1
    if is_dec:
        att_num = 2
    att_param = att_num * 4 * (hid_dim ** 2) * dec_num
    ffn_param = 3 * hid_dim * ffn_mid * dec_num
    res = att_param + ffn_param
    return res

def calc_t5_param(enc_num, dec_num, hid_dim, vocab_size=50000, ffn_mid=None):
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    enc_param = _calc_t5_parm_imp(enc_num, hid_dim, ffn_mid, False)
    dec_param = _calc_t5_parm_imp(dec_num, hid_dim, ffn_mid, True)
    lm_head_param = hid_dim * vocab_size
    res = enc_param + dec_param + lm_head_param
    return res


def _calc_t5_moe_imp(dec_num, moe_num, experts, hid_dim, ffn_mid, is_dec):
    att_num = 1
    if is_dec:
        att_num = 2
    att_param = att_num * 4 * (hid_dim ** 2) * dec_num
    ffn_param = 3 * hid_dim * ffn_mid * (dec_num - moe_num)
    moe_param = 3 * hid_dim * ffn_mid * experts * moe_num
    res = att_param + ffn_param + moe_param
    return res


def calc_t5_param_moe(enc_num, dec_num, hid_dim,  moe_enc_num, moe_dec_num, experts,  vocab_size=50000, ffn_mid=None):
    if ffn_mid is None:
        ffn_mid = 4 * hid_dim
    enc_param = _calc_t5_moe_imp(enc_num, moe_enc_num, experts, hid_dim, ffn_mid, False)
    dec_param = _calc_t5_moe_imp(dec_num, moe_dec_num, experts, hid_dim, ffn_mid, True)
    lm_head_param = hid_dim * vocab_size
    res = enc_param + dec_param + lm_head_param
    return res


if __name__ == '__main__':
    hid = 4096
    layers = 32
    params = calc_gpt_params(layers, hid, 32000, 14336)
    print(params / 1e9)
    print("now calc MOE experts ")
    hid  = 4096
    layers = 32 
    moe_layers = 32 
    num_exp = 8
    params = calc_gpt_moe_params(layers, hid, moe_layers, num_exp, 32000, 14336)
    print(params / 1e9)

    hid = 8192
    layers = 48
    params = calc_gpt_params(layers, hid, 32000)
    print(params / 1e9)
    print("now calc MOE experts ")
    hid = 8192
    layers = 48
    moe_layers = 48
    num_exp = 8
    params = calc_gpt_moe_params(layers, hid, moe_layers, num_exp, 32000)
    print(params / 1e9)


    hid = 10240
    layers = 120
    print('single model calc param ')
    params = calc_gpt_params(layers, hid, 32000)
    print(params / 1e9)
    print("now calc MOE experts ")
    hid = 10240
    layers = 120
    moe_layers = 120
    num_exp = 16 
    params = calc_gpt_moe_params(layers, hid, moe_layers, num_exp, 32000)
    print(params / 1e9)



