# coding=utf8

import pandas as pd
from dataclasses import dataclass


@dataclass
class ModelConfig:
    layers: int
    heads: int
    hid_dim: int
    ffn_dim: int
    voc: int
    moe_layers: int = 0
    experts: int = 0
    selected_experts: int = 0


@dataclass
class SysConfig:
    peak_FLOPS: int = 312  # the peak FLOPS for fp16
    peak_Mem: int = 80   # the maximun device memory
    pure_comp_util: float = 0.65
    intra_bandwidth: float = 200.
    inter_bandwidth: float = 17.5
    local_scale: int = 8
    sys_name: str = 'HB8'


@dataclass
class TrainInfo:
    glob_bs: int = 1024
    micro_bs: int = 1
    precision: str = 'fp16'
    seq_len: int = 4096
    recomp: str = 'recomp'  # ['no-recomp', 'recomp', 'selective']
    tp: int = 1
    pp: int = 1
    dp: int = 1
    cp: int = 1
    pp_interleaved: bool = False
    pp_chunk_layers: int = 2


@dataclass
class JobConfig:
    model_cfg: ModelConfig
    sys_cfg: SysConfig
    train_cfg: TrainInfo


def get_job_from_excel(f_path):
    items = _read_excel(f_path)
    jobs = [_parse_job(item) for item in items]
    return jobs, items


def _read_excel(input_path):
    df = pd.read_excel(input_path)
    res = df.to_dict('records')
    return res


def _parse_job(item):
    mod_cfg = ModelConfig(
        item.get('tot_layers'),
        item.get('heads'),
        item.get('hidden_dim'),
        item.get('ffn_dim'),
        item.get('voc', 50000),
        item.get('moe_layers', 0),
        item.get('MOE', 0),
        item.get('selected_experts', 2)
    )
    if mod_cfg.ffn_dim is None:
        if mod_cfg.hid_dim is not None:
            mod_cfg.ffn_dim = 4 * mod_cfg.hid_dim

    sys_cfg = SysConfig(
        item.get('gpu_flops'),
        item.get('gpu_mem', 80),
        item.get('gpu_util', 0.55),
        item.get('act_intra_bandwidth'),
        item.get('act_inter_bandwidth'),
        item.get('local_scale', 8),
        item.get('topo', 'HB8')
    )

    train_cfg = TrainInfo(
        item.get('global_batch_size'),
        item.get('micro_batch_size'),
        item.get('precision', 'fp16'),
        item.get('seq_len'),
        item.get('recomp', 'recomp'),
        item.get('TP'),
        item.get('PP'),
        item.get('DP'),
        item.get('CP', 1),
        item.get('pp_interleaved', False),
        item.get('pp_chunk_layers', 2),
    )
    job_cfg = JobConfig(mod_cfg, sys_cfg, train_cfg)
    return job_cfg


if __name__ == '__main__':
    import sys
    test_job_cfg = get_job_from_excel(sys.argv[1])
    print(test_job_cfg)
