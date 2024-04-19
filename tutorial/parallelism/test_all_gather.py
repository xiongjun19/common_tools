# coding=utf8

import torch
import torch.distributed as dist


def test(device, rank, local_rank):
    x1 = torch.ones([3], dtype=torch.int32) + rank
    x2 = x1.to(device)
    print("before allgather @rank: {rank}, input_val: {x2}")
    world_size = dist.get_world_size()
    tensor_list = [torch.empty_like(x2) for _ in range(world_size)]
    tensor_list[rank] = x2
    torch.distributed.all_gather(tensor_list, x2)
    print(f"@rank: {rank}, tensor_list: {tensor_list}")


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    device_num = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % device_num
    device = torch.device('cuda', local_rank)
    test(device, rank, local_rank)
