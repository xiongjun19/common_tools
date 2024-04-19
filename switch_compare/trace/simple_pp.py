# coding=utf8


import json
import time
import threading
from threading import Thread
from dataclasses import dataclass


@dataclass
class FwdWorker:
    micro_id: int
    stage: int
    chunk_id: int
    layers: int
    exec_time: float
    pp_msg_time: float


@dataclass
class BwdWorker:
    micro_id: int
    stage: int
    chunk_id: int
    lyaers: int
    exec_time: float
    pp_msg_time: float


lock = threading.Lock()
fwd_states = []
bwd_states = []


def _get_chunk_id(micro_id, chunks, m, pp, forward=True):
    if chunks == 1:
        return 0
    base = min(m, pp)
    if m <= pp:
        ch_id = micro_id // base
        if forward:
            return ch_id
        else:
            return chunks - ch_id - 1
    else:
        times = (m + pp) // pp
        reg_tho = (times - 1) * pp * chunks
        res_base = m % pp
        if micro_id < reg_tho:
            res = micro_id % (pp * chunks)
            ch_id = res // base
            if forward:
                return ch_id
            else:
                return chunks - ch_id - 1
        ch_id = (micro_id - reg_tho) // res_base
        if not forward:
            return chunks - ch_id - 1
        return ch_id


def _mid2detail_id(micro_id, chunks, m, pp, forward=True):
    if m < pp:
        base = m
        group_size = chunks * base
        g_id = micro_id // group_size
        ch_id = micro_id % group_size // base
        b_id = micro_id % base
        if not forward:
            ch_id = chunks - ch_id - 1
        return g_id, ch_id, b_id
    else:
        base = pp
        group_size = chunks * base
        max_num_groups = (m + pp) // pp
        reg_tho = (max_num_groups - 1) * group_size
        res_base = m % pp
        g_id = micro_id // group_size
        if micro_id >= reg_tho:
            micro_id = (micro_id - reg_tho)
            base = res_base
        ch_id = micro_id % group_size // base
        b_id = micro_id % group_size % base
        if not forward:
            ch_id = chunks - ch_id - 1
        return g_id, ch_id, b_id


def _detail_id2mid(g_id, ch_id, b_id, chunks, m, pp, forward=True):
    if m < pp:
        base = m
        group_size = chunks * base
        if not forward:
            ch_id = chunks - ch_id - 1
        m_id = g_id * group_size + ch_id * base + b_id
        return m_id
    else:
        base = pp
        group_size = chunks * base
        max_num_groups = (m + pp) // pp
        res_base = m % pp
        if g_id >= max_num_groups - 1:
            base = res_base
        if not forward:
            ch_id = chunks - ch_id - 1
        m_id = g_id * group_size + ch_id * base + b_id
        return m_id


def generate(pp, m, layers, chunk_layers, fwd_time=10, bwd_time=20, msg_time=1):
    device_worker_arr = []
    chunks = layers // chunk_layers
    init_state_info(chunks, m, pp)
    for i in range(pp):
        device_worker_arr.append(_init_device_workers(i, chunks, chunk_layers, m, pp, fwd_time, bwd_time, msg_time))
    thread_arr = []
    for i in range(pp):
        f_path = f'simple_pp_rank_{i}'
        fwd_workers, bwd_workers = device_worker_arr[i]
        t = Thread(target=_generate_one_device,
                   args=(fwd_workers, bwd_workers,
                         i, chunks, layers, m,  pp, f_path))
        thread_arr.append(t)
    for t in thread_arr:
        t.start()

    for t in thread_arr:
        t.join()


def init_state_info(chunks, m, pp):
    global fwd_states
    global bwd_states
    for i in range(pp):
        s_fwd_input, s_bwd_input = _init_single_state(i, chunks, m, pp)
        fwd_states.append(s_fwd_input)
        bwd_states.append(s_bwd_input)


def _init_single_state(rank, chunks, m, pp):
    tot_num = m * chunks
    fwd_input_arr = [-1] * tot_num
    bwd_input_arr = [-1] * tot_num
    for i in range(tot_num):
        ch_id = _get_chunk_id(i, chunks, m, pp, True)
        if rank == 0 and ch_id == 0:
            fwd_input_arr[i] = 0.
        ch_id = _get_chunk_id(i, chunks, m, pp, False)
        if rank == pp - 1 and ch_id == chunks - 1:
            bwd_input_arr[i] = 0. 
    return fwd_input_arr,  bwd_input_arr


def _generate_one_device(fwd_workers, bwd_wokers, rank, chunks, layers, m, pp, f_path):
    if chunks == 1:
        _generate_no_interleave(fwd_workers, bwd_wokers, rank, chunks, layers, m, pp, f_path)
    else:
        _generate_interleave(fwd_workers, bwd_wokers, rank, chunks, layers, m, pp, f_path)


def _generate_no_interleave(fwd_workers, bwd_workers, rank, chunks, layers, m, pp, trace_f_name):
    warmup_num =  pp - rank - 1
    _gen_general(warmup_num, fwd_workers, bwd_workers,
                 rank, chunks, layers, m, pp, trace_f_name)


def _generate_interleave(fwd_workers, bwd_workers, rank, chunks, layers, m, pp, trace_f_name):
    # warmup stage
    warmup_num = 2 * (pp - rank - 1) + (chunks - 1) * pp
    # warmup_num = (pp - rank - 1) + (chunks - 1) * pp
    _gen_general(warmup_num, fwd_workers, bwd_workers,
                 rank, chunks, layers, m, pp, trace_f_name)


def _gen_general(warmup_num, fwd_workers, bwd_workers,
                 rank, chunks, layers, m, pp, trace_f_name):
    tot_num = len(fwd_workers)
    warmup_num = min(warmup_num, tot_num)
    cur_time = 0.

    with open(trace_f_name, 'w') as _in:
        for i in range(warmup_num):
            fwd_worker = fwd_workers[i]
            cur_time, fwd_info = run_forward(cur_time, fwd_worker, rank, chunks, m, pp)
            _in.write(json.dumps(fwd_info) + "\n")
        # steady stage
        remain_num = tot_num - warmup_num
        if remain_num > 0:
            for i in range(remain_num):
                m_id = i + warmup_num
                fwd_worker = fwd_workers[m_id]
                cur_time, fwd_info = run_forward(cur_time, fwd_worker, rank, chunks, m, pp)
                _in.write(json.dumps(fwd_info) + "\n")
                bwd_worker = bwd_workers[i]
                cur_time, bwd_info = run_backward(cur_time, bwd_worker, rank, chunks, m, pp)
                _in.write(json.dumps(bwd_info) + "\n")

        # cooldown stage
        for i in range(warmup_num):
            m_id = i + remain_num
            bwd_worker = bwd_workers[m_id]
            cur_time, bwd_info = run_backward(cur_time, bwd_worker, rank, chunks, m, pp)
            _in.write(json.dumps(bwd_info) + "\n")


def run_forward(cur_time, fwd_worker, rank, chunks, m, pp):
    need_wait = True
    micro_id = fwd_worker.micro_id
    stage = fwd_worker.stage
    exec_time = fwd_worker.exec_time
    next_ss = cur_time
    res = {'sign': 'fwd', 'start_time': cur_time, 'micro_id': micro_id}
    global fwd_states
    global lock
    while need_wait:
        print(f" in_forward, rank is: {rank}, micro_id: {micro_id} to start acquire lock")
        lock.acquire()
        print(f"in_forward, rank is: {rank}, micro_id: {micro_id} after acquire lock")
        ss = fwd_states[stage][micro_id]
        print(f"in_forward, rank is: {rank}, micro_id: {micro_id} ss: {ss}")
        if ss >= 0.:
            print(f"in_forward, rank is: {rank}, micro_id: {micro_id} enter into ")
            if cur_time < ss:
                cur_time = ss
            next_ss = cur_time + exec_time
            next_state = next_ss + fwd_worker.pp_msg_time
            res['end_time'] = next_ss
            res['start_time'] = cur_time
            if stage != pp - 1:
                fwd_states[stage + 1][micro_id] = next_state
                res['pp_comm'] = {
                    'start_time': next_ss,
                    'end_time': next_state,
                    'src': stage, 'dst': stage + 1, 'op': 'sendmsg',
                }
            else:
                g_id, ch_id, b_id = \
                    _mid2detail_id(micro_id, chunks, m, pp, forward=True)
                if ch_id < chunks - 1:
                    ch_id += 1
                    new_mid = _detail_id2mid(g_id, ch_id,
                                             b_id, chunks, m, pp, forward=True)
                    fwd_states[0][new_mid] = next_state
                    res['pp_comm'] = {
                        'start_time': next_ss,
                        'end_time': next_state,
                        'src': stage, 'dst': 0, 'op': 'sendmsg',
                    }

            print(f"in_forward, rank is: {rank}, micro_id: {micro_id} before lock release ")
            lock.release()
            print(f"in_forward, rank is: {rank}, micro_id: {micro_id} lock release ")
            need_wait = False
            break
        else:
            print(f"in_forward, rank is: {rank}, micro_id: {micro_id} before lock release  when ss < 0")
            lock.release()
            print(f"in_forward, rank is: {rank}, micro_id: {micro_id} after lock release  when ss < 0")
            time.sleep(1)
    print(f"in_forward, rank is: {rank}, micro_id: {micro_id} finished")
    return next_ss, res


def run_backward(cur_time, bwd_worker, rank, chunks, m, pp):
    need_wait = True
    micro_id = bwd_worker.micro_id
    stage = bwd_worker.stage
    exec_time = bwd_worker.exec_time
    next_ss = cur_time
    global bwd_states
    global lock
    res = {'sign': 'bwd', 'start_time': cur_time, 'micro_id': micro_id}
    while need_wait:
        print(f" in bwd rank is: {rank}, micro_id: {micro_id} to start acquire lock")
        lock.acquire()
        print(f" in bwd rank is: {rank}, micro_id: {micro_id} to after acquire lock")
        ss = bwd_states[stage][micro_id]
        if ss >= 0.:
            if cur_time < ss:
                cur_time = ss
            next_ss = cur_time + exec_time
            next_state = next_ss + bwd_worker.pp_msg_time
            res['end_time'] = next_ss
            res['start_time'] = cur_time
            if stage != 0:
                bwd_states[stage-1][micro_id] = next_state
                res['pp_comm'] = {
                    'start_time': next_ss,
                    'end_time': next_state,
                    'src': stage, 'dst': stage - 1, 'op': 'sendmsg',
                }
            else:
                g_id, ch_id, b_id = \
                    _mid2detail_id(micro_id, chunks, m, pp, forward=False)
                if ch_id > 0:
                    ch_id -= 1
                    new_mid = _detail_id2mid(g_id, ch_id, b_id,
                                             chunks, m, pp, forward=False)
                    bwd_states[pp - 1][new_mid] = next_state
                    res['pp_comm'] = {
                        'start_time': next_ss,
                        'end_time': next_state,
                        'src': stage, 'dst': pp - 1, 'op': 'sendmsg',
                    }
            lock.release()
            need_wait = False
            break
        else:
            lock.release()
            time.sleep(0.01)
    return next_ss, res


def _init_device_workers(device, chunks, layers,
                         m, pp, fwd_time, bwd_time, pp_msg_time):
    fwd_workers = []
    bwd_workers = []
    tot_num = chunks * m
    for i in range(tot_num):
        ch_id = _get_chunk_id(i, chunks, m, pp)
        fwd_worker = FwdWorker(i, device, ch_id, layers, fwd_time, pp_msg_time)
        ch_id = _get_chunk_id(i, chunks, m, pp, forward=False)
        bwd_worker = BwdWorker(i, device, ch_id, layers, bwd_time, pp_msg_time)
        fwd_workers.append(fwd_worker)
        bwd_workers.append(bwd_worker)
    return fwd_workers, bwd_workers


if __name__ == '__main__':
    pp = 3
    m = 6
    layers = 8
    chunk_layers = 1 
    generate(pp, m, layers, chunk_layers,
             fwd_time=10, bwd_time=20, msg_time=1)
