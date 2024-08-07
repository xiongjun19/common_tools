# coding=utf8


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
        # else:
        #     ch_id = (micro_id - reg_tho) // res_base
        #     b_id = (micro_id - reg_tho) % res_base
        #     if not forward:
        #         ch_id = chunks - ch_id - 1
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
        # reg_tho = (max_num_groups - 1) * group_size
        res_base = m % pp
        if g_id >= max_num_groups - 1:
            base = res_base
        if not forward:
            ch_id = chunks - ch_id - 1
        m_id = g_id * group_size + ch_id * base + b_id
        return m_id


def is_last_model_chunk(micro_id, chunks, m, pp):
    if m < pp:
        base = m
        group_size = chunks * base
        g_id = micro_id // group_size
        b_id = micro_id % base
        if b_id == base - 1:
            return True
        return False
    else:
        base = pp
        group_size = chunks * base
        max_num_groups = (m + pp) // pp
        reg_tho = (max_num_groups - 1) * group_size
        res_base = m % pp
        g_id = micro_id // group_size
        if res_base == 0:
            max_num_groups -= 1
            b_id = micro_id % group_size % base
            return g_id == max_num_groups - 1 and b_id == base - 1
        else:
            base = res_base
            if micro_id < reg_tho:
                return False
            micro_id = (micro_id - reg_tho)
            b_id = micro_id % base
            return b_id == base - 1



if __name__ == '__main__':
    res_arr = []
    res_bwd_arr = []
    m = 8
    chunks = 2
    for m_id in range(m * chunks):
        ch_id = _get_chunk_id(m_id, chunks, m, 3, forward=True)
        res_arr.append(ch_id)

        ch_id = _get_chunk_id(m_id, chunks, m, 3, forward=False)
        res_bwd_arr.append(ch_id)

    print("fwd_chunks_id: ")
    print(list(range(m * chunks)))
    print(res_arr)
    print("bwd_chunks_id: ")
    print(res_bwd_arr)

    print("* " * 35)
    res_arr = []
    res_bwd_arr = []
    m = 8 
    chunks = 2
    for m_id in range(m * chunks):
        ch_id = _mid2detail_id(m_id, chunks, m, 3, forward=True)
        res_arr.append(ch_id)
        ch_id = _mid2detail_id(m_id, chunks, m, 3, forward=False)
        res_bwd_arr.append(ch_id)

    print("fwd_chunks_id: ")
    print(list(range(m * chunks)))
    print(res_arr)
    print("bwd_chunks_id: ")
    print(res_bwd_arr)

    print("* " * 35)
    fwd_m_id = []
    bwd_m_id = []
    last_chunk_state = []
    for i in range(len(res_arr)):
        g_id, ch_id, b_id = res_arr[i]
        m_id = _detail_id2mid(g_id, ch_id, b_id, chunks, m, 3)
        fwd_m_id.append(m_id)
        g_id, ch_id, b_id = res_bwd_arr[i]
        m_id = _detail_id2mid(g_id, ch_id, b_id, chunks, m, 3, False)
        bwd_m_id.append(m_id)
        last_chunk_state.append(is_last_model_chunk(i, chunks, m, 3))

    print("original_id: ")
    print(list(range(m * chunks)))
    print("fwd_m_id: ")
    print(fwd_m_id)
    print("bwd_m_id: ")
    print(bwd_m_id)
    print("last_chunk states: ")
    print(last_chunk_state)

