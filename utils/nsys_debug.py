# coding=utf8


import sqlite3


def do_parse(f_path, dev):
    mydb = sqlite3.connect(f_path)
    cursor = mydb.cursor()
    tot_time = _parse_tot_time(cursor)
    kernel_time, nccl_time = _parse_kernel(cursor, dev)
    mem_time = _parse_mem(cursor, dev)
    utilization = (kernel_time - nccl_time) / tot_time
    nccl_ratio = nccl_time / (1e-9 + kernel_time + mem_time)
    mem_ratio = mem_time / (1e-9 + kernel_time + mem_time)
    cursor.close()
    mydb.close()
    return tot_time, utilization, nccl_ratio, mem_ratio


def _parse_tot_time(cursor):
    sql = "select duration from ANALYSIS_DETAILS;" # return [(802818559,)]
    tot  = exec_and_parse(cursor, sql)
    return tot


def _parse_kernel(cursor, device=0):
    sql = f"select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId={device};"
    ker_time = exec_and_parse(cursor, sql)
    sql = f"select sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL where deviceId={device} AND Shortname in (select id from StringIds where  value like '%nccl%');"
    nccl_time = exec_and_parse(cursor, sql)
    print(f"ker_time @gpu {device} is: {ker_time}, nccl_time: {nccl_time}, pure_ker: {ker_time - nccl_time}")
    return ker_time, nccl_time


def _parse_mem(cursor, device=0):
    sql = f"select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMSET where deviceId={device};"
    set_time = exec_and_parse(cursor, sql)
    sql = f"select sum(end - start) from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId={device};"
    cpy_time = exec_and_parse(cursor, sql)
    res = set_time + cpy_time
    return res


def exec_query(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()

def exec_and_parse(cursor, sql):
    try:
        cursor.execute(sql)
        items = cursor.fetchall()
        item = items[0][0]
        res = 0.
        if item is  not None:
            res = float(item)
        return res
    except sqlite3.OperationalError as e:
        print(e)
        return 0.


def test(f_path, device=0):
    res = do_parse(f_path, device)
    print(res)
    return res


def main(f_path, device_arr):
    for device in device_arr:
        res = do_parse(f_path, device)
        tot_time, utilization, nccl_ratio, mem_ratio = res
        print(f'GPU{device}: utilization: {round(utilization, 3)}; nccl_ratio: {round(nccl_ratio, 3)}; mem_ratio: {round(mem_ratio, 3)}')


if __name__ == '__main__':
    import sys
    device_num = 1
    t_path = sys.argv[1]
    if len(sys.argv) > 2:
        device_num = int(sys.argv[2])
    device_arr = range(device_num)
    main(t_path, device_arr)

