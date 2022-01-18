# coding=utf8

"""
所有节点分为：
   1. 未知节点；
   2. embed点；
   3. layer 节点；
   4. tail 节点
本文件主要用来分析onnx的输入输出信息， 得到
1. 每个layer的输入输出信息， 让它变得更加好读一点
     输入信息：
        输入的是谁的输出， 否则就是一个外部节点
     输出信息：
         输出的节点编号， 相较于上一个输出的变化；
"""

import json
import argparse
import copy
from dataclasses import dataclass


LAY_BEG_IDX = 44
LAY_OPS = 192
node_types = ['unk', 'emb_node', 'layer_node', 'tail_node']
unk_types = ['smaller_unk', 'larger_unk', 'weight_unk', 'unk_unk']


@dataclass
class OnnxNode:
    node_name: str
    layer_num: str
    node_type: str


@dataclass
class TensorInfo:
    tensor_id: str
    node_name: str

@dataclass
class UnkProp:
    tensor_id: str
    unk_type: str


def parse_seq(f_path):
    with open(f_path) as in_:
        dict_ = json.load(in_)
        elems =[0] * len(dict_)
        for key, val in dict_.items():
            key_name, key_num = key.split('_')
            key_num = int(key_num)
            elems[key_num] = (key, val)
        return elems


class Worker(object):
    def __init__(self, f_path, layers):
        self.seqs = parse_seq(f_path)
        self.layers = layers
        self.node_map = self._cons_node_map()
        self.tensor_map = self._cons_tensor_map()
        self.begin_tensor = self._get_begin_tensor()
        self.lg_out_tensor = self._get_largest_out()
        self.t_unk_map = self._cons_t_unk_map()
        out_diff = self.get_layer_out_info(1)
        self.out_diff = [int(x[1][0]) for x in out_diff]
        in_diff = self.get_layer_info(1)
        self.in_diff = [[int(x[-1]) for x in y[1] if x[-1].isnumeric()] for y in in_diff]

    def _get_begin_tensor(self):
        out_arr = list(self.seqs[0][1]['outputs'].keys())
        return int(out_arr[0])

    def _get_largest_out(self):
        out_arr = list(self.seqs[-2][1]['outputs'].keys())
        return int(out_arr[0])

    def _cons_node_map(self):
        res = {}
        for node_name, _ in self.seqs:
            node_name_arr = node_name.split("_")
            node_num = int(node_name_arr[1])
            node_type = node_types[self._get_node_type(node_num)]
            ly_num = self._get_layer_num(node_num)
            res[node_name] = OnnxNode(node_name, ly_num, node_type)
        return res

    def _cons_tensor_map(self):
        res = {}
        for node_name, node_info in self.seqs:
            outputs = node_info['outputs']
            for key in outputs.keys():
                res[key] = TensorInfo(key, node_name)
        return res

    def _cons_t_unk_map(self):
        res = {}
        for _, node_info in self.seqs:
            inputs = node_info['inputs']
            for t_id in inputs.keys():
                unk_type = self._comp_unk_type(t_id)
                if unk_type is not None:
                    res[t_id] = unk_type
            outputs = node_info['outputs']
            for t_id in outputs.keys():
                unk_type = self._comp_unk_type(t_id)
                if unk_type is not None:
                    res[t_id] = unk_type
        return res

    def _comp_unk_type(self, t_id):
        # unk_types = ['smaller_unk', 'larger_unk', 'weight_unk', 'unk_unk']
        n_type = self.get_tensor_type(t_id)
        if n_type != node_types[0]:
            return None
        idx = -1
        if t_id.isnumeric():
            t_id_int = int(t_id)
            if t_id_int < self.begin_tensor:
                idx = 0  # unk_smaller
                return unk_types[idx]
            if t_id_int > self.lg_out_tensor:
                idx = 1  # unk_larger
                return unk_types[idx]
        elif "model" in t_id:
            idx = 2  # weight_unk
            return unk_types[idx]
        return unk_types[idx]

    def _get_node_type(self, node_num):
        if 0 <= node_num < LAY_BEG_IDX:
            return 1
        if LAY_BEG_IDX <= node_num < LAY_BEG_IDX + self.layers * LAY_OPS:
            return 2
        if LAY_BEG_IDX + self.layers * LAY_OPS <= node_num < len(self.seqs):
            return 3
        return 0

    def _get_layer_num(self, node_num):
        if LAY_BEG_IDX <= node_num < LAY_BEG_IDX + self.layers * LAY_OPS:
            res = str((node_num - LAY_BEG_IDX) // LAY_OPS)
            return res
        return "unk"

    def _get_n_ds(self, node_name):
        return self.node_map.get(node_name,
                                 OnnxNode('unk', 'unk', node_types[0]))

    def _get_ts_info(self, tensor_id):
        return self.tensor_map.get(tensor_id, TensorInfo('tensor_id', 'unk'))

    def get_tensor_type(self, tensor_id):
        ts_info = self._get_ts_info(tensor_id)
        node_name = ts_info.node_name
        node_ds = self._get_n_ds(node_name)
        node_type = node_ds.node_type
        return node_type

    def get_tensor_unk(self, tensor_id):
        unk_type  = self.t_unk_map.get(tensor_id, None)
        if unk_type is None:
            if 'larger_unk' in tensor_id:
                return unk_types[1]
        return unk_type

    def get_layer_info(self, l_num):
        assert l_num < self.layers, f"l_num should less than total layers: {self.layers}"
        beg_idx = LAY_BEG_IDX + l_num * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        res = []
        for name, node_info in sub_arr:
            input_ids = node_info['inputs']
            node_info_arr = []
            for t_id in input_ids.keys():
                ts_info = self._get_ts_info(t_id)
                node_info = self._get_n_ds(ts_info.node_name)
                diff = self._calc_node_name_diff(name, node_info)
                node_info_arr.append([t_id, node_info, diff])
            res.append([name, node_info_arr])
        return res

    def _calc_node_name_diff(self, name, node_info):
        if node_info.node_type == node_types[0]:
            return "unk"
        if node_info.node_type != node_types[2]:
            return "unk"
        new_name = node_info.node_name
        num1 = name.split("_")[-1]
        num2 = new_name.split("_")[-1]
        diff = _cus_calc_out_diff(num1, num2)
        return diff

    def get_layer_out_info(self, l_num):
        assert l_num < self.layers, f"l_num should less than total layers: {self.layers}"
        beg_idx = LAY_BEG_IDX + l_num * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        res = []
        for i, elem in enumerate(sub_arr):
            name, node_info = elem
            outputs = node_info['outputs']
            pre_elem = self.seqs[beg_idx + i - 1]
            pre_node_info = pre_elem[1]
            pre_outputs = pre_node_info['outputs']
            pre_diff = _calc_diff(list(outputs.keys()),
                                  list(pre_outputs.keys()))
            res.append([name, pre_diff])
        return res

    def print_special(self):
        print("the special info about this info is: ")
        print(" @@@ " *  20)
        print(f"layers num is: {self.layers}")
        print(f"smallest out tenor_id is: {self.begin_tensor}")
        print(f"largest out tensor id is: {self.lg_out_tensor}")
        print(f"following is unk tensor_id: ")
        for key, val in self.t_unk_map.items():
            print(f"{key}\t{val}")
        print(f'following is out_diff info')
        print(self.out_diff)
        print(f'following is in_diff info')
        print(self.in_diff)
        print(" @@@ " *  20)


    def get_a_layer(self, layer_idx, need_norm=False):
        beg_idx = LAY_BEG_IDX + layer_idx * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        if need_norm:
            self._norm_seqs(sub_arr)
        return sub_arr

    def generate_a_layer(self, last_layer_num, need_norm=False):
        # deep_copy last_layer_info
        beg_idx = LAY_BEG_IDX + last_layer_num * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        if need_norm:
            self._norm_seqs(sub_arr)
        new_arr = copy.deepcopy(sub_arr)
        # modify each elem, based on sub_arr , and new_arr
        layer_idx = last_layer_num + 1
        for i, elem in enumerate(new_arr):
            new_elem = self._modify_layer_elem(elem, i, new_arr, sub_arr, layer_idx)
            new_arr[i] = new_elem
        return new_arr

    def _modify_layer_elem(self, elem, i, new_arr, sub_arr, layer_idx):
        n_name, n_info = elem
        # update node name
        n_name_arr = n_name.split("_")
        n_num = int(n_name_arr[-1])
        n_num += LAY_OPS
        prefix_arr = n_name_arr[:-1]
        prefix_arr.append(str(n_num))
        new_name = "_".join(prefix_arr)
        # update input tensor
        self._update_layer_input_tensor(n_info, i, new_arr, sub_arr, layer_idx)
        # update output tensor
        self._update_layer_output_tensor(n_info, i, new_arr, sub_arr, layer_idx)
        return [new_name, n_info]

    def _update_layer_input_tensor(self, node_info, i, new_arr, sub_arr, layer_idx):
        # node_types = ['unk', 'emb_node', 'layer_node', 'tail_node']
        inputs = node_info['inputs']
        res = {}
        cnt = 0
        for t_id, val in inputs.items():
            new_t_id = t_id
            node_type = self.get_tensor_type(t_id)
            # 当tensor 是layer_node 的输出时， 需要依据差值计算出是那个layer_node的输出
            if node_type == node_types[2]:
                diff_arr = self.in_diff[i]
                diff_num = diff_arr[cnt]
                cnt += 1
                pre_elem_idx = i - diff_num
                # import ipdb; ipdb.set_trace()
                pre_elem = None
                if pre_elem_idx < 0:
                    pre_elem = sub_arr[pre_elem_idx]
                else:
                    pre_elem = new_arr[pre_elem_idx]
                if pre_elem is None:
                    import ipdb; ipdb.set_trace()
                new_t_id = list(pre_elem[1]['outputs'].keys())[0]
            # 当tensor 是unk, 并且是weight_unk的时候， 需要依据模型的的layer 来更改tensor_id
            # unk_types = ['smaller_unk', 'larger_unk', 'weight_unk', 'unk_unk']
            elif node_type == node_types[0]:
                unk_type = self.get_tensor_unk(t_id)
                if unk_type == unk_types[2]:
                    t_id_arr = t_id.split(".")
                    t_id_arr[2] = str(layer_idx)
                    new_t_id = ".".join(t_id_arr)
            res[new_t_id] = val
        node_info['inputs'] = res

    def _update_layer_output_tensor(self, node_info, i, new_arr, sub_arr, layer_idx):
        outputs = node_info['outputs']
        res = {}
        for t_id, val in outputs.items():
            pre_idx = i - 1
            if pre_idx < 0:
                pre_elem = sub_arr[pre_idx]
            else:
                pre_elem = new_arr[pre_idx]
            diff_num = self.out_diff[i]
            pre_t_id = list(pre_elem[1]['outputs'].keys())[0]
            pre_t_id = int(pre_t_id)
            new_t_id = str(pre_t_id + diff_num)
            res[new_t_id] = val
        node_info['outputs'] = res

    def _norm_seqs(self, seqs):
        cnt = 0
        for _, node_info in seqs:
            inputs = node_info['inputs']
            n_inputs = {}
            for key, val in inputs.items():
                new_key = key
                t_unk_type = self.get_tensor_unk(key)
                if t_unk_type is not None and unk_types[1] == t_unk_type:
                    new_key = f'{unk_types[1]}_sep_{cnt}'
                    cnt += 1
                n_inputs[new_key] = val
            node_info['inputs'] = n_inputs


def _calc_diff(l1, l2):
    s1 = sorted(l1)
    s2 = sorted(l2)
    _len = min(len(s1), len(s2))
    res = []
    for i in range(_len):
        cur_res = _cus_calc_out_diff(s1[i], s2[i])
        res.append(cur_res)
    return res


def _cus_calc_out_diff(str1, str2):
    if str1.isnumeric() and str2.isnumeric():
        res = int(str1) - int(str2)
        return str(res)
    if str1 == str2:
        return '0'
    res = f'({str1} - {str2})'
    return res


def inspect_out_diff_var(worker, layers):
    out_diff_arr = []
    for l_num in range(1, layers):
        out_diff_arr.append(worker.get_layer_out_info(l_num))
    res_arr = []
    var_sum = 0
    for i in range(1, len(out_diff_arr)):
        out_diff2 = out_diff_arr[i]
        out_diff = out_diff_arr[i-1]
        cur_res = []
        for elem1, elem2 in zip(out_diff2, out_diff):
            n1, diff_arr1 = elem1
            n2, diff_arr2 = elem2
            var_arr = _calc_diff(diff_arr1, diff_arr2)
            for elem in var_arr:
                var_sum += int(elem)
            cur_res.append([n1, n2, var_arr])
        res_arr.append(cur_res)
    return res_arr, var_sum


def inspect_in_diff_var(worker, layers):
    in_layer_arr = []
    for l_num in range(1, layers):
        in_layer_arr.append(worker.get_layer_info(l_num))
    res_arr = []
    var_sum = 0
    for i in range(1, len(in_layer_arr)):
        layer_elem_arr2 = in_layer_arr[i]
        layer_elem_arr = in_layer_arr[i-1]
        cur_res = []
        for elem1, elem2 in zip(layer_elem_arr2, layer_elem_arr):
            # import ipdb; ipdb.set_trace()
            n1, diff_arr1 = elem1
            diff_arr1 = [x[-1] for x in diff_arr1]
            n2, diff_arr2 = elem2
            diff_arr2 = [x[-1] for x in diff_arr2]
            var_arr = _calc_diff(diff_arr1, diff_arr2)
            for _x in var_arr:
                if _x.isnumeric():
                    var_sum += abs(int(_x))
            cur_res.append([n1, n2, var_arr])
        res_arr.append(cur_res)
    return res_arr, var_sum


def inspect_out_diff(worker, layer_idx):
    info_arr = worker.get_layer_out_info(layer_idx)
    print_arr(info_arr)


def print_arr(arr):
    for i, elem in enumerate(arr):
        print("***" * 30)
        print(f"info for layer is {i} ")
        l_info_arr = elem
        for l_info in l_info_arr:
            print(l_info)


def test(args):
    in_path = args.input
    layers = args.f_ls
    t_worker = Worker(in_path, layers)
    for l_num in range(1, layers):
        print("***" * 30)
        print(f"info for layer is {l_num} ")
        l_info_arr = t_worker.get_layer_info(l_num)
        for l_info in l_info_arr:
            print(l_info)
    # t_worker.print_special()
    # diff_arr_var, var_sum = inspect_out_diff_var(t_worker, layers)
    # print_arr(diff_arr_var)
    # print(f"diff var sum is: {var_sum}")
    # inspect_out_diff(t_worker, 1)
    # in_arr_var, in_var_sum = inspect_in_diff_var(t_worker, layers)
    # print_arr(in_arr_var)
    # print(f"in diff var sum is: {in_var_sum}")
    print("now compare generated and non_generated")
    generated_layer = t_worker.generate_a_layer(layers - 2, need_norm=True)
    org_layer = t_worker.get_a_layer(layers - 1, need_norm=True)
    print_arr(zip(org_layer, generated_layer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-r', '--ref', type=str)
    parser.add_argument('-l1', '--f_ls', type=int, help="the first layers")
    parser.add_argument('-l2', '--f_ls2', type=int, help="the second layers")
    parser.add_argument('-n', '--num', type=int, default=48)
    parser.add_argument('-o', '--output', type=str, help='save the diff result')
    args = parser.parse_args()
    test(args)

