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
from collections import OrderedDict


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
        self.nid_2_tensor = self._cons_nid_2_tensor()
        self.begin_tensor = self._get_begin_tensor()
        self.lg_out_tensor = self._get_largest_out()
        self.t_unk_map = self._cons_t_unk_map()
        # import ipdb; ipdb.set_trace()
        out_diff = self.get_layer_out_info(1)
        self.out_diff = [int(x[1][0]) for x in out_diff]
        in_diff, out_idx_arr = self.get_layer_info(1)
        self.in_diff = [[int(x[-1]) for x in y[1] if x[-1].isnumeric()] for y in in_diff]
        self.in_out_idx_arr = out_idx_arr
        tail_out_diff = self.get_tail_out_info()
        self.tail_out_diff = [int(x[1][0]) for x in tail_out_diff if x[1][0].isnumeric()]
        tail_in_diff = self.get_tail_input_info()
        self.tail_in_diff = [[int(x[-1]) for x in y[1] if x[-1].isnumeric()] for y in tail_in_diff]

    def _update_specs(self):
        self.begin_tensor = self._get_begin_tensor()
        self.lg_out_tensor = self._get_largest_out()

    def _update_map_info(self):
        self.node_map = self._cons_node_map()
        self.tensor_map = self._cons_tensor_map()
        self.t_unk_map = self._cons_t_unk_map()
        self.nid_2_tensor = self._cons_nid_2_tensor()

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
                if key in res:
                    print("already output by other node ?")
                res[key] = TensorInfo(key, node_name)
        return res

    def _cons_nid_2_tensor(self):
        """
        用来构建node_id 到其所有输出tensor_id的map
        """
        res = {}
        for node_name, node_info in self.seqs:
            outputs = node_info['outputs']
            res[node_name] = list(sorted(outputs.keys()))
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
        unk_type = self.t_unk_map.get(tensor_id, None)
        if unk_type is None:
            if 'larger_unk' in tensor_id:
                return unk_types[1]
        return unk_type

    def get_layer_info(self, l_num):
        assert l_num < self.layers, f"l_num should less than total layers: {self.layers}"
        beg_idx = LAY_BEG_IDX + l_num * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        res = []
        res_out_idx = []
        for name, node_info in sub_arr:
            input_ids = node_info['inputs']
            node_info_arr = []
            out_idx_arr = []
            for t_id in input_ids.keys():
                ts_info = self._get_ts_info(t_id)
                node_info = self._get_n_ds(ts_info.node_name)
                diff = self._calc_node_name_diff(name, node_info)
                node_info_arr.append([t_id, node_info, diff])
                if node_info.node_name in self.nid_2_tensor and t_id.isnumeric():
                    prev_out_tensor_arr = self.nid_2_tensor.get(node_info.node_name)
                    out_idx = prev_out_tensor_arr.index(t_id)
                    out_idx_arr.append(out_idx)
            res.append([name, node_info_arr])
            res_out_idx.append(out_idx_arr)
        return res, res_out_idx

    def get_tail_input_info(self):
        beg_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        sub_arr = self.seqs[beg_idx:]
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
        if node_info.node_type != node_types[2] and node_info.node_type != node_types[3]:
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

    def get_tail_out_info(self):
        beg_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        sub_arr = self.seqs[beg_idx:]
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
        print(" @@@ " * 20)
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
        print(" @@@ " * 20)

    def get_a_layer(self, layer_idx, need_norm=False):
        beg_idx = LAY_BEG_IDX + layer_idx * LAY_OPS
        sub_arr = self.seqs[beg_idx:beg_idx + LAY_OPS]
        if need_norm:
            self._norm_seqs(sub_arr)
        return sub_arr

    def get_tail_layer(self, need_norm=False):
        beg_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        sub_arr = self.seqs[beg_idx:]
        if need_norm:
            self._norm_seqs(sub_arr)
        return sub_arr

    def increase_once(self):
        # first norm_seqs
        self._norm_seqs(self.seqs)
        # generate a layer
        new_layer = self.generate_a_layer(self.layers-1)
        # update layer info and layers
        last_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        new_seqs = self.seqs[0:last_idx]
        new_seqs.extend(new_layer)
        new_seqs.extend(self.seqs[last_idx:])
        self.seqs = new_seqs
        self.layers += 1
        # generate  tail
        new_tail = self.generate_tail()
        # update tail to seqs
        beg_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        for i, elem in enumerate(new_tail):
            self.seqs[beg_idx+i] = elem
        self._update_specs()
        # convert specical tokens back
        self.post_process()
        self._update_map_info()

    def increase_n_layers(self, n):
        for i in range(n):
            self.increase_once()

    def post_process(self):
        # unk_types = ['smaller_unk', 'larger_unk', 'weight_unk', 'unk_unk']
        cnt = 0
        initial_val =  self.lg_out_tensor + 2
        for elem in self.seqs:
            node_info = elem[1]
            inputs = node_info['inputs']
            new_inputs = {}
            for key, val in inputs.items():
                new_key = key
                if unk_types[1] in key:
                    new_key = str(initial_val + cnt)
                    cnt += 1
                new_inputs[new_key] = val
            node_info['inputs'] = new_inputs

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
                idx_arr = self.in_out_idx_arr[i]
                diff_num = diff_arr[cnt]
                idx_num = idx_arr[cnt]
                cnt += 1
                pre_elem_idx = i - diff_num
                pre_elem = None
                if pre_elem_idx < 0:
                    pre_elem = sub_arr[pre_elem_idx]
                else:
                    pre_elem = new_arr[pre_elem_idx]
                if pre_elem is None:
                    import ipdb; ipdb.set_trace()
                new_t_id = list(pre_elem[1]['outputs'].keys())[idx_num]
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
        out_arr = list(sorted(outputs.items(), key=lambda x: x[0]))
        t_id, val = out_arr[0]
        new_t_id = self._update_output_by_prev_out(i, new_arr, sub_arr, res, val)
        if len(out_arr) > 1:
            for i in range(1, len(out_arr)):
                cur_t_id, cur_val = out_arr[i]
                diff = int(cur_t_id) - int(t_id)
                cur_new_t_id = str(int(new_t_id) + diff)
                res[cur_new_t_id] = cur_val
        node_info['outputs'] = res

    def _update_output_by_prev_out(self, i, new_arr, sub_arr, tmp_dict, val):
        pre_idx = i - 1
        if pre_idx < 0:
            pre_elem = sub_arr[pre_idx]
        else:
            pre_elem = new_arr[pre_idx]
        diff_num = self.out_diff[i]
        pre_t_id = sorted(list(pre_elem[1]['outputs'].keys()))[0]
        pre_t_id = int(pre_t_id)
        new_t_id = str(pre_t_id + diff_num)
        tmp_dict[new_t_id] = val
        return new_t_id

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

    def generate_tail(self, need_norm=False):
        # deep_copy last_layer_info
        beg_idx = LAY_BEG_IDX + self.layers * LAY_OPS
        sub_arr = self.seqs[beg_idx:]
        if need_norm:
            self._norm_seqs(sub_arr)
        new_arr = copy.deepcopy(sub_arr)
        # modify each elem, based on sub_arr , and new_arr
        for i, elem in enumerate(new_arr):
            new_elem = self._modify_tail_elem(elem, i, new_arr, beg_idx)
            new_arr[i] = new_elem
        return new_arr

    def _modify_tail_elem(self, elem, i, new_arr, beg_idx):
        n_name, n_info = elem
        # update node name
        if i == 0:
            pre_elem = self.seqs[beg_idx-1]
        else:
            pre_elem = new_arr[i - 1]
        pre_name = pre_elem[0]
        n_name_arr = n_name.split("_")
        n_num = int(pre_name.split("_")[-1]) + 1
        prefix_arr = n_name_arr[:-1]
        prefix_arr.append(str(n_num))
        new_name = "_".join(prefix_arr)
        # update input tensor
        self._update_tail_input_tensor(n_info, i, new_arr, beg_idx)
        # update output tensor
        self._update_tail_output_tensor(n_info, i, new_arr, beg_idx)
        return [new_name, n_info]

    def _update_tail_input_tensor(self, node_info, i, new_arr, beg_idx):
        # node_types = ['unk', 'emb_node', 'layer_node', 'tail_node']
        inputs = node_info['inputs']
        res = {}
        cnt = 0
        for t_id, val in inputs.items():
            new_t_id = t_id
            node_type = self.get_tensor_type(t_id)
            # 当tensor 是layer_node 的输出时， 需要依据差值计算出是那个layer_node的输出
            if node_type == node_types[2] or node_type == node_types[3]:
                diff_arr = self.tail_in_diff[i]
                diff_num = diff_arr[cnt]
                cnt += 1
                pre_elem_idx = i - diff_num
                pre_elem = None
                if pre_elem_idx < 0:
                    pre_elem = self.seqs[beg_idx + pre_elem_idx]
                else:
                    pre_elem = new_arr[pre_elem_idx]
                new_t_id = list(pre_elem[1]['outputs'].keys())[0]
            res[new_t_id] = val
        node_info['inputs'] = res

    def _update_tail_output_tensor(self, node_info, i, new_arr, beg_idx):
        outputs = node_info['outputs']
        res = {}
        for t_id, val in outputs.items():
            pre_idx = i - 1
            if pre_idx < 0:
                pre_elem = self.seqs[beg_idx+pre_idx]
            else:
                pre_elem = new_arr[pre_idx]
            new_t_id = t_id
            if t_id.isnumeric():
                diff_num = self.out_diff[i]
                pre_t_id = list(pre_elem[1]['outputs'].keys())[0]
                pre_t_id = int(pre_t_id)
                new_t_id = str(pre_t_id + diff_num)
            res[new_t_id] = val
        node_info['outputs'] = res

    def save_json(self, out_file):
        _dict = OrderedDict()
        for name, node_info in self.seqs:
            _dict[name] = node_info
        with open(out_file, 'w') as out_:
            json.dump(_dict, out_)


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


def inspect_tail_input_info(worker, layers):
    info_arr = worker.get_tail_input_info()
    print("following is tail info")
    print_arr(info_arr)


def inspect_tail_output_info(worker, layers):
    info_arr = worker.get_tail_out_info()
    print("following is tail out info")
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
    # for l_num in range(1, layers):
    #     print("***" * 30)
    #     print(f"info for layer is {l_num} ")
    #     l_info_arr = t_worker.get_layer_info(l_num)
    #     for l_info in l_info_arr:
    #         print(l_info)
    # t_worker.print_special()
    # diff_arr_var, var_sum = inspect_out_diff_var(t_worker, layers)
    # print_arr(diff_arr_var)
    # print(f"diff var sum is: {var_sum}")
    # inspect_out_diff(t_worker, 1)
    # in_arr_var, in_var_sum = inspect_in_diff_var(t_worker, layers)
    # print_arr(in_arr_var)
    # print(f"in diff var sum is: {in_var_sum}")
    # print("now compare generated and non_generated")
    # generated_layer = t_worker.generate_a_layer(layers - 2, need_norm=True)
    # org_layer = t_worker.get_a_layer(layers - 1, need_norm=True)
    # print_arr(zip(org_layer, generated_layer))
    # inspect_tail_input_info(t_worker, layers)
    # inspect_tail_output_info(t_worker, layers)
    # org_tail = t_worker.get_tail_layer(need_norm=True)
    # gen_tail = t_worker.generate_tail(True)
    # print_arr(zip(org_tail, gen_tail))
    t_worker.increase_n_layers(args.add_num)
    t_worker.save_json(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-r', '--ref', type=str)
    parser.add_argument('-l1', '--f_ls', type=int, help="the first layers")
    parser.add_argument('-l2', '--f_ls2', type=int, help="the second layers")
    parser.add_argument('-n', '--num', type=int, default=48)
    parser.add_argument('-o', '--output', type=str, help='save the result')
    parser.add_argument('--add_num', type=int, default=48)
    args = parser.parse_args()
    test(args)

