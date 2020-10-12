# coding=utf8

import re

han_pat = re.compile(r"[\u4e00-\u9fa5]")
sen_split_pat = re.compile(r"([\n]+)")

speech_begin = "“"
speech_end = "”"
sen_seps = {"。", "？", "！"}


def is_hit_suspect(text, sus_words):
    if not _is_contain(text, sus_words):
        return False
    org_len = len(text)
    res_len = _get_res_han_len(text, sus_words)
    if res_len < 5:
        return True
    if (8 * res_len) < (org_len * 5):
        return True
    return False


def _is_contain(text, words):
    for word in words:
        if word in text:
            return True
    return False


def _get_res_han_len(text, words):
    for word in words:
        text = text.replace(word, '')
    return get_han_len(text)


def get_han_len(text):
    if text is None:
        return 0
    iter = re.finditer(han_pat, text)
    count = 0
    for _ in iter:
        count += 1
    return count


def cut_sentence(sen):
    res = []
    is_con_begin = False
    tmp_sen = ""
    for i, ch in enumerate(sen):
        tmp_sen += ch
        if ch == speech_begin:
            is_con_begin = True
        if ch == speech_end:
            is_con_begin = False
            if i > 0 and sen[i-1] in sen_seps:
                if len(tmp_sen) > 1:
                    res.append(tmp_sen)
                tmp_sen = ""
        if ch in sen_seps:
            if not is_con_begin:
                if len(tmp_sen) > 1:
                    res.append(tmp_sen)
                tmp_sen = ""
    if len(tmp_sen) > 1:
        res.append(tmp_sen)
    return res


def remove_repeat(text):
    pre_arr = _get_pre_arr(text)
    del_idxs = _get_del_ids(pre_arr)
    res_text = []
    for i, ch in enumerate(text):
        if i not in del_idxs:
            res_text.append(ch)
    return "".join(res_text)


def _get_pre_arr(text):
    res = []
    pre_dict = dict()
    for i, ch in enumerate(text):
        pre_pos = pre_dict.get(ch, -1)
        res.append(pre_pos)
        pre_dict[ch] = i
    return res


def _get_del_ids(pre_arr):
    res = set()
    tmp_del = []
    tmp_pre = -1
    for i, idx in enumerate(pre_arr):
        if idx >= 0:
            if tmp_pre == -1:
                tmp_pre = i
                tmp_del.append(idx)
            else:
                # 第一步判断是否是同步
                if not _is_consective(idx, tmp_del):
                    # 不同步
                    tmp_del = []
                    tmp_pre = i
                tmp_del.append(idx)

            if idx == tmp_pre - 1:
                res.update(tmp_del)
                tmp_pre = -1
                tmp_del = []
        else:
            tmp_del = []
            tmp_pre = -1
    return res


def _is_consective(elem, _arr):
    if len(_arr) < 1:
        return True
    if _arr[-1] + 1 == elem:
        return True
    return False


sub_en_sen_pat = re.compile(r'([,，\n.]+)')


def split_sub_sen(sen):
    if sen is None:
        return None
    sub_arr = re.split(sub_en_sen_pat, sen)
    if len(sub_arr) % 2 != 0:
        sub_arr.append("")
    a1 = sub_arr[::2]
    a2 = sub_arr[1::2]
    x = zip(a1, a2)
    res = ["".join(elem) for elem in x]
    return res


upper_pat = re.compile(r'^[A-Z.]+$')

def preprocess_eng(str_):
    """
    本文将英文的句子处了全部是是大写的外，都改写为小写；
    """
    tmp_arr = []
    words = str_.split()
    for word in words:
        if is_all_upper(word):
            tmp_arr.append(word)
        else:
            tmp_arr.append(word.lower())
    return " ".join(tmp_arr)


def is_all_upper(word):
    if re.match(upper_pat, word):
        return True
    return False


if __name__ == "__main__":
    test_str = "“凯尔啊、算了吧！！那男人根本不行！除了动动腰部之外，别的什么都不会。如果他有你这么棒的技巧，我才要考虑让他当我的男朋友。” 他是我的好朋友。 他很牛？"
    sens = cut_sentence(test_str)
    for sen in sens:
        print(sen)
