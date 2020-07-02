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


if __name__ == "__main__":
    test_str = "“凯尔啊、算了吧！！那男人根本不行！除了动动腰部之外，别的什么都不会。如果他有你这么棒的技巧，我才要考虑让他当我的男朋友。” 他是我的好朋友。 他很牛？"
    sens = cut_sentence(test_str)
    for sen in sens:
        print(sen)
