# coding=utf8

'''
python merge_tokenizers.py
'''

from transformers.models.llama.tokenization_llama import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("merged_tokenizer_hf")
s="我是中国人，我们爱自己的祖国。 我们的祖国有着5千年的悠久的历史，沉淀这浓厚的文化和底蕴。"
res = tokenizer.tokenize(s)
print(res)
