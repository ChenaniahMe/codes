from transformers import MarianMTModel, MarianTokenizer
#English
src_text = [
   'Baidu translation app is an essential tool for English learning, which integrates dictionary, translation, bilingual content, oral assessment, AI word recitation and English short video.',
]
#Chinese
# src_text = [
#     '我是中国人',
# ]
# model_name = './opus-mt-en-roa'
#Chinese->越语
# model_name = './opus-mt-zh-vi'
#Chinese->英语
# model_name = './opus-mt-zh-en'
#英语->日语
# model_name = './opus-mt-en-jap'
#英语->汉语
model_name = './opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)