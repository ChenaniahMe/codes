from transformers import MarianMTModel, MarianTokenizer
src_text = [
   '저 는 중국 사람 입 니 ',
]
model_name = './opus-mt-ko-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)