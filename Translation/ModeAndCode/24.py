from transformers import MarianMTModel, MarianTokenizer

src_text = [
    'I am Chinese',
]

model_name = './opus-mt-en-jap'
tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print("targe_two",tgt_text)
