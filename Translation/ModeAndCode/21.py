from fairseq.models.transformer import TransformerModel

from transformers import MarianMTModel, MarianTokenizer
src_text = [
   '私は中国人',
]
model_name = './opus-mt-ja-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

en2ko = TransformerModel.from_pretrained(
    'en-ko',
    checkpoint_file='checkpoint89.pt',
    data_name_or_path='en-ko',
    bpe='sentencepiece',
    sentencepiece_model='en-ko/gutenberg.model'
)
print(en2ko.translate(tgt_text))

