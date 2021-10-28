from fairseq.models.transformer import TransformerModel

from transformers import MarianMTModel, MarianTokenizer
src_text = [
   'I am chinese',
]

tgt_text = src_text

en2ko = TransformerModel.from_pretrained(
    'en-ko',
    checkpoint_file='checkpoint89.pt',
    data_name_or_path='en-ko',
    bpe='sentencepiece',
    sentencepiece_model='en-ko/gutenberg.model'
)
print(en2ko.translate(tgt_text))

