from fairseq.models.transformer import TransformerModel

zh2ko = TransformerModel.from_pretrained(
    'zh-ko',
    checkpoint_file='checkpoint86.pt',
    data_name_or_path='zh-ko',
    bpe='sentencepiece',
    sentencepiece_model='zh-ko/wiki.ko.model'
)

en2ko = TransformerModel.from_pretrained(
    'en-ko',
    checkpoint_file='checkpoint89.pt',
    data_name_or_path='en-ko',
    bpe='sentencepiece',
    sentencepiece_model='en-ko/gutenberg.model'
)
print(en2ko.translate(['I am chinese']))

