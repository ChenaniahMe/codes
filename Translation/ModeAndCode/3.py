from transformers import MarianMTModel, MarianTokenizer
src_text = [
   'To address the limitations mentioned above, we propose a novel two-stage method, named Graph neural networks with Intra- and Inter-session information for Session-based Recommendation (GIISR), as shown in Fig. 1. To the best of our knowledge, our method is the first work that exploits both the intra- and inter-session information using graph neural networks for more accurate recommendations. GIISR mainly includes two stages: In the first stage, we construct two Graph Convolutional Networks (GCNs) to learn the relation of the intra-session items.Furthermore, in the second stage, we construct an inter-session graph, which also contains the intra-session relation fused from the first stage to aggregate the information between different sessions. Moreover, we employ the soft-attention mechanism and hybrid embedding to consider global preference and current interest. Finally, the probability of the next click item can be predicted.',
]
model_name = './opus-mt-en-zh'
tokenizer = MarianTokenizer.from_pretrained(model_name)
# print(tokenizer.supported_language_codes)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
print(tgt_text)