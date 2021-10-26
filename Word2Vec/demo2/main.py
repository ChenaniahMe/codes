from gensim.models import word2vec
sentences = word2vec.LineSentence('test.txt')
model = word2vec.Word2Vec(sentences, hs=0, min_count=1, window=3, size=100)
for val in model.wv.similar_by_word("酒店", topn=10):
    print(val[0], val[1])