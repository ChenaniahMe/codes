import numpy as np
import re,jieba
from itertools import chain
from gensim.models import word2vec
#打开文件
sentences_list = []
file_path='吴邦国重申：中国坚持和平发展道路不会因国力地位变化而改变_党建_共产党员网.txt'
fp = open(file_path,'r',encoding="utf8")
for line in fp.readlines():
        if line.strip():
            # 把元素按照[。！；？]进行分隔，得到句子。
            line_split = re.split(r'[。！；？]',line.strip())
            # [。！；？]这些符号也会划分出来，把它们去掉。
            line_split = [line.strip() for line in line_split if line.strip() not in ['。','！','？','；'] and len(line.strip())>1]
            sentences_list.append(line_split)
sentences_list = list(chain.from_iterable(sentences_list))
print("前10个句子为：\n")
print(sentences_list[:10])
print("句子总数：", len(sentences_list))

#加载停用词
stoplist= [word.strip() for word in open('stopwords.txt',encoding='GBK').readlines()]
# print(stoplist)

# 对句子进行分词
def seg_depart(sentence):
    # 去掉非汉字字符
    sentence = re.sub(r'[^\u4e00-\u9fa5]+','',sentence)
    sentence_depart = jieba.cut(sentence.strip())
    word_list = []
    for word in sentence_depart:
        if word not in stoplist:
            word_list.append(word)
    # 如果句子整个被过滤掉了，如：'02-2717:56'被过滤，那就返回[],保持句子的数量不变
    return word_list

sentence_word_list = []
for sentence in sentences_list:
    line_seg = seg_depart(sentence)
    sentence_word_list.append(line_seg)
print("一共有",len(sentences_list),'个句子。\n')
print("前10个句子分词后的结果为：\n",sentence_word_list[:10])

# 保证处理后句子的数量不变，我们后面才好根据textrank值取出未处理之前的句子作为摘要。
if len(sentences_list) == len(sentence_word_list):
    print("\n数据预处理后句子的数量不变！")

#求句子最大长度
maxLen=0
for sentence in sentences_list:
    length=0
    for wd in sentence:
        length=length+1
    if (length>maxLen):maxLen=length

#fit_on_texts函数可以将输入的文本中的每个词编号，
#编号是根据词频的，词频越大，编号越小
from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentence_word_list)
vocab = tokenizer.word_index  # 得到每个词的编号
print(vocab)

# 设置词语向量维度
num_featrues = 300
# 保证被考虑词语的最低频度，对于小语料，设置为1才可能能输出所有的词，因为有的词可能在每个句子中只出现一次
min_word_count = 1
# 设置并行化训练使用CPU计算核心数量
num_workers =4
# 设置词语上下文窗口大小
context = 5
#开始训练
model = word2vec.Word2Vec(sentence_word_list, workers=num_workers, size=num_featrues, min_count=min_word_count, window=context)
# 如果你不打算进一步训练模型，调用init_sims将使得模型的存储更加高效
model.init_sims(replace=True)

'''
# 如果有需要的话，可以输入一个路径，保存训练好的模型
model.save("w2vModel1")
print(model)
#加载模型
model = word2vec.Word2Vec.load("w2vModel1")
'''
word_embeddings = {}
count=0
for word, i in vocab.items():
    try:
        # model.wv[word]存的就是这个word的词向量
        word_embeddings[word] =model.wv[word]
    except KeyError:
        continue
print('输出了：',count,'个词')

sentence_vectors = []
for line in sentence_word_list:
    if len(line)!=0:
        # 如果句子中的词语不在字典中，那就把embedding设为300维元素为0的向量。
        # 得到句子中全部词的词向量后，求平均值，得到句子的向量表示
        #TypeError: type numpy.ndarray doesn't define __round__ method,将round改为np.round
        v = np.round(sum(word_embeddings.get(word, np.zeros((300,))) for word in line)/(len(line)))
    else:
        # 如果句子为[]，那么就向量表示为300维元素为0个向量。
        v = np.zeros((300,))
    sentence_vectors.append(v)

#计算句子之间的余弦相似度，构成相似度矩阵
sim_mat = np.zeros([len(sentences_list), len(sentences_list)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences_list)):
  for j in range(len(sentences_list)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
print("句子相似度矩阵的形状为：",sim_mat.shape)

#迭代得到句子的textrank值，排序并取出摘要"""
import networkx as nx

# 利用句子相似度矩阵构建图结构，句子为节点，句子相似度为转移概率
nx_graph = nx.from_numpy_array(sim_mat)

# 得到所有句子的textrank值
scores = nx.pagerank(nx_graph)

# 根据textrank值对未处理的句子进行排序
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences_list)), reverse=True)

# 取出得分最高的前3个句子作为摘要
sn = 3
for i in range(sn):
    print("第"+str(i+1)+"条摘要：\n\n",ranked_sentences[i][1],'\n')