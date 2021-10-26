import os
from gensim.models import word2vec
import jieba
# 读取训练数据
pos_file_list = os.listdir('data/pos')
neg_file_list = os.listdir('data/neg')
pos_file_list = [f'data/pos/{x}' for x in pos_file_list]
neg_file_list = [f'data/neg/{x}' for x in neg_file_list]
pos_neg_file_list = pos_file_list + neg_file_list
# 分词
for file in pos_neg_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().strip()  # 去读文件，并去除空格
        text_cut = jieba.cut(text)  # 使用jieba进行分词

        result = ' '.join(text_cut)  # 把分词结果用空格组成字符串

        with open('test.txt', 'a', encoding='utf-8') as fw:
            fw.write(result)  # 把分好的词写入到新的文件里面
            pass
        pass
    pass