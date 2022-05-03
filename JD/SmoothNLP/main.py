from phrase.ngram_utils import sentence_split_by_punc,remove_irregular_chars,get_scores
from datetime import datetime
import io
#from smoothnlp import config
import types


def chunk_generator_adapter(obj, chunk_size):
    '''
    返回chunk_size大小的语料preprocessing后的一个list
    :param obj:
    :param chunk_size:
    :return:
    '''
    tstart = datetime.now()
    while True:
        import sqlalchemy
        if isinstance(obj,sqlalchemy.engine.result.ResultProxy):  # 输入database connection object = conn.execute(query)
            obj_adapter = list(obj.fetchmany(chunk_size))
        elif isinstance(obj, _io.TextIOWrapper):     # 输入object = open(file_name, 'r', encoding='utf-8')
            obj_adapter = obj.readlines(chunk_size)  # list of str
        elif isinstance(obj,types.GeneratorType):
            obj_adapter = list(next(obj,''))
        else:
            raise ValueError('Input not supported!')
        if obj_adapter != None and obj_adapter != []:
            corpus_chunk = [remove_irregular_chars(sent) for r in obj_adapter for sent in
                                sentence_split_by_punc(str(r))]
            yield corpus_chunk  
        else:
            tend = datetime.now()
            sec_used = (tend-tstart).seconds
            #config.logger.info('~~~ Time used for data processing: {} seconds'.format(sec_used))
            break


def extract_phrase(corpus,
                   top_k: float = 200,
                   chunk_size: int = 1000000,
                   min_n:int = 2,
                   max_n:int=4,
                   min_freq:int = 5):
    '''
    取前k个new words或前k%的new words
    :param corpus: 输入数据
    :param top_k: 返回词的数量
    :param chunk_size: 每次进行统计的数量
    :param max_n:n-gram 中，n的最大值
    :param min_freq: 指进行n-grams的时候，grams出现的频率，大于这个频率的值进行保留
    :return:
    '''
    if isinstance(corpus,str):
        corpus_splits = [remove_irregular_chars(sent) for sent in sentence_split_by_punc(corpus)]
    elif isinstance(corpus,list):
        corpus_splits = [remove_irregular_chars(sent) for news in corpus for sent in
                                sentence_split_by_punc(str(news)) if len(remove_irregular_chars(sent)) != 0]
    else:
        corpus_splits = chunk_generator_adapter(corpus, chunk_size)
    word_info_scores = get_scores(corpus_splits,min_n,max_n,chunk_size,min_freq)
    '''test_a = word_info_scores.items() 按照最后一项从大到小排列， 排序方式有好几个只
    -1是融合了熵信息和互信息的
    -2是左熵和右熵取了最小值
    '''
    new_words = [item[0] for item in sorted(word_info_scores.items(),key=lambda item:item[1][-1],reverse = True)]
    if top_k > 1:              #输出前k个词
        return new_words[:top_k]
    elif top_k < 1:            #输出前k%的词
        return new_words[:int(top_k*len(new_words))]
# f = open("/Users/zhutianwen/Desktop/Code/sentence.txt")
# f = open("./sentence.txt")
# f = open("./data1000w.txt")
# f = open("./data3000w.txt")
f = open("./data1000.txt",encoding='utf-8')
#f = open("./res.txt")
str1=""
count = 0
while True:
    lines = f.readline()
    if not lines:
        break
    #去除包含的量词和英文字
    lines = "".join(list(filter(lambda ch: ch not in ' \t1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', lines) ))
    str1 += lines
    count += 1
    print(count)
# str1 = ["SmoothNLP在V0.3版本中正式推出知识抽取功能",
#                             "SmoothNLP专注于可解释的NLP技术",
#                             "SmoothNLP支持Python与Java",
#                             "SmoothNLP将帮助工业界与学术界更加高效的构建知识图谱",
#                             "SmoothNLP是上海文磨网络科技公司的开源项目",
#                             "SmoothNLP在V0.4版本中推出对图谱节点的分类功能",
#                             "KGExplore是SmoothNLP的一个子项目"]
print(extract_phrase(str1,top_k=1000,max_n=3))
