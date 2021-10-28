# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:45:51 2021

@author: Chenaniah
"""
class ZhMeng(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = [ 'I am chinese.',]
        model_name = './opus-mt-zh-meng'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        # print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        #print(tgt_text)

        self.dicts_zhmeng={}
        self.dicts_mengzh={}
        arr_zh=[]
        with open(u"trainmini.zh", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zh.append(line)
                if not line:
                    break
        arr_meng=[]
        with open(u"trainmini.meng", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_meng.append(line)
                if not line:
                    break
#        print(len(arr_zh))
        assert len(arr_zh)==len(arr_meng),"length is not same"
        
        for i in range(len(arr_zh)):
            self.dicts_zhmeng[arr_zh[i]] = arr_meng[i]
            self.dicts_mengzh[arr_meng[i]] = arr_zh[i]
            
#        print(self.dicts)
            
    def zh_meng(self,zh):
        arr_zh=list(zh)
        res= ""
        for i in range(len(arr_zh)):
            res+= self.dicts_zhmeng.get(arr_zh[i],"")
            res += " "
        return res
    
    def meng_zh(self,meng):
        arr_meng= meng.split()
        res= ""
        for i in range(len(arr_meng)):
            res+=self.dicts_mengzh.get(arr_meng[i],"")
            res += " "
        return res

zhmeng = ZhMeng()
res = zhmeng.zh_meng("我是一个中国人")
print(res)
        
