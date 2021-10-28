# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:45:51 2021

@author: Chenaniah
"""
class ZhWei(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = ['I am chinese.',]
        model_name = './opus-mt-wei-zh'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        # print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        #print(tgt_text)
        self.dicts_zhwei={}
        self.dicts_weizh={}
        arr_zh=[]
        with open(u"trainmini.zh", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zh.append(line)
                if not line:
                    break
        arr_wei=[]
        with open(u"trainmini.wei", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_wei.append(line)
                if not line:
                    break
#        print(len(arr_zh))
        assert len(arr_zh)==len(arr_wei),"length is not same"
        
        for i in range(len(arr_zh)):
            self.dicts_zhwei[arr_zh[i]] = arr_wei[i]
            self.dicts_weizh[arr_wei[i]] = arr_zh[i]
            
#        print(self.dicts)
            
    def zh_wei(self,zh):
        arr_zh=list(zh)
        res= ""
        for i in range(len(arr_zh)):
            res+= self.dicts_zhwei.get(arr_zh[i],"")
            res += " "
        return res
    
    def wei_zh(self,wei):
        arr_wei= wei.split()
        res= ""
        for i in range(len(arr_wei)):
            res+=self.dicts_weizh.get(arr_wei[i],"")
            res += " "
        return res

zhwei = ZhWei()
res = zhwei.wei_zh("مەن ھەئە بىرى A. in دۆلەت كىشىلەر")
print(res)
        
