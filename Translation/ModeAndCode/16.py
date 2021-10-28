# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:45:51 2021

@author: Chenaniah
"""
class ZhZang(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = [ 'I am chinese.',]
        model_name = './opus-mt-zh-zang'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        # print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        #print(tgt_text)


        self.dicts_zhzang={}
        self.dicts_zangzh={}
        arr_zh=[]
        with open(u"trainmini.zh", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zh.append(line)
                if not line:
                    break
        arr_zang=[]
        with open(u"trainmini.zang", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zang.append(line)
                if not line:
                    break
#        print(len(arr_zh))
        assert len(arr_zh)==len(arr_zang),"length is not same"
        
        for i in range(len(arr_zh)):
            self.dicts_zhzang[arr_zh[i]] = arr_zang[i]
            self.dicts_zangzh[arr_zang[i].replace(" ","")] = arr_zh[i]
#        print(self.dicts)
            
    def zh_zang(self,zh):
        arr_zh=list(zh)
        res= ""
        for i in range(len(arr_zh)):
            res+= self.dicts_zhzang.get(arr_zh[i],"")
        return res
    
    def zang_zh(self,zang):
        arr_zang= zang.split()
        res= ""
        for i in range(len(arr_zang)):
            res+=self.dicts_zangzh.get(arr_zang[i],"")
            res += " "
        return res

zhzang = ZhZang()
res = zhzang.zh_zang("我是一个中国人")
print(res)
