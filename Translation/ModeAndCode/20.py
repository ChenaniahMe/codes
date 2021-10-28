# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:45:51 2021

@author: Chenaniah
"""
class ZhZhuang(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = [ 'I am chinese.',]
        model_name = './opus-mt-zh-zhuang'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        # print(tokenizer.supported_language_codes)
        model = MarianMTModel.from_pretrained(model_name)
        translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        #print(tgt_text)

        self.dicts_zhzhuang={}
        self.dicts_zhuangzh={}
        arr_zh=[]
        with open(u"trainmini.zh", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zh.append(line)
                if not line:
                    break
                
        arr_zhuang=[]
        with open(u"trainmini.zhuang", "r", encoding='utf-8') as f:
            while True:
                line = f.readline()
                line=line.replace("\n","")
                arr_zhuang.append(line)
                if not line:
                    break
#        print(len(arr_zh))
        assert len(arr_zh)==len(arr_zhuang),"length is not same"
        
        for i in range(len(arr_zh)):
            self.dicts_zhzhuang[arr_zh[i]] = arr_zhuang[i]
            self.dicts_zhuangzh[arr_zhuang[i]] = arr_zh[i]
            
#        print(self.dicts)
            
    def zh_zhuang(self,zh):
        arr_zh=list(zh)
        res= ""
        for i in range(len(arr_zh)):
            res+= self.dicts_zhzhuang.get(arr_zh[i],"")
            res += " "
        return res
    
    def zhuang_zh(self,zhuang):
        arr_zhuang= zhuang.split()
        res= ""
        for i in range(len(arr_zhuang)):
            res+=self.dicts_zhuangzh.get(arr_zhuang[i],"")
            res += " "
        return res

zhzhuang = ZhZhuang()
res = zhzhuang.zh_zhuang("我是一个中国人人")
print(res)
        
