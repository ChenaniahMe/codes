import tornado.ioloop
import tornado.web
from tornado.options import define, options, parse_command_line
import os
from transformers import MarianMTModel, MarianTokenizer
from fairseq.models.transformer import TransformerModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 定义默认的端口
define('port', default=8100, type=int)

##越语-英语
#model_name = './opus-mt-vi-en'
#tokenizer_vien = MarianTokenizer.from_pretrained(model_name)
#model_vien = MarianMTModel.from_pretrained(model_name)
#
##英语-中文
model_name = './opus-mt-en-zh'
tokenizer_enzh = MarianTokenizer.from_pretrained(model_name)
model_enzh = MarianMTModel.from_pretrained(model_name)

##中文-越语
#model_name = './opus-mt-zh-vi'
#tokenizer_zhvi = MarianTokenizer.from_pretrained(model_name)
#model_zhvi = MarianMTModel.from_pretrained(model_name)
#
##中文-英文
model_name = './opus-mt-zh-en'
tokenizer_zhen = MarianTokenizer.from_pretrained(model_name)
model_zhen = MarianMTModel.from_pretrained(model_name)

##日语-英语
#model_name = './opus-mt-ja-en'
#tokenizer_jaen = MarianTokenizer.from_pretrained(model_name)
#model_jaen = MarianMTModel.from_pretrained(model_name)
#
##英语-日语
#model_name = './opus-mt-en-jap'
#tokenizer_enjap = MarianTokenizer.from_pretrained(model_name)
#model_enjap = MarianMTModel.from_pretrained(model_name)
#
##韩语-英语
#model_name = './opus-mt-ko-en'
#tokenizer_koen = MarianTokenizer.from_pretrained(model_name)
## print(tokenizer.supported_language_codes)
#model_koen = MarianMTModel.from_pretrained(model_name)
#
##英语-朝语
en2ko = TransformerModel.from_pretrained(
            'en-ko',
            checkpoint_file='checkpoint89.pt',
            data_name_or_path='en-ko',
            bpe='sentencepiece',
            sentencepiece_model='en-ko/gutenberg.model'
        )
#
##俄语-汉语
#model_name = './opus-mt-ru-en'
#tokenizer_ruen = MarianTokenizer.from_pretrained(model_name)
#model_ruen = MarianMTModel.from_pretrained(model_name)
#
##英语-俄语
#model_name = './opus-mt-en-ru'
#tokenizer_enru = MarianTokenizer.from_pretrained(model_name)
#model_enru = MarianMTModel.from_pretrained(model_name)
#
##阿语-英语
#model_name = './opus-mt-ar-en'
#tokenizer_aren = MarianTokenizer.from_pretrained(model_name)
#model_aren = MarianMTModel.from_pretrained(model_name)
#
##英语-阿语
#model_name = './opus-mt-en-ar'
#tokenizer_enar = MarianTokenizer.from_pretrained(model_name)
#model_enar = MarianMTModel.from_pretrained(model_name)
#
##日语-德语
#model_name = './opus-mt-ja-de'
#tokenizer_jade = MarianTokenizer.from_pretrained(model_name)
#model_jade = MarianMTModel.from_pretrained(model_name)
#
##德语-英语
#model_name = './opus-mt-de-en'
#tokenizer_deen = MarianTokenizer.from_pretrained(model_name)
#model_deen = MarianMTModel.from_pretrained(model_name)
#
##韩语-德语
#model_name = './opus-mt-ko-de'
#tokenizer_kode = MarianTokenizer.from_pretrained(model_name)
#model_kode = MarianMTModel.from_pretrained(model_name)
#
##英语-德语
#model_name = './opus-mt-en-de'
#tokenizer_ende = MarianTokenizer.from_pretrained(model_name)
#model_ende = MarianMTModel.from_pretrained(model_name)

print("模型加载完成")
class trans(tornado.web.RequestHandler):
#    def __init__(self,A,B):
#        self.res = "翻译结果为:"
    
    def get(self, *args, **kwargs):
        res = "翻译结果为:"
        tgt_text = ""
        # 获取请求URL中传递的name参数
        st=self.get_argument('st', 'CE')
        if st=="VC":
            res = "越语-汉语的翻译1，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_vien.generate(**tokenizer_vien.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_vien.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="CV":
            res = "汉语-越语的翻译2，" +res
            translated = model_zhvi.generate(**tokenizer_zhvi.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_zhvi.decode(t, skip_special_tokens=True) for t in translated]
        
        elif st=="EC":
            res = "英语-汉语的翻译3，" +res
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
        elif st=="CE":
            res = "汉语-英语的翻译4，" +res
            translated = model_zhen.generate(**tokenizer_zhen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_zhen.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="JC":
            res = "日语-汉语的翻译5，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_jaen.generate(**tokenizer_jaen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_jaen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="CJ":
            res = "汉语-日语的翻译6，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_zhen.generate(**tokenizer_zhen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_zhen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enjap.generate(**tokenizer_enjap.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enjap.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="KC":
            res = "韩语-汉语的翻译6，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_koen.generate(**tokenizer_koen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_koen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
                  
        elif st=="CK":
            res = "汉语-朝语的翻译8，" +res
            translated = model_zhen.generate(**tokenizer_zhen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_zhen.decode(t, skip_special_tokens=True) for t in translated]
            tgt_text = en2ko.translate(tgt_text)
            
        elif st=="RC":
            res = "俄语-汉语的翻译9，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_ruen.generate(**tokenizer_ruen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_ruen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="CR":
            res = "汉语-俄语的翻译10，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_zhen.generate(**tokenizer_zhen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_zhen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enru.generate(**tokenizer_enru.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enru.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="AC":
            res = "阿语-汉语的翻译11，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_aren.generate(**tokenizer_aren.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_aren.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enzh.generate(**tokenizer_enzh.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enzh.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="CA":
            res = "汉语-阿语的翻译12，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_zhen.generate(**tokenizer_zhen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_zhen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enar.generate(**tokenizer_enar.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enar.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="WC":
            res = "维语-汉语的翻译13，" +res
            zhwei = ZhWei()
            tgt_text = zhwei.wei_zh(src_text)
            
        elif st=="CW":
            res = "汉语-维语的翻译14，" +res
            zhwei = ZhWei()
            tgt_text = zhwei.zh_wei(src_text)
            
        elif st=="ZC":
            res = "藏语-汉语的翻译15，" +res
            zhzang = ZhZang()
            tgt_text = zhzang.zang_zh(src_text)
            
        elif st=="CZ":
            res = "汉语-藏语的翻译16，" +res
            zhzang = ZhZang()
            tgt_text = zhzang.zh_zang(src_text)
            
        elif st=="MC":
            res = "蒙语-汉语的翻译17，" +res
            zhmeng = ZhMeng()
            tgt_text = zhmeng.meng_zh(src_text)
            
        elif st=="CM":
            res = "汉语-蒙语的翻译18，" +res
            zhmeng = ZhMeng()
            tgt_text = zhmeng.zh_meng(src_text)
            
        elif st=="ZHC":
            res = "壮语-汉语的翻译19，" +res
            zhzhuang = ZhZhuang()
            tgt_text = zhzhuang.zhuang_zh(src_text)
            
        elif st=="CZH":
            res = "汉语-壮语的翻译20，" +res
            zhzhuang = ZhZhuang()
            tgt_text = zhzhuang.zh_zhuang(src_text)
            
        elif st=="JK":
            res = "日语-韩语的翻译21，" +res
            translated = model_jaen.generate(**tokenizer_jaen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_jaen.decode(t, skip_special_tokens=True) for t in translated]
            tgt_text = en2ko.translate(tgt_text)
            
        elif st=="KJ":
            res = "韩语-日语的翻译22，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_koen.generate(**tokenizer_koen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_koen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enjap.generate(**tokenizer_enjap.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enjap.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="JE":
            res = "日语-英语的翻译23，" +res
            translated = model_jaen.generate(**tokenizer_jaen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_jaen.decode(t, skip_special_tokens=True) for t in translated]
                   
        elif st=="EZ":
            res = "英语-日语的翻译24，" +res
            translated = model_enjap.generate(**tokenizer_enjap.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_enjap.decode(t, skip_special_tokens=True) for t in translated]
             
        elif st=="JG":
            res = "日语-德语的翻译25，" +res
            translated = model_jade.generate(**tokenizer_jade.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_jade.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="GJ":
            res = "德语-日语的翻译26，" +res
            src_text=[self.get_argument('content', '')]
            translated = model_deen.generate(**tokenizer_deen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_deen.decode(t, skip_special_tokens=True) for t in translated]
            src_text=tgt_text
            translated = model_enjap.generate(**tokenizer_enjap.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            translated.cuda()
            tgt_text = [tokenizer_enjap.decode(t, skip_special_tokens=True) for t in translated]
        
        elif st=="KE":
            res = "韩语-英语的翻译27，" +res
            translated = model_koen.generate(**tokenizer_koen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_koen.decode(t, skip_special_tokens=True) for t in translated]
            
            
        elif st=="EK":
            res = "英语-韩语的翻译28，" +res
            tgt_text = en2ko.translate(tgt_text)
            
        elif st=="KG":
            res = "韩语-德语的翻译29，" +res
            translated = model_kode.generate(**tokenizer_kode.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_kode.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="GK":
            res = "德语-韩语的翻译30，" +res
            translated = model_deen.generate(**tokenizer_deen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_deen.decode(t, skip_special_tokens=True) for t in translated]
            tgt_text = en2ko.translate(tgt_text)
        elif st=="GE":
            res = "德语-英语的翻译31，" +res
            translated = model_deen.generate(**tokenizer_deen.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_deen.decode(t, skip_special_tokens=True) for t in translated]
            
        elif st=="EG":
            res = "英语-德语的翻译32，" +res
            translated = model_ende.generate(**tokenizer_ende.prepare_seq2seq_batch(src_text, return_tensors="pt"))
            tgt_text = [tokenizer_ende.decode(t, skip_special_tokens=True) for t in translated]
               
        res = res + tgt_text
        self.write("Hello, %s" % res)
        
class ZhWei(object):
    def __init__(self):
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
    
class ZhZang(object):
    def __init__(self):

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
    
class ZhMeng(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = [ 'I am chinese.',]
        model_name = './opus-mt-meng-zh'
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

class ZhZhuang(object):
    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        src_text = [ 'I am chinese.',]
        model_name = './opus-mt-zhuang-zh'
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

def make_app():
    return tornado.web.Application(handlers=[
        (r"/trans", trans),
    ])

if __name__ == "__main__":
    parse_command_line()
    app = make_app()
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()