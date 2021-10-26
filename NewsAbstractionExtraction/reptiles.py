import requests
from  lxml import html

etree=html.etree
url='http://news.12371.cn/2013/01/28/ARTI1359357184590944.shtml'
data=requests.get(url)
data.encoding='utf-8'
#print(data)
s=etree.HTML(data.text)
text1=s.xpath('//*[@id="font_area"]/p/text()')#得到的文本是一个列表，里面有6项，代表6个自然段
title=s.xpath('/html/head/title/text()')[0].strip()#[0]是取标题的第一项，trip()去掉首尾空格
print("爬取文本：\n","标题：",title,"\n正文：",text1)
text=""

# 将得到的文本写入文件
for i in range(0,len(text1)-1):
   text+=text1[i]
sentence_list=[]
print(text)
title=title+'.txt'
with open(title, 'w', encoding='utf-8') as f:
   f.writelines(text)