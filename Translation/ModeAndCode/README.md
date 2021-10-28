### 中文到韩文翻译

***

**需要安装以下库**

```python
pip install fairseq fastBPE sacremoses subword_nmt sentencepiece
```

 `fairseq` 最好安装在 `Linux ，MacOS` 系统，`windows` 会出现报错等情况 (一直没装上)

***

**文件目录**

- korean
  - zh-ko
    - checkpoint86.pt ：模型文件
    - dict.ko.txt ：韩文vocab文件
    - dict.zh.txt：中文vocab文件
    - wiki.ko.model：sentencepiece-model
  - demo.py
  - README.md

***

**在korean文件夹下执行中文到韩文的翻译**

命令行交互式翻译：

```python
fairseq-interactive zh-ko/ --path zh-ko/checkpoint86.pt --beam 5 --source-lang zh --target-lang ko --bpe sentencepiece --dataset-impl fasta --sentencepiece-model zh-ko/wiki.ko.model
```

第一个参数 `zh-ko` 表示 `zh-ko` 文件夹下，会自动从中加载 `dict.ko.txt`, `dict.zh.txt` 

`--path` 表示加载模型路径,  指定 `--bpe` 为`sentencepiece` (需要安装 `sentencepiece` 库， 并指定 `--sentencepiece-model` 参数)

其余参数：

https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive



或者执行下面的命令

```python
python demo.py
```

***

**训练新的基于 `Transformer` 的模型**

https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model

