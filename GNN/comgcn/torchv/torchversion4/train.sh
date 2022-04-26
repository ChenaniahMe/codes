#!/bin/bash
rm -rf ./resdata/res.txt
rm -rf ./resdata/resm.txt
for i in {1..20}; do
    echo "正在进行第多少次测试$i"
    python train.py
done
python resdata.py
