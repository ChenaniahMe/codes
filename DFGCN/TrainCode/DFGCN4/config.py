import numpy as np
from collections import Counter
# adj="adj.npz"
numClass=[]
# file_name = "Data"
file_name = "SmallFlevoland"
# file_features = "/data/ztw/PolSARPubPaperOne/Experiment/DealData/" + file_name + "/python/label.npy"
#file_name = "T3-Nonfiltered-SmallFarm"
# file_name = "ReduceSmallFlevoland"
file_features = "/data/ztw/PolSARPubPaperOne/Experiment/DealData/" + file_name + "/python/label.npy"
# file_features = "/data/ztw/PolSARPubPaperOne/Experiment/DealData/" + file_name + "/python_reduce/label.npy"

label=np.load(file_features)
# print("label",label)
dict=Counter(label)
each_class=dict.keys()
for i in range(int(max(list(each_class)))):
    numClass.append(dict[i+1])
print(numClass)
