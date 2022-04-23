import numpy as np
label=np.load("label.npy")
from collections import Counter
label=np.array(label)
dict=Counter(label)
dictA=[]
for i in range(1, len(dict)+1):
	dictA.append(dict[i])
print(dictA)
print(sum(np.array([90.73,90.57, 91.71, 85.67, 99.86, 89.34, 92.81,89.92, 89.05, 88.73, 90.52, 89.43, 89.22, 91.19, 90.73])*dictA)/sum(dictA))


