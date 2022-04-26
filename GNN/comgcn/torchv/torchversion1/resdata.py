import numpy as np
import config
data = open(config.resfile)
data = data.readlines()
res = [float(strs[:-2]) for strs in data]

f = open(config.resmfile, "a+")
f.write("对所有结果值求均值的结果："+str(np.mean(res)))
f.write("\n")
