# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:40:57 2021

@author: Chenaniah
"""


import numpy as np
np.random.seed(2)  
import data
from sko.PSO import PSO
import pandas as pd

def demo_func(x):
#    x = np.array(x)
#    print(x,x[0:3],x[3:6])
#    print(np.array(x))
#    print("xxxxxxxxxxxx",x)
    for i in range(len(x)):
        x[i] = np.around(x[i])
    
    objectoa = sum((data.cit*x[:81].reshape(data.cit.shape)*data.t_ij).flatten())
    objectob = sum(x[:81]*data.crj-x[:81]*data.cri)
#    objectoc = sum(x[81:90] * data.cis)
    
    objectta = max(((x[:81]*data.tik).reshape(data.cit.shape)+data.t_ij).flatten())
    
#    print(x*data.crj)
#    print(x*data.cri)
#    print("three",three)
    return 0.75*(objectoa+objectob)+0.25*objectta
#    return 0.5*(objectoa)+0.5*objectta

constraint_ueq = (
#        [x[i] for i in range(len(x)) if i%3==0][:9]
#       [arr[i] for i in range(len(arr)) if i%3==0][9:18]
#[arr[i+2] for i in range(len(arr)) if i%3==0]
    lambda x: data.consfjk.flatten()[0]-sum([x[i] for i in range(len(x)) if i%3==0][:9]),
    lambda x: data.consfjk.flatten()[1]-sum([x[i+1] for i in range(len(x)) if i%3==0][:9]),
    lambda x: data.consfjk.flatten()[2]-sum([x[i+2] for i in range(len(x)) if i%3==0][:9]),
    lambda x: data.consfjk.flatten()[3]-sum([x[i] for i in range(len(x)) if i%3==0][9:18]),
    lambda x: data.consfjk.flatten()[4]-sum([x[i+1] for i in range(len(x)) if i%3==0][9:18]),
    lambda x: data.consfjk.flatten()[5]-sum([x[i+2] for i in range(len(x)) if i%3==0][9:18]),
    lambda x: data.consfjk.flatten()[6]-sum([x[i] for i in range(len(x)) if i%3==0][18:27]),
    lambda x: data.consfjk.flatten()[7]-sum([x[i+1] for i in range(len(x)) if i%3==0][18:27]),
    lambda x: data.consfjk.flatten()[8]-sum([x[i+2] for i in range(len(x)) if i%3==0][18:27]),
    
    lambda x: sum(x[0:3])-data.Q.flatten()[0],
    lambda x: sum(x[3:6])-data.Q.flatten()[1],
    lambda x: sum(x[6:9])-data.Q.flatten()[2],
    lambda x: sum(x[9:12])-data.Q.flatten()[3],
    lambda x: sum(x[12:15])-data.Q.flatten()[4],
    lambda x: sum(x[15:18])-data.Q.flatten()[5],
    lambda x: sum(x[18:21])-data.Q.flatten()[6],
    lambda x: sum(x[21:24])-data.Q.flatten()[7],
    lambda x: sum(x[24:27])-data.Q.flatten()[8],
    lambda x: sum(x[27:30])-data.Q.flatten()[9],
    lambda x: sum(x[30:33])-data.Q.flatten()[10],
    lambda x: sum(x[33:36])-data.Q.flatten()[11],
    lambda x: sum(x[36:39])-data.Q.flatten()[12],
    lambda x: sum(x[39:42])-data.Q.flatten()[13],
    lambda x: sum(x[42:45])-data.Q.flatten()[14],
    lambda x: sum(x[45:48])-data.Q.flatten()[15],
    lambda x: sum(x[48:51])-data.Q.flatten()[16],
    lambda x: sum(x[51:54])-data.Q.flatten()[17],
    lambda x: sum(x[54:57])-data.Q.flatten()[18],
    lambda x: sum(x[57:60])-data.Q.flatten()[19],
    lambda x: sum(x[60:63])-data.Q.flatten()[20],
    lambda x: sum(x[63:66])-data.Q.flatten()[21],
    lambda x: sum(x[66:69])-data.Q.flatten()[22],
    lambda x: sum(x[69:72])-data.Q.flatten()[23],
    lambda x: sum(x[72:75])-data.Q.flatten()[24],
    lambda x: sum(x[75:78])-data.Q.flatten()[25],
    lambda x: sum(x[78:81])-data.Q.flatten()[26],
)

max_iter = 1000
#max_iter = 1
#pop增大对效果有好的影响
pso = PSO(func=demo_func, n_dim=81, pop=40, max_iter=max_iter, lb=data.xlb.flatten(), ub=data.xub.flatten()
          , constraint_ueq=constraint_ueq,verbose=True)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

#pso.record_value["X"]
res = []
for i in range(len(pso.record_value["Y"])):
    res.append(np.mean(pso.record_value["Y"][i]))
    
#print(pso.gbest_x.reshape(3,9,3))

#print(np.array([pso.gbest_x.reshape(9,9)[:3].T,pso.gbest_x.reshape(9,9)[3:6].T,pso.gbest_x.reshape(9,9)[6:9].T]))
print(data.consfjk)
print(data.Q)
for i in range(len(pso.gbest_x)):
    pso.gbest_x[i] = np.around(pso.gbest_x[i])
#print(pso.gbest_x.reshape(3,9,3).astype(np.int16))
x = pso.gbest_x[:81]
x = x.reshape(3,9,3).astype(np.int16)
y = pso.gbest_x[81:]
print(x)
print(y)
pd.DataFrame(res).plot()
#x = pso.gbest_x.reshape(3,9,3)
#x = x.flatten()
#print(data.consfjk.flatten()[4],[x[i+1] for i in range(len(x)) if i%3==0][9:18])