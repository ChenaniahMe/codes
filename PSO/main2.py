import numpy as np
from sko.PSO import PSO


def demo_func(x):
    x1, x2 = x
    return x1**2 + x2**2


constraint_ueq = (
    lambda x: x[0]-x[1]
    ,
)


max_iter = 50
pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=[-2, -2], ub=[2, 2]
          , constraint_ueq=constraint_ueq)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

