import numpy as np
from sko.PSO import PSO


def demo_func(x):
    return sum(x**2)


constraint_ueq = (
    lambda x: x[0]-x[1]
    ,
)


max_iter = 50
pso = PSO(func=demo_func, n_dim=2, pop=40, max_iter=max_iter, lb=np.array([-2, -2]), ub=np.array([10000, 10000])
          , constraint_ueq=constraint_ueq)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

