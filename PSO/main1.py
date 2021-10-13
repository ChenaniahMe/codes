#https://github.com/nathanrooy/particle-swarm-optimization
from pso import pso_simple
from pso.cost_functions import sphere
import numpy as np

initial=[5,5]
bounds=[(5,10),(5,10)]
pso_simple.minimize(sphere, initial, bounds, num_particles=15, maxiter=30, verbose=True)


#储备过量点Si的单位运输成本


