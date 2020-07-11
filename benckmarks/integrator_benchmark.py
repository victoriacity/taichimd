import os, sys, json
from timeit import default_timer as timer
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from taichimd.examples import ljsystem, chain
from taichimd.integrator import *


'''
Benchmarks various integrators.
'''
def benchmark(system, nframe, irender, filepath=None):
    energy = []
    momentum = []
    start = timer()
    for i in range(nframe):
        system.step()
        energy.append(system.energy())
        vcm = np.sum(system.velocity.to_numpy(), axis=0)
        momentum.append(np.sum(vcm ** 2) ** 0.5)
        if filepath is not None and i % irender == 0:
            system.gui.show(os.path.join(filepath, "%i.png" % i))
    end = timer()
    return energy, momentum, end - start


if __name__ == "__main__":
    time = 3
    dts = [0.005, 0.01, 0.02, 0.04]
    integrators = [
        ForwardEulerIntegrator,
        VerletIntegrator,
        BackwardJacobiIntegrator,
        MidpointJacobiIntegrator,
        MidpointFixedIntegrator
    ]
    name = ['lj', 'chain']
    results = {}
    for i in range(2):
        if i == 0:
            continue
        results[name[i]] = {}
        for integrator in integrators:
            results[name[i]][integrator.__name__] = {}
            for dt in dts:
                ti.init(ti.cuda)
                if i == 0:
                    md = ljsystem(4096, 0.1, 1, dt, integrator, gui=False)
                    time = 3
                else:
                    dt /= 10
                    md = chain(10, 573, dt, integrator, gui=False)
                    time = 0.3
                results[name[i]][integrator.__name__][dt] = {}
                nframe = int(time / dt)
                energy, momentum, walltime = benchmark(md, nframe, 1)
                print("dt = %f, perf: %.3f s per 1 time unit" % (dt, walltime / time))
                results[name[i]][integrator.__name__][dt]['energy'] = energy
                results[name[i]][integrator.__name__][dt]['momentum'] = momentum
                results[name[i]][integrator.__name__][dt]['perf'] = walltime / time
            with open("result.json", 'w+') as f:
                json.dump(results, f)
    
