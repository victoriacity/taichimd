import argparse
import taichi as ti
import numpy as np
from taichimd.system import MolecularDynamics
from taichimd.interaction import *
from taichimd.integrator import *
from taichimd.ui import CanvasShader

def ljsystem(n, rho, temp, dt, integrator, gui=True):
    boxlength = (n / rho) ** (1 / DIM)
    md = MolecularDynamics(n, boxlength, integrator, dt,
        temperature=temp,
        nonbond=LennardJones(1, 1, 2.5),
        shader=CanvasShader if gui else None)
    md.grid_initialize()
    return md

def oscillator(dt, integrator, gui=True):
    n = 1
    boxlength = 10
    pot = QuadraticWell(2, [5,5,5])
    md = MolecularDynamics(n, boxlength, integrator, dt,
        temperature = 1 if integrator == NVTIntegrator else -1,
        external=pot,
        shader=CanvasShader if gui else None)
    pos_init = np.array([[1., 0., 0.]])
    vel_init = np.array([[0., 3., 0.]])
    md.read_restart(pos_init, vel_init, centered=True)
    return md

def chain(nchain, temp, dt, integrator, gui=True):
    boxlength = 160
    l0 = 1.54
    lchain = 100
    n = nchain * lchain
    pos = np.zeros((nchain, lchain, DIM))
    pos[:, :, 0] = np.arange(lchain) * l0
    for i in range(nchain):
        pos[i, :, 1] += boxlength / 1.2 / nchain * i
    pos = pos.reshape(n, DIM)
    # center the molecule
    pos -= np.mean(pos, axis=0)
    temperature = 573
    md = MolecularDynamics(n, boxlength, integrator, dt,
        temperature=temperature,
        nonbond=LennardJones(3.95, 46, 14, rcutin=2*l0),
        bonded=HarmonicPair(96500 / 2, l0),
        shader=CanvasShader if gui else None)
    for i in range(n - 1):
        if i % lchain != lchain - 1:
            md.make_bond(i, i + 1)
    md.read_restart(pos, centered=True)
    return md