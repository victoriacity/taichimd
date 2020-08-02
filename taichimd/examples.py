import argparse
import taichi as ti
import numpy as np
from taichimd.system import MolecularDynamics
from taichimd.interaction import *
from taichimd.molecule import Molecule
from taichimd.forcefield import ClassicalFF
from taichimd.integrator import *
from taichimd.ui import CanvasRenderer

def ljsystem(n, rho, temp, dt, integrator, gui=True):
    boxlength = (n / rho) ** (1 / DIM)
    mol = Molecule(1)
    ff = ClassicalFF(nonbond=LennardJones(rcut=2.5))
    ff.set_params(nonbond={1:[1,1]})
    md = MolecularDynamics({mol: n}, boxlength, dt, ff,
        integrator, temperature=temp,
        renderer=CanvasRenderer if gui else None)
    md.grid_initialize()
    return md

def oscillator(dt, integrator, gui=True):
    n = 1
    boxlength = 10
    mol = Molecule(1)
    ff = ClassicalFF(external=QuadraticWell(2., [5.,5.,5.]))
    md = MolecularDynamics({mol: n}, boxlength, dt, ff,
        integrator, temperature=1. if integrator == NVTIntegrator else -1,
        renderer=CanvasRenderer if gui else None)
    pos_init = np.array([[1., 0., 0.]])
    vel_init = np.array([[0., 3., 0.]])
    md.read_restart(pos_init, vel_init, centered=True)
    return md

def chain(nchain, temp, dt, integrator, gui=True):
    boxlength = 160
    l0 = 1.54
    lchain = 100
    pos = np.zeros((nchain, lchain, DIM))
    pos[:, :, 0] = np.arange(lchain) * l0
    for i in range(nchain):
        pos[i, :, 1] += boxlength / 1.2 / nchain * i
    pos = pos.reshape(nchain * lchain, DIM)
    # center the molecule
    pos -= np.mean(pos, axis=0)
    temperature = 573.0
    # intramolecular exclusion matrix
    # exclude intramolecular interaction when
    # two atoms are less than 4 bonds apart
    adj = np.eye(lchain, k=1) + np.eye(lchain, k=-1)
    intra = (adj + adj@adj + adj@adj@adj).astype(np.bool)
    
    mol = Molecule([1] * lchain, 
        bond=[[1, i, i+1] for i in range(lchain - 1)],
        intra=intra)
    ff = ClassicalFF(nonbond=LennardJones(rcut=14),
            bonded=HarmonicPair())
    ff.set_params(nonbond={1:[3.95,460.0]}, bonded={1:[96500.0/2, l0]})
    md = MolecularDynamics({mol: nchain}, boxlength, dt, ff,
        integrator, temperature=temperature,
        renderer=CanvasRenderer if gui else None)
    md.read_restart(pos, centered=True)
    return md