import argparse
import taichi as ti
import numpy as np
from .system import MolecularDynamics
from .interaction import *
from .integrator import *
from .ui import CanvasShader


def ljsystem(n, rho, temp, dt, integrator):
    boxlength = (n / rho) ** (1 / DIM)
    md = MolecularDynamics(n, boxlength, integrator, dt,
        temperature=temp,
        nonbond=LennardJones(1, 1, 2.5),
        shader=CanvasShader)
    md.grid_initialize()
    return md

def oscillator(dt, integrator):
    n = 1
    boxlength = 10
    pot = QuadraticWell(2, [5,5,5])
    md = MolecularDynamics(n, boxlength, integrator, dt,
        temperature = 1 if integrator == NVTIntegrator else -1,
        external=pot,
        shader=CanvasShader)
    pos_init = np.array([[1., 0., 0.]])
    vel_init = np.array([[0., 3., 0.]])
    md.read_restart(pos_init, vel_init, centered=True)
    return md

def chain(nchain, temp, dt, integrator):
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
        shader=CanvasShader)
    for i in range(n - 1):
        if i % lchain != lchain - 1:
            md.make_bond(i, i + 1)
    md.read_restart(pos, centered=True)
    return md


def parse_args():
    parser = argparse.ArgumentParser(description='Run taichimd examples')
    parser.add_argument('example', type=str, help='[lj | ho | chain]\n\
                    lj: Lenneard-Jones system with 4096 molecules, in reduced units;\n\
                    ho: Harmonic oscillator around the center of the simulation box;\n\
                    chain: 5 harmonic-bond chain molecules with 100 atoms each,\
                    bond bending and torsion not included, in real units')
    parser.add_argument('ensemble', type=str, help='[NVE | NVT]\n\
        NVE ensemble with verlet integration or NVT ensmble with Nose-Hoover thermostat')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ti.init(arch=ti.gpu)
    if args.ensemble == 'NVE':
        integrator = VerletIntegrator
    elif args.ensemble == 'NVT':
        integrator = NVTIntegrator
    else:
        raise ValueError("Unknown ensemble!")
    if args.example == 'lj':
        md = ljsystem(4096, 0.1, 1, 0.01, integrator)
    elif args.example == 'ho':
        md = oscillator(0.01, integrator)
    elif args.example == 'chain':
        md = chain(5, 573, 0.0005, integrator)
    else:
        raise ValueError("Unknown system!")
    md.run()