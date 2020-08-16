import os, sys
import taichi as ti
sys.path.append(os.getcwd())
from taichimd.examples import *

def parse_args():
    parser = argparse.ArgumentParser(description='Run taichimd examples')
    parser.add_argument('example', type=str, help='[lj | ho | chain]\n\
                    lj: Lenneard-Jones system with 4096 molecules, in reduced units;\n\
                    biglj: Lenneard-Jones system with 0.5 million molecules, in reduced units\n\
                    ho: Harmonic oscillator around the center of the simulation box;\n\
                    chain: 5 harmonic-bond chain molecules with 100 atoms each,\
                    bond bending and torsion not included, in real units\
                    pr: 512 propane molecules using the TraPPE-UA force field with a harmonic bond\
                    stretching potential at 423 K in a 50*50*50 angstrom box')
    parser.add_argument('ensemble', type=str, help='[NVE | NVT]\n\
        NVE ensemble with verlet integration or NVT ensmble with Nose-Hoover thermostat')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ti.init(arch=ti.cuda)
    if args.ensemble == 'NVE':
        integrator = VerletIntegrator
    elif args.ensemble == 'NVT':
        integrator = NVTIntegrator
    else:
        raise ValueError("Unknown ensemble!")
    irender = 5
    if args.example == 'lj':
        md = ljsystem(4096, 0.1, 1.5, 0.01, integrator)
    elif args.example == 'biglj':
        try:
            ti.init(arch=ti.cuda, device_memory_GB=4)
            md = ljsystem(524288, 0.1, 1.5, 0.01, integrator, use_grid=True)
            irender = 1
        except RuntimeError:
            print("Not enough resources: a CUDA-enabled GPU with at least 6 GB of memory is required to run this example.")
            exit(1)
    elif args.example == 'ho':
        md = oscillator(0.01, integrator)
    elif args.example == 'chain':
        md = chain(10, 573, 0.0005, integrator)
    elif args.example == 'pr':
        md = propane(512, 423, 0.001, integrator)
    else:
        raise ValueError("Unknown system!")
    md.run(irender=irender)