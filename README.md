# TaichiMD
Interactive, GPU-accelerated Molecular (& Macroscopic) Dynamics using the Taichi programming language

**The TaichiMD package is still in a very early stage and is undergoing constant development.**

![](preview.gif)
## Goals
* To extend capabilities of the Taichi programming language in computer graphics to molecular simulation education and research
* To achieve interactive, real-time molecular dynamics simulations accelerated by GPUs
* To provide a platform for rapid implementation of novel simulation algorithms and machine-learned simulations

## Examples
`python run_examples.py [-h] example ensemble`

positional arguments:
```
  example     [lj | ho | chain]
              lj: Lenneard-Jones system with 4096 molecules, in reduced units; 
              ho: Harmonic oscillator around the center of the simulation box; 
              chain: 5 harmonic-bond chain molecules with 100 atoms each, bond bending and torsion not included, in real
              units
              pr: 512 propane molecules using the TraPPE-UA force field with a harmonic bond
              stretching potential at 423 K in a 50*50*50 angstrom box
  ensemble    [NVE | NVT] 
              NVE ensemble with Verlet integration or NVT ensemble with Nose-Hoover thermostat
```
optional arguments: `-h, --help  show help message and exit`


## Graphics
TaichiMD mainly utilizes the [`taichi_three`](https://github.com/taichi-dev/taichi_three/tree/dev) (**0.0.3** or higher) package for rendering graphics. To obtain the working version of taichi_three with TaichiMD (current latest version on PyPI is 0.0.2), please download the package from the **dev branch** of taichi_three repository.

If taichi_three was not imported correctly, the example simualtions will use the Taichi GUI canvas for graphics.


## Future work
### Microscopic forces
* Add support for torsional potentials as cosine series
* Add Coulomb forces between charged atoms
### Acceleration algorithms
* Implement grid-based neighbor lists for large systems
* Implement Ewald summation and/or fast multipole method for long-range forces (Coulomb)
### Macroscopic forces
* Implement gravity and wall boundaries
* Incorporate macroscopic particle simulation algorithms (SPH, MPM, etc.)
* Incorporate agent-based simulation algorithms (crowd simulation, particle swarm) and optimization-based integrators

