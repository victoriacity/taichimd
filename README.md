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
  ensemble    [NVE | NVT] 
              NVE ensemble with verlet integration or NVT ensmble with Nose-Hoover thermostat
```
optional arguments: `-h, --help  show help message and exit`


## Future work
### Microscopic forces
* Add support for harmonic bond bending potentials and torsional potentials as cosine series
* Add Coulomb forces between charged atoms
### Acceleration algorithms
* Implement grid-based neighbor lists for large systems
* Implement Ewald summation and/or fast multipole method for long-range forces (Coulomb)
### Macroscopic forces
* Implement gravity and wall boundaries
* Incorporate macroscopic particle simulation algorithms (SPH, MPM, etc.)
* Incorporate agent-based simulation algorithms (crowd simulation, particle swarm) and optimization-based integrators
### Graphics
* Add interactive 3D renderer, for particle simulations an SDF renderer is preferred due to the types of primitives to render

