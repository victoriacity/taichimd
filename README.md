# TaichiMD
Interactive, GPU-accelerated Molecular (& Macroscopic) Dynamics using the Taichi programming language

**The TaichiMD package is still in a very early stage and is undergoing constant development.**

![](propane.gif)![](mpm99.gif)

## Goals
* To extend capabilities of the Taichi programming language in computer graphics to molecular simulation education and research
* To achieve interactive, real-time particle-based simulations accelerated by GPUs
* To provide a platform for rapid implementation of novel simulation algorithms and machine-learned simulations

## Examples

### Molecular simulation examples
`python run_examples.py [-h] example ensemble`

positional arguments:
```
  example     [lj | biglj | ho | chain | pr
              lj: Lenneard-Jones system with 4096 molecules, in reduced units; 
              biglj: Lenneard-Jones system with 0.5 million molecules, in reduced units\n\
              ho: Harmonic oscillator around the center of the simulation box; 
              chain: 5 harmonic-bond chain molecules with 100 atoms each, bond bending and torsion not included, in real
              units
              pr: 512 propane molecules using the TraPPE-UA force field with a harmonic bond
              stretching potential at 423 K in a 50*50*50 angstrom box
  ensemble    [NVE | NVT] 
              NVE ensemble with Verlet integration or NVT ensemble with Nose-Hoover thermostat
```
optional arguments: `-h, --help  show help message and exit`

### Macroscopic simulation examples
`mpm28.py`: The MPM88 Taichi example (fluid simulation) in 3 dimensions.
`mpm48.py`: The MPM99 Taichi example (fluid, jelly and snow) in 3 dimensions.


## Graphics
TaichiMD mainly utilizes my *experimental version* of [`taichi_three`](https://github.com/victoriacity/taichi_three) package for rendering graphics. To obtain the working version of taichi_three with TaichiMD (0.0.3 with multiple light support, point lights, and multiple framebuffers), please download the package from the my fork of taichi_three repository.

If taichi_three was not imported correctly, the example simulations will use the Taichi GUI canvas for graphics.


## Future work
### Microscopic forces
* Add support for torsional potentials as cosine series
* Add Coulomb forces between charged atoms
### Acceleration algorithms
* Implement Ewald summation and/or fast multipole method for long-range forces (Coulomb)
### Macroscopic forces
* Implement gravity and wall boundaries
* Incorporate macroscopic particle simulation algorithms (SPH, etc.)
* Incorporate agent-based simulation algorithms (crowd simulation, particle swarm) and optimization-based integrators

