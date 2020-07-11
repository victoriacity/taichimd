import md
from md import MolecularDynamics


if __name__ == "__main__":
    # number of particles
    n = 4000
    density = 0.1
    temperature = 1.5
    boxlength = (n / density) ** (1 / md.DIM)
    md = MolecularDynamics(density, temperature, boxlength)
    print(md.n_particles, boxlength)
    # melt and equilibrate
    md.set_temp(1.5)
    md.run(100, irender=-1)
    md.set_temp(temperature)
    md.run(200, irender=-1)
    # production
    md.run(10)
    md.run(md.dt * 300, save=True)
