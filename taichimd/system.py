import taichi as ti
import numpy as np
from .ui import GUI
from .consts import *

@ti.data_oriented
class MolecularDynamics:

    MAX_BOND = 6
    
    '''
    Initializes the object, set up python scope
    variables and taichi vectors.
    '''
    def __init__(self, n_particles, boxlength, integrator, dt, 
                temperature=-1, bonded=None, nonbond=None, external=None,
                shader=None):
        # python scope variables
        self.boxlength = boxlength
        self.temperature = temperature
        self.n_particles = n_particles
        # set up interactions
        self.bonded_potential = bonded
        self.nonbond_potential = nonbond
        self.external_potential = external
        # ti variables
        self.position = ti.Vector(DIM, dt=ti.f32)
        self.velocity = ti.Vector(DIM, dt=ti.f32)
        self.force = ti.Vector(DIM, dt=ti.f32) 
        ti.root.dense(ti.i, self.n_particles).place(
            self.position, self.velocity, self.force)
        #self.type = ti.var(dt=ti.u16, shape=self.n_particles)
        self.nbonds = 0
        self.bond_table = ti.Vector(2, dt=ti.u32, shape=self.MAX_BOND * self.n_particles) 
        # kinetic energy
        self.ek = ti.var(dt=ti.f32, shape=())
        # potential energy
        self.ep = ti.var(dt=ti.f32, shape=())
        self.time = -1
        self.integrator = integrator(self, dt)
        if self.integrator.requires_hessian:
            self.hessian = ti.Matrix(DIM, DIM, dt=ti.f32, 
                shape=(self.n_particles, self.n_particles))
        if self.temperature > 0:
            self.set_temp(temperature)
        # spawns GUI
        if shader:
            self.gui = GUI(self, shader)
        
    
    def set_temp(self, temperature):
        self.temperature = temperature
        self.integrator.set_temp(temperature)

    @ti.pyfunc
    def get_temp(self) -> ti.f32:
        return 2 * self.ek[None] / (self.n_particles * DIM)

    @ti.pyfunc
    def energy(self):
        return self.ek[None] + self.ep[None]
    
    @ti.kernel
    def calculate_energy(self) -> ti.f32:
        self.calc_force()
        self.ek[None] = 0.0
        for i in self.position:
            self.ek[None] += (self.velocity[i] ** 2).sum() / 2.0
        return self.energy()

    def make_bond(self, i, j):
        if i < j:
            i, j = j, i
        self.bond_table[self.nbonds][0] = i
        self.bond_table[self.nbonds][1] = j
        self.nbonds += 1

    '''
    Initializes the simulation system by placing particles on a regular grid
    and randomize their velocities according to the temperature. 
    '''
    def grid_initialize(self, temperature=None):
        if temperature == None:
            if self.temperature < 0:
                raise Exception("Temperature is required for grid initialization!")
            else:
                temperature = self.temperature
        n_pow = int(self.n_particles ** (1. / DIM))
        # n_axes = [nx, ny, ...] is the number of particles along each axis to be placed.
        n_axes = np.array([n_pow] * DIM)
        for i in range(DIM):
            if n_pow ** (DIM - i) * (n_pow + 1) * i < self.n_particles:
                n_axes[i] += 1
        disp = self.boxlength / n_axes
        coords_1d = [d * (0.5 + np.arange(n)) for d, n in zip(disp, n_axes)]
        self.position.from_numpy(
            np.stack(np.meshgrid(*coords_1d)).reshape(DIM, -1)[:, :self.n_particles].T)
        self.time = 0
        self.randomize_velocity()
        self.calculate_energy()

    def randomize_velocity(self):
        vs = np.random.random((self.n_particles, DIM)) - 0.5
        vcm = np.mean(vs, axis=0).reshape((1, DIM))
        vs -= vcm
        vs *= np.sqrt(DIM * self.temperature * self.n_particles / np.sum(vs ** 2))
        self.velocity.from_numpy(vs)



    def read_restart(self, position, velocity=None, centered=False):
        if centered:
            position += np.ones(DIM) * self.boxlength / 2
        self.position.from_numpy(position)
        if velocity is not None:
            self.velocity.from_numpy(velocity)
        elif self.temperature > 0:
            self.randomize_velocity()
        else:
            self.velocity.fill(0)
        self.calculate_energy()
        self.time = 0

    def set_dt(dt):
        self.integrator.dt = dt

    '''
    Calculates distance with periodic boundary conditions
    and wraps a particle into the simulation box.
    '''
    @ti.func
    def calc_distance(self, x1, x2):
        dist = ti.Vector([0.0] * DIM)
        for i in ti.static(range(DIM)):
            dist[i] = x1[i] - x2[i]
            if dist[i] <= -0.5 * self.boxlength:
                dist[i] += self.boxlength
            elif dist[i] > 0.5 * self.boxlength:
                dist[i] -= self.boxlength
        return dist

    @ti.func
    def wrap(self, x):
        for i in ti.static(range(DIM)):
            if x[i] <= 0:
                x[i] += self.boxlength
            elif x[i] > self.boxlength:
                x[i] -= self.boxlength
        return x

    @ti.func
    def calc_force(self):
        self.ep[None] = 0.0
        for i in self.force:
            self.force[i].fill(0)
            if ti.static(self.external_potential != None):
                self.ep[None] += self.external_potential(self.position[i])
                self.force[i] += self.external_potential.force(self.position[i])
                if ti.static(self.integrator.requires_hessian):
                    self.hessian[i, i] += self.external_potential.hessian(self.position[i])
        if ti.static(self.nonbond_potential != None):
            for i, j in ti.ndrange(self.n_particles, self.n_particles):
                if i < j:
                    d = self.calc_distance(self.position[j], self.position[i])
                    r2 = (d ** 2).sum()
                    uij = self.nonbond_potential(r2)
                    if uij != 0:
                        self.ep[None] += uij
                        force = self.nonbond_potential.force(d, r2)
                        # += performs atomic add
                        self.force[i] += force
                        self.force[j] -= force
                        if ti.static(self.integrator.requires_hessian):
                            h = self.nonbond_potential.hessian(d, r2)
                            self.hessian[i, j] = h
                            self.hessian[j, i] = h
                            self.hessian[i, i] -= h
                            self.hessian[j, j] -= h
        if ti.static(self.nbonds > 0):
            for n in range(self.nbonds):
                i = self.bond_table[n][0]
                j = self.bond_table[n][1]
                d = self.calc_distance(self.position[j], self.position[i])
                if ti.static(self.bonded_potential == None):
                    raise NotImplementedError("Rigid bonds are not implemented yet!") 
                else:
                    r2 = (d ** 2).sum()
                    self.ep[None] += self.bonded_potential(r2)
                    force = self.bonded_potential.force(d, r2)
                    self.force[i] += force
                    self.force[j] -= force
                    if ti.static(self.integrator.requires_hessian):
                        h = self.bonded_potential.hessian(d, r2)
                        self.hessian[i, j] = h
                        self.hessian[j, i] = h
                        self.hessian[i, i] -= h
                        self.hessian[j, j] -= h


    def step(self):
        self.integrator.integrate()
        self.time += self.integrator.dt

    '''
    Runs the simulation.
    '''
    def run(self, nframe=0, irender=10, save=False):
        if self.time < 0:
            raise Exception("System has not been initialized!")
        if nframe == 0:
            nframe = int(1e12)
        for i in range(nframe):
            self.step()
            if self.gui is not None and irender > 0 and i % irender == 0:
                if save:
                    self.gui.show("frame%i.png" % (i // irender)) 
                else:
                    self.gui.show()