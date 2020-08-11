import taichi as ti
import numpy as np
from .ui import GUI
from .consts import *
from .integrator import VerletIntegrator

@ti.data_oriented
class MolecularDynamics:

    
    '''
    Initializes the object, set up python scope
    variables and taichi vectors.
    '''
    def __init__(self, composition, boxlength, dt, forcefield,
                integrator=VerletIntegrator, temperature=-1,
                renderer=None):
        # python scope variables
        self.boxlength = float(boxlength)
        self.temperature = float(temperature)
        self.composition = composition
        self.mol_objs = list(composition.keys())
        self.n_molecules = sum(composition.values())
        self.n_particles = sum(m.natoms * n for m, n in composition.items())
        max_atoms = max(m.natoms for m in composition.keys())
        self.is_atomic = max_atoms == 1
        self.forcefield = forcefield.register(self)
        # particle properties
        self.type = ti.var(dt=ti.i32)
        self.position = ti.Vector(DIM, dt=ti.f32)
        self.velocity = ti.Vector(DIM, dt=ti.f32)
        self.force = ti.Vector(DIM, dt=ti.f32)
        ti.root.dense(ti.i, self.n_particles).place(
            self.position, self.velocity, self.force, self.type)
        # molecule table
        if not self.is_atomic:
            self.molecules = ti.var(dt=ti.i32)
            self.moltypes = ti.var(dt=ti.i32)
            ti.root.dense(ti.i, self.n_molecules).place(self.moltypes, self.molecules)
        # kinetic energy
        self.ek = ti.var(dt=ti.f32, shape=())
        # potential energy
        self.ep = ti.var(dt=ti.f32, shape=())
        self.time = -1
        # spawns GUI
        if renderer:
            self.gui = GUI(self, renderer)


        self.integrator = integrator(self, dt)
        if self.integrator.requires_hessian:
            self.hessian = ti.Matrix(DIM, DIM, dt=ti.f32)
            ti.root.dense(ti.ij, (self.n_particles, self.n_particles)).place(self.hessian)
        self.place_molecules()
        if temperature > 0:
            self.set_temp(temperature)
        
    def set_temp(self, temperature):
        self.temperature = temperature
        self.integrator.set_temp(temperature)

    def place_molecules(self):
        types = []
        for m, n in self.composition.items():
            types += m.atoms * n
        self.type.from_numpy(np.array(types, dtype=np.int))
        if not self.is_atomic:
            i0 = 0
            moltypes = []
            mol_prefix = []
            for i, packed in enumerate(self.composition.items()):
                m, n = packed
                mol_prefix.append(np.arange(n) * m.natoms)
                moltypes += [i] * n
                self.forcefield.populate_tables(i0, m, n)
                i0 += n
            self.moltypes.from_numpy(np.array(moltypes))
            self.molecules.from_numpy(np.hstack(mol_prefix))
        self.forcefield.build()
            

    @ti.kernel
    def add_molecules(self, i0: ti.i32, natom: ti.i32, nmolec: ti.i32):
        for i in range(natom * nmolec):
            iatom = i % natom
            imolec = i // natom
            self.molecules[imolec, iatom] = i + i0

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
        n_pow = int(self.n_molecules ** (1. / DIM))
        # n_axes = [nx, ny, ...] is the number of particles along each axis to be placed.
        n_axes = np.array([n_pow] * DIM)
        for i in range(DIM):
            if n_pow ** (DIM - i) * (n_pow + 1) ** i < self.n_molecules:
                n_axes[i] += 1
        disp = self.boxlength / n_axes
        coords_1d = [d * (0.5 + np.arange(n)) for d, n in zip(disp, n_axes)]
        pos_cm = (np.stack(np.meshgrid(*coords_1d)).reshape(DIM, -1)\
                [:, :self.n_molecules].T)
        pos_all = []
        i0 = 0
        for m, n in self.composition.items():
            pos_mol = np.repeat(pos_cm[i0: i0 + n], m.natoms, axis=0)\
                + np.tile(m.struc, (n, 1))
            i0 += n
            pos_all.append(pos_mol)
        self.position.from_numpy(np.vstack(pos_all))
        self.time = 0
        self.randomize_velocity()
        self.calculate_energy()

    #@ti.kernel
    def randomize_velocity(self, keep_molecules=True):
        vs = np.random.random((self.n_particles, DIM)) - 0.5
        vcm = np.mean(vs, axis=0).reshape((1, DIM))
        vs -= vcm
        i0 = 0
        #if keep_molecules:
        for m, n in self.composition.items():
            vel_mol = vs[i0: i0 + n * m.natoms].reshape(-1, m.natoms, DIM)
            vel_mol = np.tile(np.mean(vel_mol, axis=1).reshape(-1, 1, DIM) / m.natoms, (1, m.natoms, 1))
            vs[i0: i0 + n * m.natoms] = np.repeat(np.mean(vel_mol, axis=1) / m.natoms, m.natoms, axis=0)
            i0 += n * m.natoms
        vs *= np.sqrt(DIM * self.temperature * self.n_particles / np.sum(vs ** 2))
        self.velocity.from_numpy(vs)
        '''
        vs = np.random.random((self.n_molecules, DIM)) - 0.5
        vcm = ti.Vector([0, 0, 0])
        for i in range(self.n_molecules):
            m = self.molecule_types[self.type[i]]
            for j in ti.static(range(m.natoms)):
                self.velocity[i, j] = vs[i]
                vcm += self.velocity[i, j]
        vcm /= self.natoms
        v2tot = 0.0
        for i in range(self.n_molecules):
            m = self.molecule_types[self.type[i]]
            for j in ti.static(range(m.natoms)):
                self.velocity[i, j] -= vcm
                v2tot += (self.velocity[i, j] ** 2).sum()
        vscale = ti.sqrt(DIM * self.temperature * self.natoms / v2tot)
        for i in range(self.n_molecules):
            m = self.molecule_types[self.type[i]]
            for j in ti.static(range(m.natoms)):
                self.velocity[i, j] *= vscale
        '''

    @ti.kernel
    def init_molecules(self):
        #print(self.molecule_types)
        for i in range(self.n_molecules):
            m = self.molecule_types[self.type[i]]
            xcm = self.position[i, 0]
            for j in ti.static(range(m.natoms)):
                self.position[i, j] = xcm + m.struc[j]


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
        self.forcefield.calc_force()


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
        play = True
        for i in range(nframe):
            if play:
                self.step()
            if self.gui is not None and irender > 0 and i % irender == 0:
                if save:
                    play = self.gui.show("frame%i.png" % (i // irender)) 
                else:
                    play = self.gui.show()