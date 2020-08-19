import taichi as ti
import numpy as np
from .ui import GUI, TemperatureControl, MDRenderer
from .consts import *
from .integrator import VerletIntegrator
from .common import *
from .analyzer import *
from .grid import *

class Boundary:
    '''
    Enum class for boundary conditions
    '''
    FREE = 0
    PERIODIC = 1
    REFLECTIVE = 2


@ti.data_oriented
class Simulation:

    is_atomic = True
    
    '''
    A general simulation class.
    '''
    def __init__(self, n_particles, 
                integrator,
                forcefield=None,
                boundary=Boundary.FREE,
                boxlength=1,
                grid=None,
                renderer=MDRenderer):
        self.n_particles = n_particles
        self.boundary = boundary
        self.boxlength = boxlength
        self.built = False
        self.particle_snode = ti.root.dense(ti.i, self.n_particles)
        self.variable_snode = ti.root
        if ti.static(grid is not None):
            self.grid_snode = None
        self.modules = []
        self.analyzers = []
        self.grids = []
        self.fields = []
        self.initvals = []

        self.add_attr("position")
        self.add_attr("velocity")
        self.add_attr("type", dims=(), dtype=ti.i32)
        self.integrator = self.add_module(integrator)
        self.dt = integrator.dt

        if self.integrator.requires_force:
            self.add_attr("force")
        if self.integrator.requires_hessian:
            self.hessian = ti.Matrix(DIM, DIM, dt=ti.f32)
            self.pair_snode = ti.root.dense(ti.ij, (self.n_particles, self.n_particles))
            self.pair_snode.place(self.hessian)

        self.forcefield = self.add_module(forcefield)
        if self.forcefield and self.forcefield.is_conservative:
            self.add_var("ep")
        
        if type(grid) == list:
            for g in grid:
                self.add_module(g)
        else:
            self.add_module(grid)

        # spawns GUI
        if renderer:
            self.gui = GUI(self, renderer)


    def add_module(self, module):
        if module is None:
            return None
        module.register(self)
        self.modules.append(module)
        if issubclass(type(module), Analyzer):
            self.analyzers.append(module)
        elif issubclass(type(module), Grid):
            self.grids.append(module)
        return module

    def add_layout(self, name, snode, dims=(), dtype=ti.f32, init=0):
        ti_field = create_field(dims, dtype)
        setattr(self, name, ti_field)
        snode.place(ti_field)
        self.fields.append(ti_field)
        self.initvals.append(init)
       
    def add_attr(self, name, dims=(DIM,), dtype=ti.f32):
        self.add_layout(name, self.particle_snode, dims, dtype)

    def add_var(self, name, dims=(), dtype=ti.f32):
        self.add_layout(name, self.variable_snode, dims, dtype)

    def add_field(self, name, dims=(DIM,), dtype=ti.f32):
        if ti.static(self.grid_snode is not None):
            self.add_layout(name, self.grid_snode, dims, dtype)
        else:
            print("Warning: the simulation system is not using grids,"
                "adding scalar/vector fields will have no effect.")

    def build(self):
        self.fill_composition()
        for m in ti.static(self.modules):
            m.build()
        self.built = True
        if hasattr(self, "pos_np"):
            from_numpy_chk(self.position, self.pos_np)
        if hasattr(self, "vel_np"):
            from_numpy_chk(self.velocity, self.vel_np)
        if hasattr(self, "type_np"):
            from_numpy_chk(self.type, self.type_np)
        print("[TaichiMD] Simulation system has been built")

    def fill_composition(self):
        self.type.fill(0)

    def init_random(self, center=(0.5, 0.5, 0.5), length=1,
            start=0, end=None, inittype=None):
        end = end or self.n_particles
        l = self.boxlength
        if not hasattr(self, "pos_np"):
            self.pos_np = np.random.rand(self.n_particles, DIM) * l
        n = end - start
        center = np.array(center).reshape(-1, DIM)
        origin = center - length / 2
        self.pos_np[start:end, :] = np.random.rand(n, DIM) * length + origin
        if inittype is not None:
            if not hasattr(self, "type_np"):
                self.type_np = np.zeros(self.n_particles).astype(np.int)
            self.type_np[start:end] = inittype
        



    '''
    Runs the simulation.
    '''
    def run(self, nframe=0, irender=10, save=False, pause=False):
        if not self.built:
            self.build()
        if nframe == 0:
            nframe = int(1e12)
        play = not pause
        for i in range(nframe):
            if play:
                self.step()
            if self.gui is not None and irender > 0 and i % irender == 0:
                if save:
                    play = self.gui.show("frame%i.png" % (i // irender)) 
                else:
                    play = self.gui.show()
                if pause:
                    play = not play

    @ti.kernel
    def step(self):
        self.integrator.prestep()
        self.calc_force()
        self.integrator.poststep()
        self.apply_boundary()
        if ti.static(self.analyzers):
            for analyzer in ti.static(self.analyzers):
                analyzer.use()
        

    @ti.func
    def calc_force(self):
        if ti.static(self.grids):
            for g in ti.static(self.grids):
                g.use()
        if ti.static(self.forcefield is not None):
            self.forcefield.calc_force()

    '''
    Calculates distance with periodic boundary conditions
    and wraps a particle into the simulation box.
    '''
    @ti.func
    def calc_distance(self, x1, x2):
        dist = ti.Vector([0.0] * DIM)
        for i in ti.static(range(DIM)):
            dist[i] = x1[i] - x2[i]
            if ti.static(self.boundary == Boundary.PERIODIC):
                while dist[i] <= -0.5 * self.boxlength:
                    dist[i] = dist[i] + self.boxlength
                while dist[i] > 0.5 * self.boxlength:
                    dist[i] = dist[i] - self.boxlength
        return dist

    @ti.func
    def wrap(self, x):
        for i in ti.static(range(DIM)):
            if x[i] <= 0:
                x[i] = x[i] + self.boxlength
            elif x[i] > self.boxlength:
                x[i] = x[i] - self.boxlength
        return x

    @ti.func
    def apply_boundary(self):
        if ti.static(self.boundary == Boundary.PERIODIC):
            for i in self.position:  
                self.position[i] = self.wrap(self.position[i])
        elif ti.static(self.boundary == Boundary.REFLECTIVE):
            for i in self.position:
                for d in ti.static(range(DIM)):
                    if self.position[i][d] < 0 or self.position[i][d] > self.boxlength:
                        self.velocity[i][d] = -self.velocity[i][d]

        

@ti.data_oriented
class MolecularDynamics(Simulation):
  
    '''
    Initializes the object, set up python scope
    variables and taichi vectors.
    '''
    def __init__(self, composition, boxlength, dt, forcefield,
                integrator=VerletIntegrator, temperature=-1,
                use_grid=False,
                renderer=None):
        n_particles = sum(m.natoms * n for m, n in composition.items())
        max_atoms = max(m.natoms for m in composition.keys())
        self.is_atomic = max_atoms == 1
        self.temperature = float(temperature)
        self.composition = composition
        self.mol_objs = list(composition.keys())
        self.n_molecules = sum(composition.values())

        if use_grid and forcefield.nonbond is not None:
            grid = NeighborList(forcefield.nonbond.rcut)
        else:
            grid = None

        super().__init__(n_particles, integrator(dt), forcefield, 
            boundary=Boundary.PERIODIC,
            grid=grid,
            boxlength=float(boxlength), renderer=renderer)

        # molecule table
        if not self.is_atomic:
            self.molecule_snode = ti.root.dense(ti.i, self.n_molecules)
            self.add_layout("moltypes", self.molecule_snode, dims=(), dtype=ti.i32)
        if self.gui is not None and temperature > 0:
            self.gui.add_component(TemperatureControl())
        self.energy = EnergyAnalyzer()
        self.add_module(self.energy)
        
           
    def set_temp(self, temperature):
        self.temperature = temperature
        self.integrator.set_temp(temperature)

    def fill_composition(self):
        types = []
        for m, n in self.composition.items():
            types += m.atoms * n
        from_numpy_chk(self.type, np.array(types, dtype=np.int))
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
            from_numpy_chk(self.moltypes, np.array(moltypes))
            

    @ti.pyfunc
    def get_temp(self) -> ti.f32:
        return 2 * self.ek[None] / (self.n_particles * DIM)

    '''
    Initializes the simulation system by placing particles on a regular grid
    and randomize their velocities according to the temperature. 
    '''
    def grid_initialize(self):
        self.build()
        n_pow = int(self.n_molecules ** (1. / DIM))
        # n_axes = [nx, ny, ...] is the number of particles along each axis to be placed.
        n_axes = np.array([n_pow] * DIM)
        for i in range(DIM):
            if n_pow ** (DIM - i) * (n_pow + 1) ** i < self.n_molecules:
                n_axes[i] += 1
        dx = self.boxlength / n_axes
        pos_cm = make_cell(n_axes, dx, self.n_molecules, x0=dx/2)
        pos_all = []
        i0 = 0
        for m, n in self.composition.items():
            pos_mol = np.repeat(pos_cm[i0: i0 + n], m.natoms, axis=0)\
                + np.tile(m.struc, (n, 1))
            i0 += n
            pos_all.append(pos_mol)
        from_numpy_chk(self.position, np.vstack(pos_all))
        if self.temperature > 0:
            self.randomize_velocity()
        else:
            self.velocity.fill(0)
        self.energy.calculate_energy()
        

    
    def randomize_velocity(self, keep_molecules=True):
        vs = np.random.random((self.n_particles, DIM)) - 0.5
        vcm = np.mean(vs, axis=0).reshape((1, DIM))
        vs -= vcm
        i0 = 0
        if not self.is_atomic and keep_molecules:
            for m, n in self.composition.items():
                vel_mol = vs[i0: i0 + n * m.natoms].reshape(-1, m.natoms, DIM)
                vel_mol = np.tile(np.mean(vel_mol, axis=1).reshape(-1, 1, DIM) / m.natoms, (1, m.natoms, 1))
                vs[i0: i0 + n * m.natoms] = np.repeat(np.mean(vel_mol, axis=1) / m.natoms, m.natoms, axis=0)
                i0 += n * m.natoms
        vs *= np.sqrt(DIM * self.temperature * self.n_particles / np.sum(vs ** 2))
        from_numpy_chk(self.velocity, vs)


    def read_restart(self, position, velocity=None, centered=False):
        self.build()
        if centered:
            position += np.ones(DIM) * self.boxlength / 2
        from_numpy_chk(self.position, position)
        if velocity is not None:
            from_numpy_chk(self.velocity, velocity)
        elif self.temperature > 0:
            self.randomize_velocity()
        else:
            self.velocity.fill(0)
        self.energy.calculate_energy()


