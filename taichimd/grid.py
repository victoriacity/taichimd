import taichi as ti
from .consts import *
from .common import *

@ti.data_oriented
class Grid(Module):

    dynamics = True
    origin = 0

    def __init__(self, gridsize=None):
        self.gridsize = gridsize
        self.grid = None

    def register(self, system):
        super().register(system)
        self.dx = system.boxlength / self.gridsize
        self.inv_dx = 1 / self.dx
        self.position = system.position
        self.system.grid_snode = self.get_snode()
        self.layout()
        return self

    def get_snode(self):
        return ti.root.dense(ti.indices(*range(DIM)), (self.gridsize,) * DIM)

    def layout(self):
        pass

    @ti.func
    def grid_index(self, x):
        return (x * self.inv_dx - self.origin).cast(ti.i32)

    def clear(self):
        raise NotImplementedError

    @ti.func
    def use(self):
        self.clear()
        for i in self.position:
            X = self.grid_index(self.position[i])
            self.p2g(i, X)
        if ti.static(self.dynamics):
            if ti.static(self.grid is not None):
                for X in ti.grouped(self.grid):
                    self.grid_step(X)
            else:
                for X in ti.grouped(ti.ndrange(*((self.gridsize,) * DIM))):
                    self.grid_step(X)              
        for i in self.position:
            X = self.grid_index(self.position[i])
            self.g2p(i, X)

    @ti.func
    def p2g(self, i, X):
        raise NotImplementedError

    @ti.func
    def grid_step(self, X):
        raise NotImplementedError

    @ti.func
    def g2p(self, i, X):
        raise NotImplementedError


class NeighborList(Grid):

    dynamics = False
    max_cell = 64
    max_neighbors = 256

    def __init__(self, rcut):
        self.rcut = rcut

    def register(self, system):
        self.gridsize = int(system.boxlength / self.rcut)
        return super().register(system)


    def layout(self):
        self.cell_snode = self.system.grid_snode.dense(ti.indices(DIM), self.max_cell)
        #self.cell_snode = self.get_snode().dynamic(ti.indices(DIM), self.max_cell)
        self.neighbor_snode = self.system.particle_snode.dense(ti.j, self.max_neighbors)
        #self.neighbor_snode = ti.root.dense(ti.i, self.system.n_particles).dynamic(ti.j, self.max_neighbors)
        self.system.add_field("grid_n_particles", dims=(), dtype=ti.i32)
        self.system.add_attr("n_neighbors", dims=(), dtype=ti.i32)
        self.system.add_layout("grid_particles", self.cell_snode, dims=(), dtype=ti.i32)
        self.system.add_layout("neighbors", self.neighbor_snode, dims=(), dtype=ti.i32)

    def build(self):
        self.system.grid_particles.fill(0)
        self.system.neighbors.fill(0)

    @ti.func
    def clear(self):
        memset(self.system.grid_n_particles, 0)
        memset(self.system.grid_particles, 0)
        memset(self.system.neighbors, 0)
        memset(self.system.n_neighbors, 0)

    @ti.func
    def p2g(self, i, X):
        n = self.system.grid_n_particles[X].atomic_add(1)
        self.system.grid_particles[X, n] = i
        #ti.append(self.system.grid_particles.parent(), X, i)
        
    @ti.func
    def g2p(self, i, X):
        n_nb = 0
        for dX in ti.static(ti.grouped(ti.ndrange(*(((-1, 2),) * DIM)))):
            I = (X + dX) % self.gridsize # periodic boundary conditions
            #for j in range(ti.length(self.system.grid_particles.parent(), I)):
            #    nb = self.system.grid_particles[I, j]
            #    ti.append(self.system.neighbors.parent(), i, nb)
            for j in range(self.system.grid_n_particles[I]):
                nb = self.system.grid_particles[I, j]
                if i != nb:
                    self.system.neighbors[i, n_nb] = nb
                    n_nb += 1
        self.system.n_neighbors[i] = n_nb
       

class NeighborTable(Grid):

    dynamics = False
    max_density = 5

    def __init__(self, rcut):
        self.rcut = rcut

    def register(self, system):
        self.gridsize = int(system.boxlength / self.rcut)
        self.max_cell = int(self.max_density * self.rcut ** DIM)
        return super().register(system)


    def layout(self):
        self.system.add_field("grid_n_particles", dims=(), dtype=ti.i32)
        self.cell_snode = self.system.grid_snode.dense(ti.indices(DIM), self.max_cell)
        self.system.add_layout("grid_particles", self.cell_snode, dims=(), dtype=ti.i32)
        n = self.system.n_particles
        self.neighbor_snode = ti.root.bitmasked(ti.ij, (n, n))
        self.system.add_layout("neighbors", self.neighbor_snode, dims=(), dtype=ti.i32)

    def build(self):
        ti.root.deactivate_all()

    @ti.func
    def clear(self):
        memset(self.system.grid_n_particles, 0)
        memset(self.system.grid_particles, 0)
        memset(self.system.neighbors, 0)

    @ti.func
    def p2g(self, i, X):
        n = self.system.grid_n_particles[X].atomic_add(1)
        self.system.grid_particles[X, n] = i      


    @ti.func
    def g2p(self, i, X):
        for dX in ti.static(ti.grouped(ti.ndrange(*(((-1, 2),) * DIM)))):
            I = (X + dX) % self.gridsize # periodic boundary conditions
            for j in range(self.system.grid_n_particles[I]):
                nb = self.system.grid_particles[I, j]
                self.system.neighbors[i, nb] = 1

class ParticleInCell(Grid):

    buffer = 2

    def __init__(self, kernel, gridsize=None, mass=1, gravity=10):
        self.kernel = kernel
        self.origin = self.kernel.origin
        self.mass = mass
        self.gravity = gravity
        super().__init__(gridsize)

    def layout(self):
        self.system.add_field("grid_m", dims=(), dtype=ti.f32)
        self.system.add_field("grid_v", dims=DIM, dtype=ti.f32)
        self.grid = self.system.grid_m

    @ti.func
    def clear(self):
        for X in ti.grouped(self.system.grid_m):
            self.system.grid_m[X] = 0
            self.system.grid_v[X].fill(0)
    
    @ti.func
    def p2g(self, i, X):
        # fractional position of particle in the grid stencil w.r.t the lowest-indexed corner
        fx = self.position[i] * self.inv_dx - X 
        w = self.kernel(fx)
        for dX in ti.static(self.kernel.stencil()):
            weight = 1.0
            for d in ti.static(range(DIM)):
                weight *= w[dX[d]][d]
            self.system.grid_v[X + dX] += weight * (self.mass * self.system.velocity[i])
            self.system.grid_m[X + dX] += weight * self.mass

    @ti.func
    def grid_step(self, X):
        if self.system.grid_m[X] > 0:
            self.system.grid_v[X] = self.system.grid_v[X] / self.system.grid_m[X] 
            self.velocity_step(X)
            for d in ti.static(range(DIM)):
                cond = (X[d] < self.buffer and self.system.grid_v[X][d] < 0) \
                    or (X[d] > self.gridsize - self.buffer and self.system.grid_v[X][d] > 0)
                if cond:
                    self.system.grid_v[X][d] = 0 # Boundary conditions

    @ti.func
    def velocity_step(self, X):
        self.system.grid_v[X][1] -= self.system.dt * self.gravity # gravity

    
    @ti.func
    def g2p(self, i, X):
        fx = self.position[i] * self.inv_dx - X
        w = self.kernel(fx)
        new_v = ti.Vector.zero(ti.f32, DIM)
        for dX in ti.static(self.kernel.stencil()): # loop over 3x3 grid node neighborhood
            weight = 1.0
            for d in ti.static(range(DIM)):
                weight *= w[dX[d]][d]
            new_v += weight * self.system.grid_v[X + dX]
        self.system.velocity[i] = new_v

    @ti.func
    def kernel(self, fx):
        raise NotImplementedError

class APIC(ParticleInCell):

    def layout(self):
        super().layout()
        # needs to add affine velocity field
        self.system.add_attr("C", dims=(DIM, DIM), dtype=ti.f32)

    @ti.func
    def affine(self, i):
        return self.system.C[i]

    @ti.func
    def p2g(self, i, X):
        affine = self.affine(i)
        # fractional position of particle in the grid stencil w.r.t the lowest-indexed corner
        fx = self.position[i] * self.inv_dx - X 
        w = self.kernel(fx)
        for dX in ti.static(self.kernel.stencil()):
            offset_x = (dX - fx) * self.dx
            weight = 1.0
            for d in ti.static(range(DIM)):
                weight *= w[dX[d]][d]
            self.system.grid_v[X + dX] += weight * (self.mass * self.system.velocity[i] + affine @ offset_x)
            self.system.grid_m[X + dX] += weight * self.mass

    @ti.func
    def g2p(self, i, X):
        fx = self.position[i] * self.inv_dx - X
        w = self.kernel(fx)
        new_v = ti.Vector.zero(ti.f32, DIM)
        new_C = ti.Matrix.zero(ti.f32, DIM, DIM)
        for dX in ti.static(self.kernel.stencil()): # loop over 3x3 grid node neighborhood
            offset_X = dX - fx
            weight = 1.0
            g_v = self.system.grid_v[X + dX]
            for d in ti.static(range(DIM)):
                weight *= w[dX[d]][d]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(offset_X) * self.inv_dx
        self.system.velocity[i] = new_v
        self.system.C[i] = new_C

class Kernel:
    stencil_size = 0
    origin = 0

    @ti.func
    def __call__(self, fx):
        raise NotImplementedError
    
    @ti.func
    def stencil(self):
        return ti.grouped(ti.ndrange(*((self.stencil_size,) * DIM)))

class QuadraticKernel(Kernel):
    origin = 0.5
    stencil_size = 3 # 3x3 stencil for quadratic kernel

    @ti.func
    def __call__(self, fx):
        return [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]