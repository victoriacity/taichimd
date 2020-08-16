import taichi as ti
from .consts import *
from .common import *

@ti.data_oriented
class Grid(Module):

    dynamics = True

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
        return (x * self.inv_dx).cast(ti.i32)

    def clear(self):
        pass

    @ti.func
    def use(self):
        self.clear()
        for i in self.position:
            X = self.grid_index(self.position[i])
            self.p2g(i, X)
        if ti.static(self.dynamics):
            if ti.static(self.grid == None):
                for X in ti.grouped(ti.ndrange(*((self.gridsize,) * DIM))):
                    self.grid_step(X)
            else:
                for X in self.grid:
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

