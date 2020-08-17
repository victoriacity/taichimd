import taichi as ti
from taichimd import Simulation, MDRenderer, DIM
from taichimd.grid import APIC, QuadraticKernel
from taichimd.integrator import ForwardEulerIntegrator

class MPMFluidGrid(APIC):
    @ti.func
    def affine(self, i):
        stress = -self.system.dt * self.vol * (self.system.J[i] - 1) * 4 * self.inv_dx * self.inv_dx * self.E
        return ti.Matrix.identity(ti.f32, DIM) * stress + self.mass * self.system.C[i]
    @ti.func
    def g2p(self, i, X):
        super(MPMGrid, self).g2p(i, X)
        self.system.J[i] *= 1 + self.system.dt * self.system.C[i].trace()
        
ti.init(arch=ti.cuda)
dt = 1e-4
n_particles, n_grid = 65536, 64
p_vol, p_rho = (0.5 / n_grid)**2, 1
mpmgrid = MPMFluidGrid(QuadraticKernel(), n_grid, mass=p_vol * p_rho, gravity=10)
mpmgrid.vol, mpmgrid.E = p_vol, 400
sim = Simulation(n_particles, integrator=ForwardEulerIntegrator(dt, False), grid=mpmgrid)
sim.add_attr("J", dims=(), dtype=ti.f32)
sim.init_random(center=(0.4, 0.4, 0.4), length=0.4)
sim.build() # will only materialize after calling build
sim.velocity.fill([0, -1, 0])
sim.J.fill(1)
sim.run(irender=10)
