import taichi as ti
from taichimd import Simulation, MDRenderer, DIM
from taichimd.grid import APIC, QuadraticKernel
from taichimd.integrator import ForwardEulerIntegrator as FE
MDRenderer.radius = 0.2

class MPMGrid(APIC):
    @ti.func
    def affine(self, i):
        self.system.F[i] = (ti.Matrix.identity(ti.f32, DIM) + self.system.dt * self.system.C[i]) @ self.system.F[i]
        h = 0.3 if self.system.type[i] == 1 else ti.exp(10 * (1.0 - self.system.Jp[i]))
        la = self.lambda_0 * h
        U, sig, V = ti.svd(self.system.F[i])
        J = 1.0
        for d in ti.static(range(DIM)):
            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3) if self.system.type[i] == 2 else sig[d, d]
            self.system.Jp[i] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if self.system.type[i] == 0:  # Reset deformation gradient to avoid numerical instability
            self.system.F[i] = ti.Matrix.identity(ti.f32, DIM)
            self.system.F[i][0, 0] = J
        elif self.system.type[i] == 2:
            self.system.F[i] = U @ sig @ V.transpose() # Reconstruct elastic deformation gradient after plasticity
        stress = ti.Matrix.identity(ti.f32, DIM) * la * J * (J - 1)
        if self.system.type[i] != 0:
            stress += 2 * self.mu_0 * h * (self.system.F[i] - U @ V.transpose()) @ self.system.F[i].transpose()
        stress = (-self.system.dt * self.vol * 4 * self.inv_dx * self.inv_dx) * stress
        return stress + self.mass * self.system.C[i]

ti.init(arch=ti.cuda, device_memory_GB=3)
dt = 2e-4
n_particles, n_grid = 32768, 64
p_vol, p_rho = (0.5 / n_grid)**2, 1
mpmgrid = MPMGrid(QuadraticKernel(), n_grid, mass=p_vol * p_rho, gravity=50)
mpmgrid.vol = p_vol
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mpmgrid.mu_0, mpmgrid.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
sim = Simulation(n_particles, integrator=FE(dt, False), grid=mpmgrid)
sim.add_attr("F", dims=(DIM, DIM), dtype=ti.f32) # affine velocity field, deformation gradient
sim.add_attr("Jp", dims=(), dtype=ti.f32) # plastic deformation
sim.gui.set_colors([[6/255, 133/255, 135/255], [237/255, 85/255, 89/255], [238/255, 238/255, 240/255]]) # mpm99 colors
sim.init_random(center=(0.4, 0.15, 0.7), length=0.2, start=0, end=n_particles//3, inittype=0)
sim.init_random(center=(0.4, 0.45, 0.7), length=0.2, start=n_particles//3, end=2*n_particles//3, inittype=1)
sim.init_random(center=(0.5, 0.75, 0.7), length=0.2, start=2*n_particles//3, end=n_particles, inittype=2)
sim.build()
sim.velocity.fill(0); sim.F.fill(((1,0,0),(0,1,0),(0,0,1))); sim.C.fill(0); sim.Jp.fill(1)
sim.run(irender=10, pause=True)