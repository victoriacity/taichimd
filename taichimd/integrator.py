import taichi as ti
from .consts import *
from .common import *

@ti.data_oriented
class Integrator(Module):

    requires_force = True
    requires_hessian = False

    def __init__(self, dt, requires_force=None):
        self.dt = dt
        if requires_force is not None:
            self.requires_force = requires_force

    def register(self, system):
        self.system = system
        
    @ti.func
    def prestep(self):
        pass

    @ti.func
    def poststep(self):
        raise NotImplementedError

    '''
    The integrator is NVE by default, temperature
    is set by scaling the velocity of particles
    '''
    @ti.kernel
    def set_temp(self, temperature: ti.template()):
        t_actual = self.system.get_temp()
        for i in self.system.velocity:
            self.system.velocity[i] *= (temperature / t_actual) ** 0.5

@ti.data_oriented
class ForwardEulerIntegrator(Integrator):

    @ti.func
    def poststep(self):
        for i in self.system.position:
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt
            if ti.static(self.requires_force):
                self.system.velocity[i] = self.system.velocity[i] + self.system.force[i] * self.dt



@ti.data_oriented
class VerletIntegrator(Integrator):
    
    @ti.func
    def prestep(self):
        '''
        Integrate the motion of particles. Use Newton's law of motion and 
        verlet integration scheme. Also calculates the kinetic and potential energies.
        '''
        for i in self.system.position:
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt * 0.5
    @ti.func
    def poststep(self):
        for i in self.system.position:
            if ti.static(self.requires_force):
                self.system.velocity[i] = self.system.velocity[i] + self.system.force[i] * self.dt
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt * 0.5

        
@ti.data_oriented
class NVTIntegrator(VerletIntegrator):
    # thermostat "damping" coefficients
    Q1 = 5
    Q2 = 5


    def register(self, system):
        self.temp = ti.field(dtype=ti.f64, shape=())
        self.xi = ti.field(dtype=ti.f64, shape=2)
        self.vxi = ti.field(dtype=ti.f64, shape=2)
        super().register(system)

    def build(self):
        if not hasattr(self.system, "ek"):
            raise SimulationModuleError("NVT integrator requires "
            "computing the kinetic energy! Please add EnergyAnalyzer to the simulation")
        self.xi.fill(0.0)
        self.vxi.fill(0.0)
        self.set_temp(self.system.temperature)

    '''
    Substeps to integrate the thermostat.
    '''
    @ti.func
    def step_vxi1(self, G1):
        self.vxi[0] *= ti.exp(-self.vxi[1] * self.dt * 0.125)
        self.vxi[0] += G1 * self.dt * 0.25
        self.vxi[0] *= ti.exp(-self.vxi[1] * self.dt * 0.125)

    @ti.func
    def step_vxi2(self, G2):
        self.vxi[1] = self.vxi[1] + G2 * self.dt * 0.25

    @ti.func
    def step_xi(self):
        self.xi[0] += self.vxi[0] * self.dt * 0.5
        self.xi[1] += self.vxi[1] * self.dt * 0.5
    
    '''
    Integrate the thermostat by half a time step.
    '''
    @ti.func
    def integrate_thermostat(self):
        G2 = self.Q1 * self.vxi[0] ** 2 - self.temp[None]
        self.step_vxi2(G2)
        G1 = (2 * self.system.ek[None] - 3 * self.system.n_particles * self.temp[None]) / self.Q1
        self.step_vxi1(G1)
        self.step_xi()
        s = ti.exp(-self.vxi[0] * self.dt * 0.5)
        self.system.ek[None] = ti.cast(self.system.ek[None] * s * s, ti.f32)
        G1 = (2 * self.system.ek[None] - 3 * self.system.n_particles * self.temp[None]) / self.Q1
        self.step_vxi1(G1)
        G2 = (self.Q1 * self.vxi[0] ** 2 - self.temp[None]) / self.Q2
        self.step_vxi2(G2)
        for i in self.system.velocity:
            self.system.velocity[i] = ti.cast(self.system.velocity[i] * s, ti.f32)

    def set_temp(self, temperature):
        self.temp.fill(temperature)

    @ti.func
    def prestep(self):
        self.integrate_thermostat()
        VerletIntegrator.prestep(self)

    @ti.func
    def poststep(self):
        VerletIntegrator.poststep(self)
        self.integrate_thermostat()

'''
Implicit integrator solvers
'''
class JacobiSolver:
    requires_hessian = True
    '''
    Jacobi iteration for Ax=b: x_i = 1/a_ii * (b_i - sum(a_ij * x_j))
    for integrating motion:
        v_i,t+1 = (v_i,t + force / m * dt - sum(hessian[i, j] * v_j) * dt ** 2) / (1 + hessian[i, i])
    '''
    @ti.kernel
    def substep(self, dt:ti.f32) -> ti.f32:
        loss = 0.0
        for i in self.vafter:
            self.vafter[i] = self.system.velocity[i] + self.system.force[i] * dt
            self.residuals[i].fill(0)
        for i, j in ti.ndrange(self.system.n_particles, self.system.n_particles):
            if i != j:
                self.vafter[i] -= dt ** 2 * \
                    self.system.hessian[i, j] @ self.vbefore[j]
        for i in self.vafter:
            h = self.system.hessian[i, i] * dt ** 2
            self.vafter[i][0] -= (h[0, 1] * self.vbefore[i][1] + h[0, 2] * self.vbefore[i][2]) 
            self.vafter[i][1] -= (h[1, 0] * self.vbefore[i][0] + h[1, 2] * self.vbefore[i][2])
            self.vafter[i][2] -= (h[2, 0] * self.vbefore[i][0] + h[2, 1] * self.vbefore[i][1])
            for d in ti.static(range(DIM)):
                self.vafter[i][d] = self.vafter[i][d] \
                    / (1 + h[d, d]) 
        for i, j in ti.ndrange(self.system.n_particles, self.system.n_particles): 
            # if i <= j:
            #    print(i, j, self.system.hessian[i, j])
            self.residuals[i] += self.system.hessian[i, j] @ self.vafter[j] * dt ** 2
        for i in range(self.system.n_particles):
            self.vbefore[i] = self.vafter[i]
            self.residuals[i] += self.vafter[i] \
                - self.system.velocity[i] - self.system.force[i] * dt
            loss += (self.residuals[i] ** 2).sum()
        return loss


class FixedPointSolver:

    @ti.kernel
    def substep(self, dt:ti.f32) -> ti.f32:
        loss = 0.0
        for i in self.system.position:
            self.system.position[i] += self.vbefore[i] * dt
        self.system.calc_force()
        for i in self.system.position:
            self.system.position[i] -= self.vbefore[i] * dt
            self.vafter[i] = self.system.velocity[i] + self.system.force[i] * dt
            #print(i, ((self.vafter[i] - self.vbefore[i]) ** 2).sum())
            loss += ((self.vafter[i] - self.vbefore[i]) ** 2).sum()
            self.vbefore[i] = self.vafter[i]
        return loss

@ti.data_oriented
class ImplicitIntegrator(Integrator):
    '''
    Implicit integrators solve for the velocity using
    [I + beta * hessian / m * dt ** 2] @ v_t+1 = v_t + force / m * dt
    Use Jacobi method to solve the linear system until converge to EPS.
    '''
    MAX_ITER = 100
    eps = 1e-9
    

    def __init__(self, system, dt):
        super().__init__(system, dt)
        self.vbefore = ti.Vector.field(DIM, dtype=ti.f32, shape=self.system.n_particles)
        self.vafter = ti.Vector.field(DIM, dtype=ti.f32, shape=self.system.n_particles)
        self.residuals = ti.Vector.field(DIM, dtype=ti.f32, shape=self.system.n_particles)

    @ti.func   
    def solve_velocity(self, dt):
        for i in self.system.velocity:
            self.vbefore[i] = self.system.velocity[i]
        for niter in range(self.MAX_ITER):
            res = self.substep(dt)
            #print(res)
            if res < self.eps:
                break

    @ti.func
    def integrate_motion(self):
        raise NotImplementedError
    
    @ti.func
    def prestep(self):
        if self.requires_hessian:
            self.system.hessian.fill(0)

    @ti.func
    def poststep(self):
        self.solve_velocity(self.dt)
        self.integrate_motion()

@ti.data_oriented
class ImplicitMidpointIntegrator(ImplicitIntegrator):

    @ti.func
    def prestep(self):
        super().prestep()
        for i in self.system.position:
            self.system.velocity[i] = self.vafter[i]
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt * 0.5

    @ti.func
    def integrate_motion(self):
        for i in self.system.position:
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt * 0.5
            self.system.velocity[i] += self.system.force[i] * self.dt * 0.5

    @ti.func
    def poststep(self):
        self.solve_velocity(self.dt / 2)
        self.integrate_motion()


@ti.data_oriented
class BackwardEulerIntegrator(ImplicitIntegrator):

    @ti.kernel
    def integrate_motion(self):
        for i in self.system.position:
            self.system.velocity[i] = self.vafter[i]
            self.system.position[i] = self.system.position[i] \
                        + self.system.velocity[i] * self.dt



class MidpointJacobiIntegrator(ImplicitMidpointIntegrator, JacobiSolver): pass
class MidpointFixedIntegrator(ImplicitMidpointIntegrator, FixedPointSolver): pass
class BackwardJacobiIntegrator(BackwardEulerIntegrator, JacobiSolver): pass
class BackwardFixedIntegrator(BackwardEulerIntegrator, FixedPointSolver): pass
