import taichi as ti
from consts import *

class HessianNotImplemented(NotImplementedError):
    pass

class ForceNotImplemented(NotImplementedError):
    pass


class ExternalPotential:

    def __call__(self, r):
        pass

    def force(self, r):
        raise ForceNotImplemented

    def hessian(self, r):
        raise HessianNotImplemented


class QuadraticWell(ExternalPotential):

    def __init__(self, k, center):
        self.k = k
        self.center = ti.Vector(center)

    def __call__(self, r):
        return self.k * ((r - self.center) ** 2).sum() / 2

    def force(self, r):
        return -self.k * (r - self.center)

    def hessian(self, r):
        return self.k * IDENTITY

class InverseSquare(ExternalPotential):

    def __init__(self, k, center):
        self.k = k
        self.center = center

    def __call__(self, r):
        return self.k / (r - self.center).norm()

    def force(self, r):
        dr = r - self.center
        dr2 = (dr ** 2).sum()
        return -self.k * dr / dr.norm() / dr2


'''
Pair interactions
'''
class PairInteraction:

    '''
    singleton class to denote zero,
    used to avoid repeating updates of 
    force or hessian.
    '''
    class Zero: pass
    ZERO = Zero()
    
    def __call__(self, r2):
        raise NotImplementedError

    def derivative(self, r2):
        raise ForceNotImplemented

    def second_derivative(self, r2):
        raise HessianNotImplemented

    @ti.func
    def force(self, r, r2):
        return 2. * self.derivative(r2) * r
    
    @ti.func
    def hessian(self, r, r2):
        return -4. * r.outer_product(r) * self.second_derivative(r2) \
            - 2 * IDENTITY * self.derivative(r2)

class LennardJones(PairInteraction):

    def __init__(self, sigma, epsilon, rcut, rcutin=0):
        self.s6 = sigma ** 6
        self.s12 = self.s6 ** 2
        self.e = epsilon
        self.ecut = 4. * self.e * \
             ((sigma / rcut) ** 12 - (sigma / rcut) ** 6)
        self.rc2 = rcut ** 2
        self.rin2 = rcutin ** 2

    @ti.func
    def __call__(self, r2):
        u = 0.0
        if self.rin2 < r2 < self.rc2:
            u = 4. * self.e * (self.s12 / r2 ** 6 - self.s6 / r2 ** 3) - self.ecut
        return u

    @ti.func
    def derivative(self, r2):
        return 12. * self.e / r2 \
            * (-2. * self.s12 / r2 ** 6 + self.s6 / r2 ** 3) 
    @ti.func
    def second_derivative(self, r2):
        return 24. * self.e / r2 ** 2 \
            * (7. * self.s12 / r2 ** 6 - 2. * self.s6 / r2 ** 3)

class Coulomb(PairInteraction):
    
    def __init__(self, k):
        self.k = k
    
    @ti.func
    def __call__(self, r2):
        return self.k / ti.sqrt(r2)

    @ti.func
    def derivative(self, r2):
        r = ti.sqrt(r2)
        return -self.k / (2 * r * r2)

    @ti.func
    def second_derivative(self, r2):
        r = ti.sqrt(r2)
        return 3. * self.k / (4 * r * r2 * r2)


class HarmonicPair(PairInteraction):
    def __init__(self, k, r0):
        self.k = k
        self.r0 = r0

    @ti.func
    def __call__(self, r2):
        r = ti.sqrt(r2)
        return self.k * (r - self.r0) ** 2 / 2

    @ti.func
    def derivative(self, r2):
        return self.k * (1 - self.r0 / ti.sqrt(r2)) / 2

    @ti.func
    def second_derivative(self, r2):
        r = ti.sqrt(r2)
        return self.k * self.r0 / (2 * r * r2)


class ParabolicPotential(PairInteraction):

    def __init__(self, k):
        self.k = k

    @ti.func
    def __call__(self, r2):
        r = ti.sqrt(r2)
        return self.k * r ** 2 / 2

    @ti.func
    def derivative(self, r2):
        return self.k / 2

    @ti.func
    def second_derivative(self, r2):
        return 0.