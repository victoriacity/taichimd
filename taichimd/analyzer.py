import taichi as ti
from .common import Module
from .ui import Printer

class Analyzer(Module):

    def use(self):
        pass

@ti.data_oriented
class EnergyAnalyzer(Analyzer):

    def register(self, system):
        system.add_var("ek")
        self.ek = system.ek
        self.has_ep = hasattr(system, "ep")
        if ti.static(system.gui is not None):
            system.gui.add_component(Printer("Internal energy", self.energy))
        return super().register(system)

    @ti.func
    def use(self):
        self.ek[None] = 0.0
        for i in self.system.velocity:
            self.ek[None] += self.system.velocity[i].norm_sqr() / 2

    @ti.kernel
    def calculate_energy(self) -> ti.f32:
        if ti.static(self.has_ep):
            self.system.calc_force()
        self.use()
        return self.energy()

    @ti.pyfunc
    def energy(self):
        if ti.static(self.has_ep):
            return self.system.ek[None] + self.system.ep[None]
        else:
            return self.ek[None]


class MomentumAnalyzer(Analyzer):

    pass
