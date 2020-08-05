import taichi as ti
import numpy as np

WINDOW_SIZE = 1024
rgb2hex = lambda x: x[0].astype(np.int) * 65536 + x[1].astype(np.int) * 256 + x[2].astype(np.int)
alpha = lambda c1, c2, z: rgb2hex(np.outer(c1, 1 - z) + np.outer(c2, z))


class GUI:

    def __init__(self, system, renderer):
        self.system = system
        self.gui = ti.GUI("MD", res=WINDOW_SIZE)
        self.renderer = renderer(self.gui)

    def show_temperature(self):
        t_actual = self.system.get_temp()
        t_set = self.system.temperature
        while self.gui.get_event(ti.GUI.PRESS):
            if self.gui.event.key == ti.GUI.ESCAPE:
                exit()
            if self.gui.event.key == ti.GUI.UP:
                t_set = round(t_set + 0.1, 1)
                self.system.set_temp(t_set)
            if self.gui.event.key == ti.GUI.DOWN:
                if t_set <= 0.15:
                    t_set /= 2
                else:
                    t_set = round(t_set - 0.1, 1)
                self.system.set_temp(t_set) 
        color_t = 0xf56060 if abs(t_actual - t_set) > 0.02 else 0x74e662
        self.gui.text("T_set = %.3g" % t_set, (0.05, 0.25), font_size=36, color=0xffffff)
        self.gui.text("T_actual = %.3g" % t_actual, (0.05, 0.2), font_size=36, color=color_t)


    def show(self, savefile=None):
        if ti.static(not self.system.is_atomic):
            bonds = self.system.forcefield.bond.to_numpy()
        else:
            bonds = None
        self.renderer.render(self.system.position.to_numpy() \
            / self.system.boxlength, bonds)
        if self.system.temperature > 0:
            self.show_temperature()
        vcm = np.sum(self.system.velocity.to_numpy(), axis=0)
        self.gui.text("Internal energy = %.3f" % (self.system.energy()), (0.05, 0.15), font_size=36)
        self.gui.text("Momentum norm = %.3f" % np.sum(vcm ** 2), (0.05, 0.1), font_size=36)
        self.gui.show(savefile)
        


class Renderer:

    def __init__(self, gui):
        self.gui = gui

    def render(self, positions):
        raise NotImplementedError

class CanvasRenderer(Renderer):
    camera = 1
    radius = 8
    bg = np.array([17, 47, 65])
    circ = np.array([122, 200, 225])
    bond = np.array([62, 165, 45])

    def render(self, positions, bonds):
        self.gui.clear(rgb2hex(self.bg))   
        z = positions[:, 2]
        xy = positions[:, :2]
        z_order = np.argsort(z)
        sizes = self.radius * self.camera / (self.camera + (1 - z))
        z /= np.max(z)
        colors = alpha(self.bg, self.circ, z)
        if bonds is not None:
            bonds = bonds[bonds[:, 0] > 0]
            bonds_a, bonds_b = xy[bonds[:, 1]], xy[bonds[:, 2]]
            mask = np.sum((bonds_a - bonds_b) ** 2, axis=1) < 0.5 ** 2
            bond_colors = alpha(self.bg, self.bond, (z[bonds[mask, 1]] + z[bonds[mask, 2]]) / 2)
            self.gui.lines(bonds_a[mask, :], bonds_b[mask, :], color=bond_colors, radius=self.radius / 3)
        self.gui.circles(xy[z_order], radius=sizes[z_order], color=colors[z_order])