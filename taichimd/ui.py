import taichi as ti
import numpy as np
from .consts import *

WINDOW_SIZE = 1280
rgb2hex = lambda x: x[0].astype(np.int) * 65536 + x[1].astype(np.int) * 256 + x[2].astype(np.int)
alpha = lambda c1, c2, z: rgb2hex(np.outer(c1, 1 - z) + np.outer(c2, z))


class UIComponent:

    height = 0.05
    fontsize = 36

    def __init__(self):
        self.window = None
        self.pos = (0, 0)

    def show(self):
        raise NotImplementedError


class TemperatureControl(UIComponent):

    height = 0.1

    def show(self):
        system = self.window.system
        gui = self.window.gui
        t_actual = system.get_temp()
        t_set = system.temperature
        if gui.event is not None:
            if gui.event.type == ti.GUI.PRESS and gui.event.key == ti.GUI.UP:
                t_set = round(t_set + 0.1, 1)
                system.set_temp(t_set)
            elif gui.event.type == ti.GUI.PRESS and gui.event.key == ti.GUI.DOWN:
                if t_set <= 0.15:
                    t_set /= 2
                else:
                    t_set = round(t_set - 0.1, 1)
                system.set_temp(t_set)
            gui.event = None
        color_t = 0xf56060 if abs(t_actual - t_set) > 0.02 else 0x74e662
        gui.text("T_set = %.3g" % t_set, (self.pos[0], self.pos[1] + 0.05), font_size=self.fontsize, color=0xffffff)
        gui.text("T_actual = %.3g" % t_actual, self.pos, font_size=self.fontsize, color=color_t)


class Printer(UIComponent):

    def __init__(self, name, callback, fmt="%.3f"):
        self.name = name
        self.callback = callback
        self.fmt = fmt
        super().__init__()

    def show(self):
        value = self.fmt % self.callback()
        self.window.gui.text("%s = %s" % (self.name, value), self.pos, font_size=self.fontsize)


class GUI:

    left = 0.05
    bottom = 0.1

    def __init__(self, system, renderer):
        self.system = system
        self.gui = ti.GUI("MD", res=WINDOW_SIZE)
        self.renderer = renderer(self.system, self.gui)
        self.play = True
        self.ycur = self.bottom
        self.components = []

    def add_component(self, comp):
        comp.window = self
        comp.pos = (self.left, self.ycur)
        self.components.append(comp)
        self.ycur += comp.height

    def set_colors(self, colors):
        self.renderer.set_colors(colors)
        

    def show(self, savefile=None):
        self.gui.get_event()
        self.gui.running = not self.gui.is_pressed(ti.GUI.ESCAPE)
        if self.gui.event is not None and self.gui.event.type == ti.GUI.PRESS\
                and self.gui.event.key == ti.GUI.SPACE:
            self.play = not self.play
            self.gui.event = None
        self.renderer.render()
        for comp in ti.static(self.components):
            comp.show()
        self.gui.show(savefile)
        if not self.gui.running:
            exit()
        return self.play
        


class Renderer:

    colors = None

    def __init__(self, system, gui):
        self.system = system
        self.gui = gui

    def render(self, positions):
        raise NotImplementedError

    def set_colors(self, colors):
        pass

class CanvasRenderer(Renderer):
    camera = 1
    radius = 2
    bg = np.array([17, 47, 65])
    circ = np.array([122, 200, 225])
    bond = np.array([62, 165, 45])
    
    def render(self):
        self.gui.clear(rgb2hex(self.bg))
        positions = self.system.position.to_numpy() \
            / self.system.boxlength
        if DIM == 2:
            z = np.ones(positions.shape[0])
        else:
            z = positions[:, 2]
        xy = positions[:, :2]
        z_order = np.argsort(z)
        sizes = self.radius * self.camera / (self.camera + (1 - z))
        z /= np.max(z)
        colors = alpha(self.bg, self.circ, z)
        if ti.static(not self.system.is_atomic):
            bonds = self.system.forcefield.bond.to_numpy()
            bonds = bonds[bonds[:, 0] > 0]
            bonds_a, bonds_b = xy[bonds[:, 1]], xy[bonds[:, 2]]
            mask = np.sum((bonds_a - bonds_b) ** 2, axis=1) < 0.5 ** 2
            bond_colors = alpha(self.bg, self.bond, (z[bonds[mask, 1]] + z[bonds[mask, 2]]) / 2)
            self.gui.lines(bonds_a[mask, :], bonds_b[mask, :], color=bond_colors, radius=self.radius / 3)
        self.gui.circles(xy[z_order], radius=sizes[z_order], color=colors[z_order])

try:
    import taichi_three as t3
    from taichimd.graphics import MolecularModel, FalloffLight
    class T3RendererBase(Renderer, t3.common.AutoInit):
        radius = 0.15

        def __init__(self, system, gui):
            
            super().__init__(system, gui)
            boxlength = system.boxlength
            if hasattr(self.system.forcefield, "nonbond_params_d"):
                epsilon = self.system.forcefield.nonbond_params_d[1][0]
            else:
                epsilon = boxlength / 50
            self.radius = T3RendererBase.radius * epsilon
            self.scene = self._scene()
            self.camera = t3.Camera(res=(WINDOW_SIZE, WINDOW_SIZE), pos=[boxlength/2, boxlength/2, -boxlength], 
                            target=[boxlength/2, boxlength/2, boxlength/2], up=[0, 1, 0])
            self.scene.add_camera(self.camera)   

        def render(self):
            self.init()
            self.camera.from_mouse(self.gui)
            self.scene.render()
            self.gui.set_image(self.camera.img)

        def _scene(self):
            raise NotImplementedError

        def _init(self):
            raise NotImplementedError

        def _render(self):
            raise NotImplementedError


    class MDRenderer(T3RendererBase):

        def _scene(self):
            scene = t3.Scene()
            shader = t3.Shading(lambert=0.8, blinn_phong=0.3, shineness=5)
            scene.opt = shader
            self.model = MolecularModel(radius=self.radius)
            scene.add_model(self.model)
            l = self.system.boxlength
            light = FalloffLight(direction=[l, -l, 2 * l],
                target=[l/2, l/2, l/2],
                c1=0.1 / l ** 2,
                c2=0.1 / l ** 3)
            scene.add_light(light)
            return scene

        def set_colors(self, colors):
            MolecularModel.colors = colors

        def _init(self):
            self.model.register(self.system)

    class RTRenderer(T3RendererBase):
        def _scene(self):
            scene = t3.SceneRT()
            self.radius_var = ti.var(ti.f32, self.system.n_particles)
            return scene

        def _init(self):
            self.scene.set_light_dir([1, -1, 6])
            self.radius_var.fill(self.radius)
            self.scene.add_ball(self.system.position, self.radius_var)        

except ImportError as e:
    print(e)
    print("Taichi_three (>= 0.0.3) not found, only canvas renderer is available.")
    MDRenderer = CanvasRenderer
    RTRenderer = CanvasRenderer
        