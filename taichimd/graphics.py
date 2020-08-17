import taichi as ti
import taichi_three as t3
import taichi_glsl as ts
from .consts import *
import math
import numpy as np

@ti.data_oriented
class MolecularModel(t3.common.AutoInit):

    colors = COLOR_MOLECULES

    AO = False

    def __init__(self, radius):
        self.L2W = t3.transform.Affine.var(())
        self.radius = radius
        self.particles = None
        self.box = ti.Vector(3, ti.f32, (16, ))
        self.colors = ti.Vector(3, ti.f32, (6,))

    def register(self, system):
        self.system = system
        self.particles = system.position
        self.boxlength = system.boxlength
        
    def _init(self):
        self.L2W.init()
        box = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                    [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1],
                    [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]]
        self.box.from_numpy(np.array(box) * self.boxlength)
        self.colors.from_numpy(np.array(MolecularModel.colors))

    @ti.func
    def render(self, camera):
        # particles
        for i in ti.grouped(self.particles):
            render_particle(self, camera, self.particles[i], self.radius, self.colors[self.system.type[i]])
        # bonds
        if ti.static(not self.system.is_atomic and self.system.forcefield.bonded != None):
            for x in range(self.system.forcefield.nbond):
                i, j = self.system.forcefield.bond[x][1], self.system.forcefield.bond[x][2]
                if (self.particles[i] - self.particles[j]).norm_sqr() < (0.5 * self.boxlength) ** 2:
                    render_cylinder(self, camera, self.particles[i], self.particles[j], self.radius * 0.6,
                        self.colors[self.system.type[i]], self.colors[self.system.type[j]])
        # simulation box
        for i in ti.static(range(4)):
            render_line(camera, self.box[i], self.box[(i + 1) % 4])
        for i in ti.static(range(4, 8)):
            render_line(camera, self.box[i], self.box[4 if i == 7 else i + 1])
        for i in ti.static(range(8, 16, 2)):
            render_line(camera, self.box[i], self.box[i + 1])
        # postprocessing
        self.postprocess(camera)

    @ti.func
    def postprocess(self, camera):
        for X in ti.grouped(camera.img):
            # zbuf == 0 means nothing was drawn at all
            if camera.zbuf[X] > 0:
                ao = 0.0
                if ti.static(self.AO):                
                    ao = ao_kernel(camera.zbuf, X, 10)
                    
                # gamma = 2
                camera.img[X] = ti.sqrt(camera.img[X] * (1 - ao))


'''
A light subclass to implement directional lights with attenuation,
assume that the light is located at -self.dir = (x0, y0, z0).
'''
@ti.data_oriented
class FalloffLight(t3.Light):

    c1 = 0.2
    c2 = 0.2

    def __init__(self, direction=None, target=None, color=None,
            c1=None, c2=None, follow_camera=True):
        if c1 is not None: 
            self.c1 = c1
        if c2 is not None: 
            self.c2 = c2
        self.follow_camera = follow_camera
        self.target_py = target or [0, 0, 0]
        self.lightdist_py = math.sqrt(sum(x ** 2 for x in direction))
        self.target = ti.Vector(3, ti.float32, ())
        self.viewtarget = ti.Vector(3, ti.float32, ())
        self.lightdist = ti.var(ti.float32, ())
        super().__init__(direction, color)

    def _init(self):
        super()._init()
        self.target[None] = self.target_py
        self.lightdist[None] = self.lightdist_py

    @ti.func
    def set_view(self, camera):
        if ti.static(self.follow_camera):
            self.viewdir[None] = self.dir[None]
            self.viewtarget[None] = camera.untrans_pos(self.target[None])
            self.lightdist[None] = (camera.target - camera.pos).norm()
        else:
            self.viewdir[None] = camera.untrans_dir(self.dir[None])
            self.viewtarget[None] = camera.untrans_pos(self.target[None])

    @ti.func
    def intensity(self, pos):
        '''
        Calculate the distance from the object to the light
        as the distance to the normal plane of the light position,
        i.e., x0x+y0y+z0z=0.
        '''
        dist = self.lightdist[None] - ti.dot(pos - self.viewtarget[None], self.viewdir[None])
        f = 0.0
        if dist > 0:
            f = 1. / (1. + self.c1 * dist + self.c2 * dist ** 2)
        return f        


@ti.func
def uncook3(camera, pos):
    if ti.static(camera.type == camera.ORTHO):
        pos[0] *= camera.intrinsic[None][0, 0] 
        pos[1] *= camera.intrinsic[None][1, 1]
        pos[0] += camera.intrinsic[None][0, 2]
        pos[1] += camera.intrinsic[None][1, 2]
    else:
        pos = camera.intrinsic[None] @ pos
        pos[0] /= abs(pos[2])
        pos[1] /= abs(pos[2])
    return pos

@ti.func
def cook_coord(camera, pos):
    if ti.static(camera.type == camera.ORTHO):
        pos[0] -= camera.intrinsic[None][0, 2]
        pos[1] -= camera.intrinsic[None][1, 2]
        pos[0] /= camera.intrinsic[None][0, 0] 
        pos[1] /= camera.intrinsic[None][1, 1]
    else:
        pos[0] = (pos[0] - camera.cx) * pos[2] / camera.fx
        pos[1] = (pos[1] - camera.cy) * pos[2] / camera.fy
    return pos
 
ambient = 0.05

@ti.func
def ao_kernel(arr, X, h):
    ao = 0.0
    for i, j in ti.ndrange((1, h + 1), (1, h + 1)):
        tan = (arr[X.x + i, X.y + j] + arr[X.x - i, X.y - j] - 2 * arr[X])\
            / 2 / ti.sqrt(i**2 + j **2)
        ao += tan
    ao /= h ** 2
    # pick a 45-degree angle for occlusion cutoff
    if ao > 1:
        ao = 0.0
    else:
        ao = min(max(0, ao * 100), 1)
    return ao


@ti.func
def render_particle(model, camera, vertex, radius, basecolor):
    scene = model.scene
    L2W = model.L2W
    a = camera.untrans_pos(L2W @ vertex)
    A = camera.uncook(a)

    bx = camera.fx * radius / a.z
    by = camera.fy * radius / a.z
    M = A.cast(int)
    N = A.cast(int)
    M.x -= bx
    N.x += bx
    M.y -= by
    N.y += by

    M.x, N.x = min(max(M.x, 0), camera.img.shape[0]), min(max(N.x, 0), camera.img.shape[1])
    M.y, N.y = min(max(M.y, 0), camera.img.shape[0]), min(max(N.y, 0), camera.img.shape[1])
    if (M.x < N.x and M.y < N.y):
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            W = ti.cast(ti.Vector([X.x, X.y, a.z]), ti.f32)
            w = cook_coord(camera, W)
            dw = w - a
            dw2 = dw.norm_sqr()

            if dw2 > radius**2:
                continue
            dz = ti.sqrt(radius**2 - dw2)
            n = ti.Vector([dw.x, dw.y, -dz])
            zindex = 1 / (a.z - dz)
            if zindex >= ti.atomic_max(camera.zbuf[X], zindex):
                color = ts.vec3(ambient)
                for light in ti.static(scene.lights):
                    light_color = scene.opt.render_func(a + n, ts.normalize(n), \
                        ts.normalize(a + n), light) * basecolor
                    color += light_color
                camera.img[X] = color

@ti.func
def render_cylinder(model, camera, v1, v2, radius, c1, c2):
    scene = model.scene
    L2W = model.L2W
    a = camera.untrans_pos(L2W @ v1)
    b = camera.untrans_pos(L2W @ v2)
    A = camera.uncook(a)
    B = camera.uncook(b)
    bx = int(ti.ceil(camera.fx * radius / min(a.z, b.z)))
    by = int(ti.ceil(camera.fy * radius / min(a.z, b.z)))

    M, N = int(ti.floor(min(A, B))), int(ti.ceil(max(A, B)))
    M.x -= bx
    N.x += bx
    M.y -= by
    N.y += by
    M.x, N.x = min(max(M.x, 0), camera.img.shape[0]), min(max(N.x, 0), camera.img.shape[1])
    M.y, N.y = min(max(M.y, 0), camera.img.shape[0]), min(max(N.y, 0), camera.img.shape[1])
    if (M.x < N.x and M.y < N.y):
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            t = ti.dot(X - A, B - A) / (B - A).norm_sqr()
            if t < 0 or t > 1:
                continue
            proj = a * (1 - t) + b * t
            W = ti.cast(ti.Vector([X.x, X.y, proj.z]), ti.f32)
            w = cook_coord(camera, W)
            dw = w - proj
            dw2 = dw.norm_sqr()
            if dw2 > radius**2:
                continue
            dz = ti.sqrt(radius**2 - dw2)
            n = ti.Vector([dw.x, dw.y, -dz])
            zindex = 1 / (proj.z - dz)

            if zindex >= ti.atomic_max(camera.zbuf[X], zindex):
                basecolor = c1 if t < 0.5 else c2
                color = ts.vec3(ambient)
                for light in ti.static(scene.lights):
                    light_color = scene.opt.render_func(a + n, ts.normalize(n), \
                        ts.normalize(a + n), light) * basecolor
                    color += light_color
                camera.img[X] = color

@ti.func
def render_line(camera, p1, p2):
    scene = camera.scene
    a = camera.untrans_pos(p1)
    b = camera.untrans_pos(p2)
    A = camera.uncook(a)
    B = camera.uncook(b)
    # dda algorithm
    step = abs((B - A)).max()
    delta = (B - A) / step
    dz = (b.z - a.z) / step
    # needs to do screen clipping
    for i in range(step):
        X = int(A + i * delta)
        if X.x < 0 or X.x >= camera.res[0] or X.y < 0 or X.y >= camera.res[1]:
            continue
        zindex = 1 / (a.z + i * dz)    
        if zindex >= ti.atomic_max(camera.zbuf[X], zindex):
            camera.img[X] = ts.vec3(0, 0.7, 0)



