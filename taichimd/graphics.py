import taichi as ti
import taichi_glsl as ts
from .consts import *
from .grid import NeighborList
from taichimd import t3mini as t3
import math
import numpy as np

@ti.data_oriented
class MolecularModel(t3.AutoInit):

    colors = COLOR_MOLECULES
    sky_color = [0.8, 0.9, 0.95]
    floor_color = [1.0, 1.0, 1.0]

    AO = False
    gi_cutoff = 0.1

    def __init__(self, radius):
        self.radius = radius
        self.particles = None
        self.box = ti.Vector(3, ti.f32, (16, ))
        self.colors = ti.Vector(3, ti.f32, (6,))
        self.enable_gi_py = False
        self.enable_gi = ti.field(ti.i32, ())

    def register(self, system):
        self.system = system
        self.particles = system.position
        self.type = system.type
        self.grid = None
        if hasattr(system, "grid_particles"):
            for g in system.grids:
                if isinstance(g, NeighborList):
                    self.grid = g
            self.grid_n = system.grid_n_particles
            self.grid_list = system.grid_particles
        else:
            self.grid = None
        self.boxlength = system.boxlength
        
    def _init(self):
        box = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                    [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1],
                    [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 1]]
        self.box.from_numpy(np.array(box) * self.boxlength)
        self.colors.from_numpy(np.array(MolecularModel.colors))
        self.enable_gi[None] = self.enable_gi_py

    def toggle_gi(self):
        self.enable_gi[None] = 1 - self.enable_gi[None]

    def get_enable_gi(self):
        return bool(self.enable_gi[None])

    @ti.func
    def render(self, camera):
        self.render_floor(camera)
        # particles
        for i in ti.grouped(self.particles):
            render_particle(self, camera, self.particles[i], self.radius, self.colors[self.system.type[i]])
        # bonds
        if ti.static(not self.system.is_atomic and self.system.forcefield.bonded != None):
            self.render_bonds(camera)
        # simulation box
        self.draw_simulation_box(camera)
        # postprocessing
        self.calc_gi(camera)
        self.postprocess(camera)
        
    @ti.func
    def render_floor(self, camera):
        scene = camera.scene
        c_floor = ts.vec3(*self.floor_color)
        c_sky = ts.vec3(*self.sky_color)
        for X in ti.grouped(camera.img):
            W = ti.cast(ti.Vector([X.x, X.y, 1]), ti.f32)
            x = cook_coord(camera, W)
            world_dir = camera.trans_dir(ts.normalize(x))
            if world_dir[1] > 0 or camera.pos[None][1] < floor_y:
                camera.img[X] = c_sky
            else:
                f = min(1, -world_dir[1] * 50)
                n = camera.untrans_dir(ts.vec3(0, 1, 0))
                pos = camera.pos[None] + world_dir * (camera.pos[None][1] - floor_y) / abs(world_dir[1])
                #print(pos)
                pos = camera.untrans_pos(pos)
                #print(pos)
                color = ambient * c_sky * c_floor
                for light in ti.static(scene.lights):
                    light_color = scene.opt.render_func(pos, n, \
                        ts.normalize(pos), light, c_floor)
                    color += light_color
                camera.img[X] = c_sky * (1 - f) + color * f
                camera.nbuf[X] = n
                camera.zbuf[X] = 1 / pos.z

    @ti.func
    def render_bonds(self, camera):
        for x in range(self.system.forcefield.nbond):
            i, j = self.system.forcefield.bond[x][1], self.system.forcefield.bond[x][2]
            if (self.particles[i] - self.particles[j]).norm_sqr() < (0.5 * self.boxlength) ** 2:
                render_cylinder(self, camera, self.particles[i], self.particles[j], self.radius * 0.6,
                    self.colors[self.system.type[i]], self.colors[self.system.type[j]])

    @ti.func
    def draw_simulation_box(self, camera):
        for i in ti.static(range(4)):
            render_line(camera, self.box[i], self.box[(i + 1) % 4])
        for i in ti.static(range(4, 8)):
            render_line(camera, self.box[i], self.box[4 if i == 7 else i + 1])
        for i in ti.static(range(8, 16, 2)):
            render_line(camera, self.box[i], self.box[i + 1])

    @ti.func
    def calc_gi(self, camera): 
        scene = camera.scene
        for X in ti.grouped(camera.img):
            if self.enable_gi[None] == 0:
                continue
            # use an additional normal buffer at this time        
            if camera.zbuf[X] > 0:
                z = 1 / camera.zbuf[X]
                pos = cook_coord(camera, ti.cast(ti.Vector([X.x, X.y, z]), ti.f32))
                pos_world = camera.trans_pos(pos)
                norm_world = camera.trans_dir(camera.nbuf[X])
                dy = pos_world[1] - floor_y
                ftot = 0.0
                if dy > 0:
                    ftot = form_factor_floor(norm_world, dy, self.radius)
                gtot = ts.vec3(0.0)
                if ti.static(self.grid is not None):
                    base = self.grid.grid_index(pos_world)
                    for dI in ti.grouped(ti.ndrange((-4, 5), (-4, 5), (-4, 5))):
                        I = max(0, min(base + dI, self.grid.gridsize - 1))
                        for p in range(self.grid_n[I + dI]):
                            i = self.grid_list[I + dI, p]
                            if (self.particles[i] - pos_world).norm_sqr() \
                                    > (self.gi_cutoff * self.boxlength) ** 2:
                                continue
                            f, gi = get_gi_factors(camera, scene, self.particles[i], pos_world, 
                            norm_world, self.radius, self.colors[self.type[i]])
                            ftot += f
                            gtot += gi
                else:
                    for i in range(self.particles.shape[0]):
                        # don't compute for particles behind the normal
                        if ti.dot(self.particles[i] - pos_world, norm_world) < 0:
                            continue
                        f, gi = get_gi_factors(camera, scene, self.particles[i], pos_world, 
                            norm_world, self.radius, self.colors[self.type[i]])
                        ftot += f
                        gtot += gi
                camera.img[X] = camera.img[X] * ts.vec3(max(1 - ftot, 0)) + gtot * 0.3

    @ti.func
    def postprocess(self, camera):
        for X in ti.grouped(camera.img):
            # gamma = 2
            camera.img[X] = ti.sqrt(camera.img[X])


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
 

@ti.func
def render_particle(model, camera, vertex, radius, basecolor):
    scene = model.scene
    a = camera.untrans_pos(vertex)
    A = camera.uncook(a)

    bx = camera.fx * radius / a.z
    by = camera.fy * radius / a.z
    M = A
    N = A
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
                normal = ts.normalize(n)
                view = ts.normalize(a + n)
                color = get_ambient(camera, normal, view) * basecolor
                for light in ti.static(scene.lights):
                    light_color = scene.opt.render_func(a + n, normal, view, light, basecolor)
                    color += light_color
                camera.img[X] = color
                camera.nbuf[X] = normal

@ti.func
def render_cylinder(model, camera, v1, v2, radius, c1, c2):
    scene = model.scene
    a = camera.untrans_pos(v1)
    b = camera.untrans_pos(v2)
    A = camera.uncook(a)
    B = camera.uncook(b)
    bx = int(ti.ceil(camera.fx * radius / min(a.z, b.z)))
    by = int(ti.ceil(camera.fy * radius / min(a.z, b.z)))

    M, N = ti.floor(min(A, B)), ti.ceil(max(A, B))
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
                normal = ts.normalize(n)
                view = ts.normalize(a + n)
                color = get_ambient(camera, normal, view) * basecolor
                for light in ti.static(scene.lights):
                    light_color = scene.opt.render_func(a + n, normal, \
                        view, light, basecolor)
                    color += light_color
                camera.img[X] = color
                camera.nbuf[X] = normal

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

floor_y = 0.01
ambient = 0.15

@ti.func
def get_ambient(camera, normal, view):
    c_floor = ts.vec3(0.75)
    c_sky = ts.vec3(0.8, 0.9, 0.95)
    refl_dir = -ts.reflect(-view, normal)
    refl_dir = ts.normalize(camera.trans_dir(refl_dir))
    color = ts.vec3(ambient)
    if refl_dir[1] > 0:
        color *= c_sky
    else:
        f = min(1.0, -refl_dir[1] * 25.0)
        color *= c_sky * (1 - f) + c_floor * f
    return color


'''
Calculates the form factor between a fragment with normal N
and a sphere with radius R and position DPOS relative
to the fragment
'''
ff_clamp = 8.0
gi_clamp = 1.0
@ti.func
def form_factor_ball(n, dpos, r):
    d = dpos.norm()
    z = ti.dot(n, dpos) / d
    f = ff_clamp if z > 0 else 0.0
    if d > r and z > 0:
        #f1 = (d ** 2 + r ** 2 - d * r) / (d - r)
        #f2 = ti.sqrt((d - r) * (d + r))
        #f = min(2.0 * z / (3.0 * d) * (f1 - f2), ff_clamp)
        x = d / r
        f = min(2.0 * z / (3.0 * x) * (1. / (x - 1.) + 1. / 2. / x), ff_clamp)
    return f

@ti.func
def form_factor_floor(n, d, r):
    f = 0.0
    if n[1] < 0:
        x = d / r
        f = -n[1] / (x ** 2 + 1)
    return f


@ti.func
def get_gi_factors(camera, scene, pos, fragpos, normal, radius, color):
    a = camera.untrans_pos(pos)
    f = form_factor_ball(normal, (pos - fragpos), radius)
    cosfactor = 0.0
    for light in ti.static(scene.lights):
        cosfactor += max(0.5, ti.dot(light.get_dir(a), ts.normalize(pos - a)))
    gi = min(f, gi_clamp) / gi_clamp / 5 * color * cosfactor
    return f / ff_clamp, gi


class CookTorrance:

    specular = 0.6
    eps = 1e-4

    '''
    Cook-Torrance BRDF with an Lambertian factor.
    Lo(p, w0)=\int f_c*Li(p, wi)(n.wi)dwi where
    f_c = k_d*f_lambert * k_s*f_cook-torrance
    For finite point lights, the integration is evaluated as a 
    discrete sum.

    '''

    def __init__(self, **kwargs):
        self.kd = 1.5
        self.ks = 2.0
        self.roughness = 0.6
        self.metallic = 0.0
        self.__dict__.update(kwargs)

    '''
    Calculates the Cook-Torrance BRDF as
    f_lambert = color / pi
    k_s * f_specular = D * F * G / (4 * (wo.n) * (wi.n))
    '''

    @ti.func
    def brdf(self, normal, viewdir, lightdir, color):
        halfway = ts.normalize(viewdir + lightdir)
        ndotv = max(ti.dot(viewdir, normal), self.eps)
        ndotl = max(ti.dot(lightdir, normal), self.eps)
        diffuse = self.kd * color / math.pi
        specular = self.microfacet(normal, halfway)\
                    * self.frensel(viewdir, halfway, color)\
                    * self.geometry(ndotv, ndotl)
        specular /= 4 * ndotv * ndotl
        return diffuse + specular

    '''
    Trowbridge-Reitz GGX microfacet distribution
    '''
    @ti.func
    def microfacet(self, normal, halfway):
        alpha = self.roughness
        ggx = alpha ** 2 / math.pi
        ggx /= (ti.dot(normal, halfway)**2 * (alpha**2 - 1.0) + 1.0) ** 2
        return ggx
    
    '''
    Fresnel-Schlick approximation
    '''
    @ti.func
    def frensel(self, view, halfway, color):
        f0 = ts.mix(ts.vec3(self.specular), color, self.metallic) 
        hdotv = min(1, max(ti.dot(halfway, view), 0))
        return (f0 + (1.0 - f0) * (1.0 - hdotv) ** 5) * self.ks
 
    '''
    Smith's method with Schlick-GGX
    '''
    @ti.func
    def geometry(self, ndotv, ndotl):
        k = (self.roughness + 1.0) ** 2 / 8
        geom = ndotv * ndotl\
            / (ndotv * (1.0 - k) + k) / (ndotl * (1.0 - k) + k)
        return max(0, geom)
       
    
    '''
    Compared with the basic lambertian-phong shading, 
    the rendering function also takes the surface color as parameter.
    Also note that the viewdir points from the camera to the object
    so it needs to be inverted when calculating BRDF.
    '''
    @ti.func
    def render_func(self, pos, normal, viewdir, light, color):
        lightdir = light.get_dir(pos)
        costheta = max(0, ti.dot(normal, lightdir))
        l_out = ts.vec3(0.0)
        if costheta > 0:
            l_out = self.brdf(normal, -viewdir, lightdir, color)\
                 * costheta * light.get_color(pos)
        return l_out