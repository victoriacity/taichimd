'''
Miniature static version of taichi-three package with components for
visualizing taichimd simulations only.

Corresponds to taichi_three version 0.0.4 with multiple framebuffer support, see
https://github.com/victoriacity/taichi_three/tree/279e7310bf3935766416c27f7627608839eb7849/taichi_three

Requires taichi_glsl package.
'''
import taichi as ti
import taichi_glsl as ts
import math

'''
taichi_three/common.py
'''
class AutoInit:
    def init(self):
        if not hasattr(self, '_AutoInit_had_init'):
            self._init()
            self._AutoInit_had_init = True

    def _init(self):
        raise NotImplementedError


'''
taichi_three/light.py
'''
@ti.data_oriented
class Light(AutoInit):

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [x / norm for x in dir]
 
        self.dir_py = [-x for x in dir]
        self.color_py = color or [1, 1, 1] 

        self.dir = ti.Vector.field(3, ti.float32, ())
        self.color = ti.Vector.field(3, ti.float32, ())
        # store the current light direction in the view space
        # so that we don't have to compute it for each vertex
        self.viewdir = ti.Vector.field(3, ti.float32, ())

    def set(self, dir=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x**2 for x in dir))
        dir = [x / norm for x in dir]
        self.dir_py = dir
        self.color = color

    def _init(self):
        self.dir[None] = self.dir_py
        self.color[None] = self.color_py

    @ti.func
    def intensity(self, pos):
        return 1

    @ti.func
    def get_color(self, pos):
        return self.color[None] * self.intensity(pos)

    @ti.func
    def get_dir(self, pos):
        return self.viewdir

    @ti.func
    def set_view(self, camera):
        self.viewdir[None] = camera.untrans_dir(self.dir[None])


class PointLight(Light):

    def __init__(self, position=None, color=None,
            c1=None, c2=None):
        position = position or [0, 1, -3]
        if c1 is not None: 
            self.c1 = c1
        if c2 is not None: 
            self.c2 = c2
        self.pos_py = position
        self.color_py = color or [1, 1, 1] 
        self.pos = ti.Vector(3, ti.float32, ())
        self.color = ti.Vector(3, ti.float32, ())
        self.viewpos = ti.Vector(3, ti.float32, ())

    def _init(self):
        self.pos[None] = self.pos_py
        self.color[None] = self.color_py

    @ti.func
    def set_view(self, camera):
        self.viewpos[None] = camera.untrans_pos(self.pos[None])

    @ti.func
    def intensity(self, pos):
        distsq = (self.viewpos[None] - pos).norm_sqr()
        return 1. / (1. + self.c1 * ti.sqrt(distsq) + self.c2 * distsq)

    @ti.func
    def get_dir(self, pos):
        return ts.normalize(self.viewpos[None] - pos)


'''
taichi_three/scene.py
'''
@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.lights = []
        self.cameras = []
        self.opt = None
        self.models = []

    def add_model(self, model):
        model.scene = self
        self.models.append(model)

    def add_camera(self, camera):
        camera.scene = self
        self.cameras.append(camera)

    def add_light(self, light):
        light.scene = self
        self.lights.append(light)

    def _init(self):
        for light in self.lights:
            light.init()
        for camera in self.cameras:
            camera.init()
        for model in self.models:
            model.init()

    def render(self):
        self.init()
        self._render()

    @ti.kernel
    def _render(self):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                camera.clear_buffer()
                # sets up light directions
                for light in ti.static(self.lights):
                    light.set_view(camera)
                if ti.static(len(self.models)):
                    for model in ti.static(self.models):
                        model.render(camera)

'''
taichi_three/camera.py
'''
@ti.data_oriented
class Camera(AutoInit):
    ORTHO = 'Orthogonal'
    TAN_FOV = 'Tangent Perspective' # rectilinear perspective
    COS_FOV = 'Cosine Perspective' # curvilinear perspective, see en.wikipedia.org/wiki/Curvilinear_perspective

    def __init__(self, res=None, fx=None, fy=None, cx=None, cy=None,
            pos=[0, 0, -2], target=[0, 0, 0], up=[0, 1, 0], fov=30):
        self.res = res or (512, 512)
        self.buffers = []
        self.add_buffer("img", dim=3, dtype=ti.f32)
        self.add_buffer("zbuf", dim=0, dtype=ti.f32)
        self.trans = ti.Matrix(3, 3, ti.f32, ())
        self.pos = ti.Vector(3, ti.f32, ())
        self.target = ti.Vector(3, ti.f32, ())
        self.intrinsic = ti.Matrix(3, 3, ti.f32, ())
        self.type = self.TAN_FOV
        self.fov = math.radians(fov)

        self.cx = cx or self.res[0] // 2
        self.cy = cy or self.res[1] // 2
        self.fx = fx or self.cx / math.tan(self.fov)
        self.fy = fy or self.cy / math.tan(self.fov)
        # python scope camera transformations
        self.pos_py = pos
        self.target_py = target
        self.trans_py = None
        self.up_py = up
        self.set(init=True)
        # mouse position for camera control
        self.mpos = (0, 0)


    def add_buffer(self, name, dim=3, dtype=ti.f32):
        if not dim:
            buffer = ti.field(dtype, self.res)
        else:
            buffer = ti.Vector.field(dim, dtype, self.res)
        setattr(self, name, buffer)
        self.buffers.append(buffer)

    def set_intrinsic(self, fx=None, fy=None, cx=None, cy=None):
        # see http://ais.informatik.uni-freiburg.de/teaching/ws09/robotics2/pdfs/rob2-08-camera-calibration.pdf
        self.fx = fx or self.fx
        self.fy = fy or self.fy
        self.cx = cx or self.cx
        self.cy = cy or self.cy

    '''
    NOTE: taichi_three uses a LEFT HANDED coordinate system.
    that is, the +Z axis points FROM the camera TOWARDS the scene,
    with X, Y being device coordinates
    '''
    def set(self, pos=None, target=None, up=None, init=False):
        pos = self.pos_py if pos is None else pos
        target = self.target_py if target is None else target
        up = self.up_py if up is None else up
        # fwd = target - pos
        fwd = [target[i] - pos[i] for i in range(3)]
        # fwd = fwd.normalized()
        fwd_len = math.sqrt(sum(x**2 for x in fwd))
        fwd = [x / fwd_len for x in fwd]
        # right = fwd.cross(up) 
        right = [
                fwd[2] * up[1] - fwd[1] * up[2],
                fwd[0] * up[2] - fwd[2] * up[0],
                fwd[1] * up[0] - fwd[0] * up[1],
                ]
        # right = right.normalized()
        right_len = math.sqrt(sum(x**2 for x in right))
        right = [x / right_len for x in right]
        # up = right.cross(fwd)
        up = [
             right[2] * fwd[1] - right[1] * fwd[2],
             right[0] * fwd[2] - right[2] * fwd[0],
             right[1] * fwd[0] - right[0] * fwd[1],
             ]

        # trans = ti.Matrix.cols([right, up, fwd])
        trans = [right, up, fwd]
        self.trans_py = [[trans[i][j] for i in range(3)] for j in range(3)]
        self.pos_py = pos
        self.target_py = target
        if not init:
            self.pos[None] = self.pos_py
            self.trans[None] = self.trans_py
            self.target[None] = self.target_py

    def _init(self):
        self.pos[None] = self.pos_py
        self.trans[None] = self.trans_py
        self.target[None] = self.target_py
        self.intrinsic[None][0, 0] = self.fx
        self.intrinsic[None][0, 2] = self.cx
        self.intrinsic[None][1, 1] = self.fy
        self.intrinsic[None][1, 2] = self.cy
        self.intrinsic[None][2, 2] = 1.0

    @ti.func
    def clear_buffer(self):
        for buf in ti.static(self.buffers):
            for I in ti.grouped(self.img):
                buf[I] *= 0.0

    def from_mouse(self, gui):
        is_alter_move = gui.is_pressed(ti.GUI.CTRL)
        if gui.is_pressed(ti.GUI.LMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.orbit((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                    pov=is_alter_move)
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.RMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.zoom_by_mouse(mpos, (mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                        dolly=is_alter_move)
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.MMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.pan((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
        else:
            if gui.event and gui.event.key == ti.GUI.WHEEL:
                # one mouse wheel unit is (0, 120)
                self.zoom(-gui.event.delta[1] / 1200,
                    dolly=is_alter_move)
                gui.event = None
            mpos = (0, 0)
        self.mpos = mpos


    def orbit(self, delta, sensitivity=5, pov=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = self.fov
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(ds, dt, 1).normalized()
            newdir = [sum(self.trans[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            if pov:
                newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
                self.set(target=newtarget)
            else:
                newpos = [self.target_py[i] - dis * newdir[i] for i in range(3)]
                self.set(pos=newpos)

    def zoom_by_mouse(self, pos, delta, sensitivity=3, dolly=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            z = math.sqrt(ds ** 2 + dt ** 2) * sensitivity
            if (pos[0] - 0.5) * ds + (pos[1] - 0.5) * dt > 0:
                z *= -1
            self.zoom(z, dolly)
    
    def zoom(self, z, dolly=False):
        newpos = [(1 + z) * self.pos_py[i] - z * self.target_py[i] for i in range(3)]
        if dolly:
            newtarget = [z * self.pos_py[i] + (1 - z) * self.target_py[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)
        else:
            self.set(pos=newpos)

    def pan(self, delta, sensitivity=3):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = self.fov
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(-ds, -dt, 1).normalized()
            newdir = [sum(self.trans[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
            newpos = [self.pos_py[i] + newtarget[i] - self.target_py[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)

    @ti.func
    def trans_pos(self, pos):
        return self.trans[None] @ pos + self.pos[None]

    @ti.func
    def trans_dir(self, pos):
        return self.trans[None] @ pos

    @ti.func
    def untrans_pos(self, pos):
        return self.trans[None].inverse() @ (pos - self.pos[None])

    @ti.func
    def untrans_dir(self, pos):
        return self.trans[None].inverse() @ pos
    
    @ti.func
    def uncook(self, pos):
        if ti.static(self.type == self.ORTHO):
            pos[0] *= self.intrinsic[None][0, 0] 
            pos[1] *= self.intrinsic[None][1, 1]
            pos[0] += self.intrinsic[None][0, 2]
            pos[1] += self.intrinsic[None][1, 2]
        elif ti.static(self.type == self.TAN_FOV):
            pos = self.intrinsic[None] @ pos
            pos[0] /= abs(pos[2])
            pos[1] /= abs(pos[2])
        else:
            raise NotImplementedError("Curvilinear projection matrix not implemented!")
        return ts.vec2(pos[0], pos[1])