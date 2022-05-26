import taichi as ti
from .consts import *

def create_field(dims, dtype):
    if type(dims) == int:
        dims = (dims,)
    if not dims:
        ti_field = ti.field(dtype)
    elif len(dims) == 1:
        ti_field = ti.Vector.field(dims[0], dtype)
    elif len(dims) == 2:
        ti_field = ti.Matrix.field(dims[0], dims[1], dtype)
    return ti_field

def make_cell(n_axes, dx, n_particles=None, x0=(0,) * DIM):
    coords_1d = [x + d * np.arange(n) for x, d, n in zip(x0, dx, n_axes)]
    pos = np.stack(np.meshgrid(*coords_1d)).reshape(DIM, -1).T
    if n_particles:
        pos = pos[:n_particles, :]
    return pos


def from_numpy_chk(ti_arr, np_arr):
    if np_arr.dtype != int:
        np_arr = np_arr.astype(np.float32)
    ti_arr.from_numpy(np_arr)
    ti_dim = ti_arr.to_numpy().shape
    np_dim = np_arr.shape
    assert ti_dim == np_dim,\
        "Error: Taichi-scope %s and numpy shape %s mismatch!" % (ti_dim, np_dim)



@ti.func
def memset(arr, val):
    for X in ti.grouped(arr):
        arr[X] = val

class Module:

    def register(self, system):
        self.system = system
        return self

    def build(self):
        pass


class SimulationModuleError(AttributeError):
    pass
    