import taichi as ti
import numpy as np
from .consts import *

@ti.data_oriented
class Molecule:

    def __init__(self, atoms, name=None, bond=None, bending=None, 
            torsion=None, intra=None, struc=None):
        if isinstance(atoms, int):
            atoms = [atoms]
        self.atoms = atoms
        self.natoms = len(atoms)
        self.name = name
        self.bond = bond or []
        self.bending = bending or []
        self.torsion = torsion or []
        self.intra = intra if intra is not None else np.ones((self.natoms, self.natoms), dtype=np.int)
        self.struc = struc
        if struc is None:
            self._get_struc()

    def _get_struc(self):
        if self.natoms == 1:
            self.struc = np.array([0.0] * DIM).reshape(1, -1)
        else:
            pass

