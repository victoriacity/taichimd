import taichi as ti
import numpy as np

@ti.data_oriented
class ForceField:
    
    def register(self, system):
        self.system = system
        return self

    def build(self, system):
        raise NotImplementedError

    def calc_force(self):
        raise NotImplementedError



@ti.data_oriented
class ClassicalFF(ForceField):

    MAX_ATOMTYPE = 32
    MAX_BONDTYPE = 32
    MAX_BENDTYPE = 16
    MAX_TORSIONTYPE = 16

    MAX_BOND = 6
    MAX_BEND = 3
    MAX_TORSION = 1

    def __init__(self, nonbond=None, bonded=None,
                bending=None, torsional=None,
                external=None):
        # force types
        self.nonbond = nonbond
        self.bonded = bonded
        self.bending = bending
        self.torsional = torsional
        # external potential is independent of atom type
        self.external = external

    def set_params(self, nonbond=None, 
            bonded=None, bending=None, torsional=None):
        self.nonbond_params_d = nonbond
        self.bonded_params_d = bonded
        self.bending_params_d = bending
        self.torsional_params_d = torsional
        
        
    def register(self, system):
        if ti.static(system.is_atomic and (
            self.bonded != None or self.bending != None\
            or self.torsional != None
        )):
            print("Warning: the simulation system does not have "
                "any bond structure. Bonding/bending/torsional potentials will not be used.")  
        # initialize data structures
        if ti.static(self.nonbond != None):
            self.nonbond_params = ti.Vector(self.nonbond.n_params, dt=ti.f32)
            ti.root.bitmasked(ti.ij, (self.MAX_ATOMTYPE, self.MAX_ATOMTYPE)).place(self.nonbond_params)
        if ti.static(not system.is_atomic and self.bonded != None):
            self.bond_np = []
            self.bond = ti.Vector(3, dt=ti.i32)
            self.bonded_params = ti.Vector(self.bonded.n_params, dt=ti.f32)
            ti.root.dense(ti.i, self.MAX_BOND 
                    * system.n_particles).place(self.bond)
            ti.root.pointer(ti.i, self.MAX_BONDTYPE).place(self.bonded_params)
        if ti.static(not system.is_atomic and self.bending != None):
            self.bend_np = []
            self.bend = ti.Vector(4, dt=ti.i32)
            self.bending_params = ti.Vector(self.bending.n_params, dt=ti.f32)
            ti.root.dense(ti.i, self.MAX_BEND 
                    * system.n_particles).place(self.bend)
            ti.root.pointer(ti.i, self.MAX_BENDTYPE).place(self.bending_params)
        if ti.static(not system.is_atomic and self.torsional != None):
            self.torsion_np = []
            self.torsion = ti.Vector(5, dt=ti.i32)
            self.torsional_params = ti.Vector(self.torsional.n_params, dt=ti.f32)
            ti.root.dynamic(ti.i, self.MAX_TORSION 
                    * system.n_particles).place(self.torsion)
            ti.root.pointer(ti.i, self.MAX_TORSIONTYPE).place(self.torsional_params)
        if ti.static(not system.is_atomic):
            self.intra = ti.var(dt=ti.i32) # boolean
            ti.root.bitmasked(ti.ij, (system.n_particles, system.n_particles)).place(self.intra)
        return super().register(system)


    def populate_tables(self, i0, m, n):
        sys = self.system
        if ti.static(not sys.is_atomic and self.bonded != None):
            table = np.tile(np.array(m.bond), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.bond_np.append(table.reshape(-1, 3))
        if ti.static(not sys.is_atomic and self.bending != None):
            table = np.tile(np.array(m.bending), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.bend_np.append(table.reshape(-1, 4))
        if ti.static(not sys.is_atomic and self.torsional != None):
            table = np.tile(np.array(m.torsion), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.torsion_np.append(table.reshape(-1, 5))
        if ti.static(not sys.is_atomic):
            self.set_intra(i0, n, m.natoms, m.intra)

    
    def set_intra(self, i0, nmolec, natoms, imat):
        for i in range(nmolec):
            istart = i0 + i * natoms
            iend = istart + natoms
            for l in range(natoms):
                for m in range(natoms):
                    self.intra[istart + l, istart + m] = imat[l, m]


    def build(self):
        sys = self.system
        # set nonbond parameters
        if ti.static(self.nonbond != None):
            for k, v in self.nonbond_params_d.items():
                self.nonbond_params[k, k] = self.nonbond.fill_params(*v)
                for k2, v2 in self.nonbond_params_d.items():
                    v_comb = self.nonbond.combine(v, v2)
                    self.nonbond_params[k, k2] = v_comb
                    self.nonbond_params[k2, k] = v_comb
        # build bond table and set parameters
        if ti.static(not sys.is_atomic and self.bonded != None):
            for k, v in self.bonded_params_d.items():
                self.bonded_params[k] = self.bonded.fill_params(*v)
            self.bond.from_numpy(np.vstack(self.bond_np))
        # build bend table
        if ti.static(not sys.is_atomic and self.bending != None):
            for k, v in self.bending_params_d.items():
                self.bending_params[k] = self.bending.fill_params(*v)
            self.bend.from_numpy(np.vstack(self.bend_np))
        # build torsion table
        if ti.static(not sys.is_atomic and self.torsional != None):
            for k, v in self.torsional_params_d.items():
                self.torsional_params[k] = self.torsional.fill_params(*v)
            self.torsion.from_numpy(np.vstack(self.torsion_np))


    @ti.func
    def calc_force(self):
        sys = self.system
        sys.ep[None] = 0.0
        for i in sys.force:
            sys.force[i].fill(0.0)
            if ti.static(self.external != None):
                sys.ep[None] += self.external(sys.position[i])
                sys.force[i] += self.external.force(sys.position[i])
                if ti.static(sys.integrator.requires_hessian):
                    sys.hessian[i, i] += self.external.hessian(sys.position[i])
        if ti.static(self.nonbond != None):
            for i, j in ti.ndrange(sys.n_particles, sys.n_particles):
                not_excl = True
                if ti.static(not sys.is_atomic):
                    not_excl = self.intra[i, j] == 0
                if i < j and not_excl:
                    d = sys.calc_distance(sys.position[j], sys.position[i])
                    r2 = (d ** 2).sum()
                    itype = sys.type[i]
                    jtype = sys.type[j]
                    params = self.nonbond_params[itype, jtype]
                    uij = self.nonbond(r2, params)
                    if uij != 0.0:
                        sys.ep[None] += uij
                        force = self.nonbond.force(d, r2, params)
                        # += performs atomic add
                        sys.force[i] += force
                        sys.force[j] -= force
                        if ti.static(sys.integrator.requires_hessian):
                            h = self.nonbond.hessian(d, r2, params)
                            sys.hessian[i, j] = h
                            sys.hessian[j, i] = h
                            sys.hessian[i, i] -= h
                            sys.hessian[j, j] -= h
 
        if ti.static(not sys.is_atomic and self.bonded != None):
            for x in self.bond:
                bondtype, i, j = self.bond[x][0], self.bond[x][1], self.bond[x][2]
                if bondtype > 0:
                    params = self.bonded_params[bondtype]
                    d = sys.calc_distance(sys.position[j], sys.position[i])
                    if ti.static(self.bonded == None):
                        raise NotImplementedError("Rigid bonds are not implemented yet!") 
                    else:
                        r2 = (d ** 2).sum()
                        sys.ep[None] += self.bonded(r2, params)
                        force = self.bonded.force(d, r2, params)
                        sys.force[i] += force
                        sys.force[j] -= force
                        if ti.static(sys.integrator.requires_hessian):
                            h = self.bonded.hessian(d, r2, params)
                            sys.hessian[i, j] = h
                            sys.hessian[j, i] = h
                            sys.hessian[i, i] -= h
                            sys.hessian[j, j] -= h

        if ti.static(not sys.is_atomic and self.bending != None):
            if ti.static(sys.integrator.requires_hessian):
                raise NotImplementedError("Hessian not supported for bond bending potentials!")
            for x in self.bend:
                bendtype, i, j, k = self.bend[x][0], self.bend[x][1], self.bend[x][2], self.bend[x][3]
                if bendtype > 0:
                    params = self.bending_params[bendtype]
                    v1 = sys.calc_distance(sys.position[i], sys.position[j])
                    v2 = sys.calc_distance(sys.position[k], sys.position[j])
                    if ti.static(self.bonded == None):
                        raise NotImplementedError("Fixed bond angles are not implemented yet!") 
                    else:
                        l1 = v1.norm()
                        l2 = v2.norm()
                        d = v1.dot(v2)
                        cosx = d / (l1 * l2)
                        sys.ep[None] += self.bending(cosx, params)
                        d_cosx = self.bending.derivative(cosx, params)
                        u = 1 / l1 / l2
                        f1 = (v2 - d / l1 ** 2 * v1) * u
                        f2 = (v1 - d / l2 ** 2 * v2) * u
                        sys.force[i] -= f1 * d_cosx
                        sys.force[k] -= f2 * d_cosx
                        sys.force[j] += (f1 + f2) * d_cosx

                        
