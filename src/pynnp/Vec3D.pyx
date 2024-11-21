# n2p2 - A neural network potential package
# Copyright (C) 2018 Andreas Singraber (University of Vienna)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

cdef class Vec3D:
    cdef pd.Vec3D* thisptr
    cdef object owner

    def __cinit__(self, x=None, y=None, z=None):
        cdef Vec3D v = <Vec3D>x
        if x is True:
            self.owner = False
            return
        elif x is not None and isinstance(x, Vec3D):
            self.thisptr = new pd.Vec3D(deref(v.thisptr))
        elif x is not None and y is not None and z is not None:
            self.thisptr = new pd.Vec3D(x, y, z)
        elif x is not None or y is not None or z is not None:
            raise ValueError("ERROR: Wrong arguments, either set all "
                             "components or none.")
        else:
            self.thisptr = new pd.Vec3D()
        self.owner = True
    cdef set_ptr(self, pd.Vec3D* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def __getitem__(self, item):
        if item >= 3 or item < 0:
            raise IndexError
        return deref(self.thisptr).r[<size_t>item]
    def __setitem__(self, item, value):
        if item >= 3 or item < 0:
            raise IndexError
        deref(self.thisptr).r[<size_t>item] = value
    def __str__(self):
        return "x: {0:f} y: {1:f} z: {2:f}".format(*(deref(self.thisptr).r))
    def __iadd__(self, Vec3D v):
        self.thisptr.iadd(deref(v.thisptr))
        return self
    def __isub__(self, Vec3D v):
        self.thisptr.isub(deref(v.thisptr))
        return self
    def __imul__(self, a):
        self.thisptr.imul(a)
        return self
    def __itruediv__(self, a):
        self.thisptr.itruediv(a)
        return self
    def __mul__(ls, rs):
        if isinstance(ls, Vec3D) and isinstance(rs, Vec3D):
            return ls.mul_Vec3D(rs)
        elif isinstance(ls, Vec3D) and isinstance(rs, float):
            return ls.mul_rfloat(rs)
        else:
            return NotImplemented
    def __rmul__(rs, ls):
        if isinstance(ls, float) and isinstance(rs, Vec3D):
            return rs.mul_lfloat(ls)
        else:
            return NotImplemented
    def mul_Vec3D(Vec3D self, Vec3D v):
        return self.thisptr.mul_vec3d(deref(v.thisptr))
    def mul_rfloat(Vec3D self, a):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.mul_d(deref(self.thisptr), a)
        return res
    def mul_lfloat(Vec3D self, a):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.d_mul(a, deref(self.thisptr))
        return res
    def __eq__(self, Vec3D rhs):
        return self.thisptr.eq(deref(rhs.thisptr))
    def __ne__(self, Vec3D rhs):
        return self.thisptr.ne(deref(rhs.thisptr))
    def norm(self):
        return self.thisptr.norm()
    def norm2(self):
        return self.thisptr.norm2()
    def normalize(self):
        self.thisptr.normalize()
        return self
    def cross(self, Vec3D v):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = self.thisptr.cross(deref(v.thisptr))
        return res
    def __add__(Vec3D self, Vec3D rhs):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.add(deref(self.thisptr), deref(rhs.thisptr))
        return res
    def __sub__(Vec3D self, Vec3D rhs):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.sub(deref(self.thisptr), deref(rhs.thisptr))
        return res
    def __neg__(Vec3D self):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.neg(deref(self.thisptr))
        return res
    def __truediv__(self, a):
        if isinstance(a, float):
            return self.div_float(a)
        else:
            return NotImplemented
    def div_float(Vec3D self, a):
        cdef Vec3D res = Vec3D()
        res.thisptr[0] = pd.div_d(deref(self.thisptr), a)
        return res

    # r
    @property
    def r(self):
        return deref(self.thisptr).r
    @r.setter
    def r(self, value):
        deref(self.thisptr).r = value
