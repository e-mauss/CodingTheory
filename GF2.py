import numpy as np
from matplotlib import pyplot as plt


class GF:
    irreducibles = {
        2: 0b111,
        3: 0b1101,
        4: 0b11001,
        5: 0b100101,
        6: 0b1100001,
        7: 0b11000001,
        8: 0b100011101,
    }

# Constructor
    def __init__(self, e):
        self.e = e

# Num and Poly conversions
    def _num2poly(self, x: int | np.poly1d):
        if type(x) is int or type(x) is np.int_:
            return np.poly1d([int(x) for x in bin(x)[2:]])
        elif type(x) is np.poly1d:
            return x

    def _nums2polys(self, *arg):
        return tuple(self._num2poly(num) for num in arg)

    def _poly2num(self, x: int | np.poly1d):
        if type(x) is int or type(x) is np.int_:
            return x
        elif type(x) is np.poly1d:
            return int("".join([str(int(i % 2)) for i in x.c]), 2)

    def _polys2nums(self, *arg):
        return tuple(self._poly2num(num) for num in arg)

    def _coeff_adjust(self, poly):
        return np.poly1d([x % 2 for x in poly.c])

# Mathematical operations with polynomial returns
    def _polymod(self, x, m):
        x, m = self._nums2polys(x, m)
        return self._coeff_adjust((x/m)[1])

    def _polydiv(self, x, d):
        x, d = self._nums2polys(x, d)
        return self._coeff_adjust((x/d)[0])

    def _polymul(self, x, y):
        x, y, m = self._nums2polys(x, y, self.irreducibles[self.e])
        return self._polymod(x*y, m)

    def _polyinverse(self, x):
        a, b = self._nums2polys(x, self.irreducibles[self.e])
        _, inv, _ = self._eGCD(a, b)
        return inv

    def _eGCD(self, a, b):
        if self._poly2num(a) == 0:
            return b, 0, 1

        gcd, x1, y1 = self._eGCD(self._polymod(b, a), a)

        x = y1 - self._polymul(self._polydiv(b, a), x1)

        return gcd, x, x1

# Mathematical operations with integer representation returns
    def _mod(self, x, m):
        return self._poly2num(self._polymod(x, m))

    def _div(self, x, d):
        return self._poly2num(self._polydiv(x, d))

    def _mul(self, x, y):
        return self._poly2num(self._polymul(x, y))

    def _inverse(self, x):
        return self._poly2num(self._polyinverse(x))

# Functions for gaussian algorithm
    def gauss(self, generator):
        mat = np.array(generator)
        self.forward_elim(mat)
        self.backward_sub(mat)
        return mat

    def forward_elim(self, mat):
        for i in np.arange(min(mat.shape)):
            if mat[i, i] == 0:
                self.get_one(mat, i)
            self.zero_below(mat, i)

    def backward_sub(self, mat):
        for i in reversed(np.arange(min(mat.shape))):
            self.zero_above(mat, i)

    def zero_below(self, mat, i):
        for j, x in enumerate(mat[i+1:, i]):
            if x != 0:
                mat[j+i+1] = (mat[i]+mat[j+i+1]) % 2

    def zero_above(self, mat, i):
        for j, x in enumerate(mat[:i, i]):
            if x != 0:
                mat[j] = (mat[i]+mat[j]) % 2

    def get_one(self, mat, i):
        for n in np.arange(i, mat.shape[0]):
            if mat[n, i] == 1:
                mat[[i, n]] = mat[[n, i]]

# linear code utility
    def check_canonical(self, mat):
        canonical = True
        for i in np.arange(min(mat.shape)):
            if np.sum(mat[:, i]) != 1 or mat[i, i] != 1:
                canonical = False
                break
        return canonical

    def get_control_matrix(self, mat):
        k_mat = min(mat.shape)
        diff = max(mat.shape) - k_mat
        control = np.zeros([diff, diff])
        for x in np.arange(diff):
            control[x, x] = 1
        control = np.hstack([mat[:, k_mat:].T, control])
        return control

    def compute_syndromes(self, h):
        n_mat = max(h.shape)
        diffnk = min(h.shape)

        syndromes = []
        amount_of_syndromes = pow(2, diffnk)

        for err in np.arange(pow(2, n_mat)):
            errpoly = self.get_errpoly(n_mat, err)
            res = np.matmul(errpoly, h.T) % 2

            if not np.any([(syndrome == res).all() for _, syndrome in syndromes]):
                syndromes.append((errpoly, res))
                if len(syndromes) == amount_of_syndromes:
                    break

        return syndromes

    def get_errpoly(self, n_mat, err):
        errpoly = self._num2poly(err).c
        errpoly = np.pad(errpoly, [(n_mat-len(errpoly), 0)])
        errpoly = np.reshape(errpoly, (1, -1))
        return errpoly

# Interface methods (many of these methods will be removed to become tests)
    def multable(self, e):
        a = b = np.arange(pow(2, self.e))[1:]
        return np.array([[self._mul(x, y) for y in b] for x in a])

    def show_multable(self):
        plt.imshow(np.subtract(0, self.multable(self.e)), 'gray')
        plt.show()

    def inverses(self):
        for num in np.arange(1, pow(2, self.e)):
            inv = self._inverse(num)
            print(f"u = {num:>4} | u⁻¹ = {inv:>4}")

    def verify_inverses(self):
        for num in np.arange(1, pow(2, self.e)):
            inv = self._inverse(num)
            print(f"u * u⁻¹ = {self._mul(num, inv)}")

    def verify_cummutativity(self):
        a = 2
        b = pow(2, self.e-1)
        c = pow(2, self.e)-1
        ab = self._mul(a, b)
        bc = self._mul(b, c)
        print(f'a={a}, b={b}, c={c}')
        print(f'ab={ab}, bc={bc}')
        print(
            f'(ab)c={self._mul(ab, c)}, a(bc)={self._mul(a, bc)}')

    def demo_gauss(self):
        a = [[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0],
             [0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 1]]
        a = self.gauss(a)
        ct = self.get_control_matrix(a)
        for tp in self.compute_syndromes(ct):
            print(tp)


GF(4).show_multable()
GF(4).verify_cummutativity()
GF(4).inverses()
GF(4).verify_inverses()
GF(4).demo_gauss()