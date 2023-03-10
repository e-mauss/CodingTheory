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

    def get_padded_poly(self, length, poly):
        errpoly = self._num2poly(poly).c
        errpoly = np.pad(errpoly, [(length-len(errpoly), 0)])
        return errpoly

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

# Linear code utility
    def check_canonical(self, mat):
        canonical = True
        for i in np.arange(min(mat.shape)):
            if np.sum(mat[:, i]) != 1 or mat[i, i] != 1:
                canonical = False
                break
        return canonical

    def get_generator_matrix(self, mat):
        n_mat = max(mat.shape)
        k_mat = n_mat - min(mat.shape)
        gen = np.zeros([k_mat, k_mat])
        for x in np.arange(k_mat):
            gen[x, x] = 1

        gen = np.hstack([gen, mat[:, :k_mat].T])
        return gen

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

        for triv in np.arange(n_mat):
            errpoly = self.get_padded_poly(n_mat, pow(2, triv))
            res = np.matmul([errpoly], h.T) % 2
            syndromes.append((errpoly, res[0]))

        for err in np.arange(pow(2, n_mat)):
            errpoly = self.get_padded_poly(n_mat, err)
            res = np.matmul([errpoly], h.T) % 2

            if not np.any([(syndrome == res).all() for _, syndrome in syndromes]):
                syndromes.append((errpoly, res[0]))
                if len(syndromes) == amount_of_syndromes:
                    break

        return syndromes

    def decode(self, message, control, syndromes):
        errpoly = self.get_padded_poly(max(control.shape), message)
        res = np.matmul([errpoly], control.T) % 2
        for codeword, syndrome in syndromes:
            if (syndrome == res).all():
                return ((errpoly - codeword) % 2)
    
# Hamming code utility
    def hamming(self, m):
        assert(m >= 3)
        n = pow(2, m)-1
        column_values = np.arange(1, n+1)
        for i in reversed(np.arange(m)):
            column_values = np.append(column_values, column_values[pow(2, i)-1])
            column_values = np.delete(column_values, pow(2, i)-1)

        mat = map(lambda poly: self.get_padded_poly(m, poly), column_values)
        mat = np.column_stack(list(mat))
        return mat

    def hamming_decode(self, message, control):
        errpoly = self.get_padded_poly(max(control.shape), message)
        syndrome = (np.matmul([errpoly], control.T) % 2)[0]
        if not (syndrome==0).all():
            factor = syndrome[np.nonzero(syndrome)[0][0]]
            search = syndrome * factor
            error_index = np.nonzero(np.all(control.T == search, axis=1))
            errpoly[error_index] = (errpoly[error_index] - factor) % 2
        return errpoly

# Reed muller codes
    def reed_muller(self, r, m):
        if r == 0:
            return np.ones((1,pow(2,m)))
        if r > m:
            return self.reed_muller(m, m)

        tl = self.reed_muller(r, m-1)
        br = self.reed_muller(r-1, m-1)
        bl = np.zeros((br.shape[0], tl.shape[1]))
        return self.forge_four_corners(tl, br, bl)

    def forge_four_corners(self, tl, br, bl):
        top = np.hstack((tl, tl))
        bot = np.hstack((bl, br))
        return np.vstack((top, bot))

# Reed solomon codes
    def find_primitive(self):
        q = pow(2, self.e)
        for alpha in range(1, q):
            acc = 1
            ls = []
            for i in range(0, q-1):
                acc = self._polymul(acc, alpha)
                ls.append(self._poly2num(acc))
            if len(ls) == len(set(ls)):
                return alpha

    def get_polynomials(self, d):
        alpha = self._num2poly(self.find_primitive())
        q = pow(2, self.e)
        acc = 1
        roots = []
        for i in range(q):
            acc = self._polymul(acc, alpha)
            roots.append([1, acc])

        gx = [1]
        for i in range(0, d-1):
            gx = self.nested_poly_mul(gx, roots[i])
        hx = [1]
        for i in range(d-1, q-1):
            hx = self.nested_poly_mul(hx, roots[i])

        return gx, hx

    def nested_poly_mul(self, a, b):
        res = [0]*(len(a)+len(b)-1)
        for powa, coeffa in enumerate(a):
            for powb, coeffb in enumerate(b):
                if isinstance(coeffa, int):
                    coeffa = np.poly1d([coeffa])
                if isinstance(coeffb, int):
                    coeffb = np.poly1d([coeffb])
                res[powa+powb] += self._polymul(coeffa, coeffb)
        return [self._coeff_adjust(x) for x in res]

    def get_matrices(self, d):
        gx, hx = self.get_polynomials(d)
        q = pow(2, self.e)
        g = np.zeros([q-d, q-1], dtype=np.poly1d)
        h = np.zeros([d-1, q-1], dtype=np.poly1d)
        for i in range(q-d):
            g[i, i:len(gx)+i] = gx[::-1]
        for i in range(d-1):
            h[i, i:len(hx)+i] = hx[::]

        return g, h

    def vandermonde_matrix(self, d):
        alpha = self.find_primitive()
        q = pow(2, self.e)
        h = np.empty([d-1, q-1], dtype=np.poly1d)
        for i in range(1, d):
            for j in range(0,q-1):
                acc = 1
                for k in range(i*j):
                    acc = self._polymul(acc, alpha)
                h[i-1, j] = acc
        return h

# Interface methods (many of these methods will be removed to become tests)
    def multable(self):
        a = b = np.arange(pow(2, self.e))[1:]
        return np.array([[self._mul(x, y) for y in b] for x in a])

    def show_multable(self):
        plt.imshow(np.subtract(0, self.multable()), 'gray')
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
        m = 4
        generator_matrix = self.get_generator_matrix(self.hamming(m))
        generator_matrix = self.gauss(generator_matrix)
        control_matrix = self.get_control_matrix(generator_matrix)
        
        syndrome_table = self.compute_syndromes(control_matrix)
        
        for entry in syndrome_table:
            print(*entry)

        n = pow(2,m)-1

        print('There should be no prints happening here!')
        for i in np.arange(pow(2,n)):
            decoded = self.decode(i, control_matrix, syndrome_table)
            original = self.get_padded_poly(n, i)
            difference = sum(i != j for i, j in zip(original, decoded))
            if difference > 1:
                print(difference)
        print('If there were no prints, thats great!')

        print('There should be no prints happening here!')
        for i in np.arange(pow(2,n)):
            decoded = self.hamming_decode(i, control_matrix)
            original = self.get_padded_poly(n, i)
            difference = sum(i != j for i, j in zip(original, decoded))
            if difference > 1:
                print(difference)
        print('If there were no prints, thats great!')

    def verify_reed_solomon(self, d):
        g , h1 = self.get_matrices(d)
        h = self.vandermonde_matrix(d)

        a = np.matmul(g, h1.T)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = self._polymod(a[i, j], self.irreducibles[self.e])
        print(a)
