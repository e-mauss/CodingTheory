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
        """Constructor

        Args:
            e ( Integer ): The degree of extension
        """
        self.e = e

# Num and Poly conversions
    def _num2poly(self, x: int | np.poly1d):
        """Converts an Integer to a Polynomial.

        Args:
            x (int | np.poly1d): The element to be converted
        """
        if type(x) is int or type(x) is np.int_:
            return np.poly1d([int(x) for x in bin(x)[2:]])
        elif type(x) is np.poly1d:
            return x

    def _nums2polys(self, *arg):
        """Convert multiple Integers to Polynomials.
        """
        return tuple(self._num2poly(num) for num in arg)

    def _poly2num(self, x: int | np.poly1d):
        """Convert a Polynomial to an Integer.

        Args:
            x (int | np.poly1d): The element to be converted
        """
        if type(x) is int or type(x) is np.int_:
            return x
        elif type(x) is np.poly1d:
            return int("".join([str(int(i % 2)) for i in x.c]), 2)

    def _polys2nums(self, *arg):
        """Convert multiple Polynomials to Integers.
        """
        return tuple(self._poly2num(num) for num in arg)

    def _coeff_adjust(self, poly):
        """Adjusts coefficients of a Polynomial to be modulo 2.

        Args:
            poly (np.poly1d): The polynomial to be adjusted
        """
        return np.poly1d([x % 2 for x in poly.c])

# Mathematical operations with polynomial returns
    def _polymod(self, x, m):
        """Reduces an element according to the module.

        Args:
            x (int | np.poly1d): The element to be reduced
            m (int | np.poly1d): The modulus

        Returns:
            np.poly1d
        """
        x, m = self._nums2polys(x, m)
        return self._coeff_adjust((x/m)[1])

    def _polydiv(self, x, d):
        """Computes the quotient of two elements.

        Args:
            x (int | np.poly1d): The first operand
            d (int | np.poly1d): The second operand

        Returns:
            np.poly1d
        """
        x, d = self._nums2polys(x, d)
        return self._coeff_adjust((x/d)[0])

    def _polymul(self, x, y):
        """Computes the product of two elements.

        Args:
            x (int | np.poly1d): The first operand
            d (int | np.poly1d): The second operand

        Returns:
            np.poly1d
        """
        x, y, m = self._nums2polys(x, y, self.irreducibles[self.e])
        return self._polymod(x*y, m)

    def _polyinverse(self, x):
        """Comptes the inverse of an element.

        Args:
            x (int | np.poly1d): The element to be inverted

        Returns:
            np.poly1d
        """
        a, b = self._nums2polys(x, self.irreducibles[self.e])
        _, inv, _ = self._eGCD(a, b)
        return inv

    def get_padded_poly(self, length, poly):
        """Pads the array representing a polynomial to a certain length.

        Args:
            length (int): The target length
            poly (int | np.poly1d): The polynomial to be padded

        Returns:
            list
        """
        errpoly = self._num2poly(poly).c
        errpoly = np.pad(errpoly, [(length-len(errpoly), 0)])
        return errpoly

    def _eGCD(self, a, b):
        """The extended euclidean algorithm for polynomials.

        Args:
            a (int | np.poly1d): The first element
            b (int | np.poly1d): The second element
        """
        if self._poly2num(a) == 0:
            return b, 0, 1

        gcd, x1, y1 = self._eGCD(self._polymod(b, a), a)

        x = y1 - self._polymul(self._polydiv(b, a), x1)

        return gcd, x, x1

# Mathematical operations with integer representation returns
    def _mod(self, x, m):
        """Reduces an element according to the module.

        Args:
            x (int | np.poly1d): The element to be reduced
            m (int | np.poly1d): The modulus

        Returns:
            int
        """
        return self._poly2num(self._polymod(x, m))

    def _div(self, x, d):
        """Computes the quotient of two elements.

        Args:
            x (int | np.poly1d): The first operand
            d (int | np.poly1d): The second operand

        Returns:
            int
        """
        return self._poly2num(self._polydiv(x, d))

    def _mul(self, x, y):
        """Computes the product of two elements.

        Args:
            x (int | np.poly1d): The first operand
            d (int | np.poly1d): The second operand

        Returns:
            int
        """
        return self._poly2num(self._polymul(x, y))

    def _inverse(self, x):
        """Comptes the inverse of an element.

        Args:
            x (int | np.poly1d): The element to be inverted

        Returns:
            int
        """
        return self._poly2num(self._polyinverse(x))

# Functions for gaussian algorithm
    def gauss(self, generator):
        """Computes the systematic generatormatrix of a linear code.

        Args:
            generator (np.ndarray): The generatormatrix in its basic form
        """
        mat = np.array(generator)
        self.forward_elim(mat)
        self.backward_sub(mat)
        return mat

    def forward_elim(self, mat):
        """Forward elimination of the gaussian algorithm.
        """
        for i in np.arange(min(mat.shape)):
            if mat[i, i] == 0:
                self.get_one(mat, i)
            self.zero_below(mat, i)

    def backward_sub(self, mat):
        """Backward substitution of the gaussian algorithm.
        """
        for i in reversed(np.arange(min(mat.shape))):
            self.zero_above(mat, i)

    def zero_below(self, mat, i):
        """Zeroes columns below the main diagonal.
        """
        for j, x in enumerate(mat[i+1:, i]):
            if x != 0:
                mat[j+i+1] = (mat[i]+mat[j+i+1]) % 2

    def zero_above(self, mat, i):
        """Zeroes columns above the main diagonal.
        """
        for j, x in enumerate(mat[:i, i]):
            if x != 0:
                mat[j] = (mat[i]+mat[j]) % 2

    def get_one(self, mat, i):
        """Brings a 1 to an entry on the main diagonal.
        """
        for n in np.arange(i, mat.shape[0]):
            if mat[n, i] == 1:
                mat[[i, n]] = mat[[n, i]]

# Linear code utility
    def check_canonical(self, mat):
        """Checks whether a given matrix is canonical.

        Args:
            mat (np.ndarray): Matrix to be checked

        Returns:
            boolean
        """
        canonical = True
        for i in np.arange(min(mat.shape)):
            if np.sum(mat[:, i]) != 1 or mat[i, i] != 1:
                canonical = False
                break
        return canonical

    def get_generator_matrix(self, mat):
        """Creates the generatormatrix from a given controlmatrix. Controlmatrix needs to be systematic.

        Args:
            mat (np.ndarray): The systematic controlmatrix
        """
        n_mat = max(mat.shape)
        k_mat = n_mat - min(mat.shape)
        gen = np.zeros([k_mat, k_mat])
        for x in np.arange(k_mat):
            gen[x, x] = 1

        gen = np.hstack([gen, mat[:, :k_mat].T])
        return gen

    def get_control_matrix(self, mat):
        """Creates the controlmatrix from a given generatormatrix. Generatormatrix needs to be systematic.

        Args:
            mat (np.ndarray): The systematic generatormatrix
        """
        k_mat = min(mat.shape)
        diff = max(mat.shape) - k_mat
        control = np.zeros([diff, diff])
        for x in np.arange(diff):
            control[x, x] = 1
        control = np.hstack([mat[:, k_mat:].T, control])
        return control

    def compute_syndromes(self, h):
        """Computes the syndrometable of a given controlmatrix.

        Args:
            h (np.ndarray): The Controlmatrix
        """
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
        """Decodes a message using a given Controlmatrix and its syndrometable.

        Args:
            message (list): The message to be verified and corrected
            control (np.ndarray): The controlmatrix to be used
            syndromes (list): The syndrometable
        """
        errpoly = self.get_padded_poly(max(control.shape), message)
        res = np.matmul([errpoly], control.T) % 2
        for codeword, syndrome in syndromes:
            if (syndrome == res).all():
                return ((errpoly - codeword) % 2)
    
# Hamming code utility
    def hamming(self, m):
        """Generates the controlmatrix of a hamming code.

        Args:
            m ( Integer ): The m parameter of a hamming code
        """
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
        """Decodes a message using a given hamming code's controlmatrix.

        Args:
            message (list): The message to be verified and corrected
            control (np.ndarray): The controlmatrix to be used
        """
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
        """ Creates the Generator Matrix of a Reed Muller Code

        Args:
            r ( Integer ): The r parameter
            m ( Integer ): The m parameter
        """
        if r == 0:
            return np.ones((1,pow(2,m)))
        if r > m:
            return self.reed_muller(m, m)

        tl = self.reed_muller(r, m-1)
        br = self.reed_muller(r-1, m-1)
        bl = np.zeros((br.shape[0], tl.shape[1]))
        return self.forge_four_corners(tl, br, bl)

    def forge_four_corners(self, tl, br, bl):
        """Joins three matrices according to the Reed Muller Code recursion

        Args:
            tl ( np.ndarray ): Top left corner
            br ( np.ndarray ): Bottom right corner
            bl ( np.ndarray ): Bottom left corner
        """
        top = np.hstack((tl, tl))
        bot = np.hstack((bl, br))
        return np.vstack((top, bot))

# Reed solomon codes
    def find_primitive(self):
        """Finds the primitive element of the Galois Field
        """
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
        """Creates the generator and control polynomials using the primitive element

        Args:
            d ( Integer ): The minimal distance parameter of a Reed Solomon Code RS(q, d)
        """
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
        """Multiplies two nested polynomials

        Args:
            a ( list ): List containing Integers or Polynomials
            b ( list ): List containing Integers or Polynomials

        Returns:
            list: Nested polynomial product
        """
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
        """Creates the generator and control polynomials using the primitive element

        Args:
            d ( Integer ): The minimal distance parameter of a Reed Solomon Code RS(q, d)
        """
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
        """Creates the vandermonde matrix of a Reed Solomon Code

        Args:
            d ( Integer ): The minimal distance parameter of a Reed Solomon Code RS(q, d)
        """
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
        """Creates the multiplication table of the Galois Field
        """
        a = b = np.arange(pow(2, self.e))[1:]
        return np.array([[self._mul(x, y) for y in b] for x in a])

    def show_multable(self):
        """Shows the multiplication table as an image
        """
        plt.imshow(np.subtract(0, self.multable()), 'gray')
        plt.show()

    def inverses(self):
        """Compute and print all elements of the Galois Field and their inverses
        """
        for num in np.arange(1, pow(2, self.e)):
            inv = self._inverse(num)
            print(f"u = {num:>4} | u⁻¹ = {inv:>4}")

    def verify_inverses(self):
        """Verify that an element multiplied with its inverse is equal to zero. Print results.
        """
        for num in np.arange(1, pow(2, self.e)):
            inv = self._inverse(num)
            print(f"u * u⁻¹ = {self._mul(num, inv)}")

    def verify_associativity(self):
        """Verifies associativity. Prints results.
        """
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
