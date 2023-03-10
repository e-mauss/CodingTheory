from GF2 import GF
import numpy as np

def test_associativity():
    for i in range(2, 9):
        gf = GF(i)

        a = 2
        b = pow(2, 7)
        c = pow(2, 8)-1
        ab = gf._mul(a, b)
        bc = gf._mul(b, c)
        assert gf._mul(ab, c) == gf._mul(a, bc)

def test_commutativity():
    for i in range(2, 9):
        gf = GF(i)

        a = 2
        b = pow(2, 8)-1
        ab = gf._mul(a, b)
        ba = gf._mul(b, a)
        assert ab == ba


def test_inverses():
    for i in range(2, 9):
        gf = GF(i)

        for num in np.arange(1, pow(2, i)):
            inv = gf._inverse(num)
            assert gf._mul(num, inv) == 1

def test_gauss():
    gf = GF(3)
    generator_matrix = [[1,1,1,1,1,1,1],[0,1,1,1,0,1,1],[1,0,1,0,1,0,1],[0,0,0,1,0,0,0]]
    generator_matrix = gf.gauss(generator_matrix)

    for i, row in enumerate(generator_matrix):
        for j, element in enumerate(row[0:4]):
            if i == j:
                assert element == 1
            else:
                assert element == 0 

def test_hamming():
    gf = GF(8)
    m = 4
    control_matrix = gf.hamming(m)
    generator_matrix = gf.get_generator_matrix(control_matrix)
    
    syndrome_table = gf.compute_syndromes(control_matrix)
    
    n = pow(2,m)-1

    for i in np.arange(pow(2,n)):
        decoded = gf.hamming_decode(i, control_matrix)
        original = gf.get_padded_poly(n, i)
        difference = sum(i != j for i, j in zip(original, decoded))
        assert difference == 1 or difference == 0
        assert (np.matmul([decoded], control_matrix.T)%2 == 0).all() 

def test_reed_solomon():
    gf = GF(4)
    d = 5

    g , h = gf.get_matrices(d)

    a = np.matmul(g, h.T)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = gf._mod(a[i, j], gf.irreducibles[gf.e])
    assert (a == 0).all()