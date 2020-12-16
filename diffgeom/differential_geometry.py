import sympy as sp
import numpy as np


def christoffel_symbol(g_cov, g_cont, coords, indices):
    """ Computes a single christoffel coefficient """
    dim = len(coords)
    a, b, c = indices
    a_coord, b_coord, c_coord = coords[a], coords[b], coords[c]
    christoffel = 0
    for d in range(dim):
        d_coord = coords[d]
        christoffel += g_cont[a, d] * (
            sp.diff(g_cov[d, c], b_coord)
            + sp.diff(g_cov[b, d], c_coord)
            - sp.diff(g_cov[b, c], d_coord)
        )
    return 1 / 2 * christoffel


def christoffel_symbols(g_cov, g_cont, coords):
    """ Computes all christoffel symbols """
    dim = len(coords)
    christoffel_array = np.zeros([dim, dim, dim], dtype=object)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                christoffel_array[a, b, c] = christoffel_symbol(
                    g_cov, g_cont, coords, [a, b, c]
                )
    return christoffel_array


def ricci_component(christoffel_array, coords, indices):
    """ Compute a component of the Ricci curvature tensor, rank (0, 2) """
    dim = len(coords)
    a, b = indices
    a_coord, b_coord = coords[a], coords[b]
    ricci = 0
    for c in range(dim):
        c_coord = coords[c]
        ricci += sp.diff(christoffel_array[c, a, b], c_coord) - sp.diff(
            christoffel_array[c, a, c], b_coord
        )
        for d in range(dim):
            ricci += (
                christoffel_array[c, a, b] * christoffel_array[d, c, d]
                - christoffel_array[d, b, c] * christoffel_array[c, a, d]
            )
    return ricci


def ricci_tensor(christoffel_array, coords):
    """ Compute all ricci tensor components, rank(0, 2) """
    dim = len(coords)
    ricci_array = np.zeros([dim, dim], dtype=object)
    for a in range(dim):
        for b in range(dim):
            ricci_array[a, b] = -ricci_component(christoffel_array, coords, [a, b])
    return ricci_array


def riemann_component_1_3(christoffel_array, coords, indices):
    """ Compute a component of the rank (1, 3) Riemann curvature tensor """
    dim = len(coords)
    a, b, c, d = indices
    a_coord, b_coord, c_coord, d_coord = coords[a], coords[b], coords[c], coords[d]
    riemann = \
            sp.diff(christoffel_array[a, d, b], c_coord) - \
            sp.diff(christoffel_array[a, c, b], d_coord)
    for i in range(dim):
        riemann += christoffel_array[a, c, i] * christoffel_array[i, d, b]
        riemann -= christoffel_array[a, d, i] * christoffel_array[i, c, b]
    return riemann


def riemann_tensor(christoffel_array, coords, riemann_component_func):
    """ Computes all elements of the Riemann curvature tensor, rank specified by 3. argument """
    dim = len(coords)
    riemann_array = np.zeros([dim, dim, dim, dim], dtype=object)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    riemann_array[a, b, c, d] = riemann_component_func(
                        christoffel_array, coords, [a, b, c, d]
                    )
    return riemann_array


def riemann_tensor_cov(riemann_1_3, g_cov):
    """ Contract first index of (1, 3) Riemann with (0,2) metric tensor """
    dim = riemann_1_3.shape[0]
    riemann_array_cov = np.zeros([dim, dim, dim, dim], dtype=object)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    for i in range(dim):
                        riemann_array_cov[a, b, c, d] += \
                            g_cov[a, i] * riemann_1_3[i, b, c, d]
    return riemann_array_cov


def riemann_tensor_cont(riemann_1_3, g_cont):
    """ Contract three last indices of (1, 3) Riemann with (2, 0) metric tensor """
    dim = riemann_1_3.shape[0]
    riemann_array_cont = np.zeros([dim, dim, dim, dim], dtype=object)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    for i in range(dim):
                        for j in range(dim):
                            for k in range(dim):
                                riemann_array_cont[a, b, c, d] += \
                                    g_cont[b, i] * \
                                    g_cont[c, j] * \
                                    g_cont[d, k] * \
                                    riemann_1_3[a, i, j, k]
    return riemann_array_cont


def kretschmann_scalar(riemann_cov, riemann_cont):
    """ Computes the kretschmann_scalar """
    dim = riemann_cov.shape[0]
    return sp.simplify(np.sum(riemann_cov * riemann_cont))


if __name__ == "__main__":
    # Define symbols/coordinates
    coords = sp.symbols("t r theta phi")
    t, r, theta, phi = coords

    # General static, isomorphic metric
    g_cov = sp.diag("A(r)", "-B(r)", -(r ** 2), -(r ** 2) * sp.sin(theta) ** 2)
    g_cont = g_cov.inv()
    print("Metric: ")
    sp.pprint(g_cov)  # print the metric
    print("")

    # compute all christoffel connection coeffs
    christoffel_array = christoffel_symbols(g_cov, g_cont, coords)

    # Ricci tensor components
    ricci_array = ricci_tensor(christoffel_array, coords)
    print("Non zero Ricci tensor components (diagonal)")
    # print Ricci tensor components
    for i in range(len(coords)):
        # Only printing diagonal elements, since we now for this metric that
        # the ricci tensor is diagonal
        sp.pprint(sp.nsimplify(ricci_array[i, i]))
    print("")

    # Riemann tensor, rank (1, 3)
    riemann_array = riemann_tensor(christoffel_array, coords, riemann_component_1_3)
    # Riemann tensor of rank (0, 4)
    riemann_array_cov = riemann_tensor_cov(riemann_array, g_cov)
    # Riemann tensor of rank (4, 0)
    riemann_array_cont = riemann_tensor_cont(riemann_array, g_cont)

    # The Kretschmann scalar
    kretschmann = kretschmann_scalar(riemann_array_cov, riemann_array_cont)
    print("The kretschmann scalar")
    sp.pprint(kretschmann)
