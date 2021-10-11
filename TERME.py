from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import eye, csr_matrix, find
from scipy.sparse.linalg import eigsh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn import manifold, datasets
import umap
import time


def locally_linear_embedding(
        X, n_neighbors, n_components, new=True, gamma=100, reg=1e-3, eigen_solver='auto', tol=1e-6,
        max_iter=100, method='standard', hessian_tol=1E-4, modified_tol=1E-12,
        random_state=None, n_jobs=None):
    if eigen_solver not in ('auto', 'arpack', 'dense'):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ('standard', 'hessian', 'modified', 'ltsa'):
        raise ValueError("unrecognized method '%s'" % method)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError("output dimension must be less than or equal "
                         "to input dimension")
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = (eigen_solver != 'dense')

    if method == 'standard':
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs, new=new, gamma=gamma)

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[::M.shape[0] + 1] += 1  # W = W - I = W - I
    N = null_space(M, n_components, k_skip=1, eigen_solver=eigen_solver,
                   tol=tol, max_iter=max_iter, random_state=random_state)

    return N


def barycenter_weights(X, Z, reg=1e-3, new=True, gamma=100):
    """Compute barycenter weights of X from Y along the first axis
    We estimate the weights to assign to each point in Y[i] to recover
    the point X[i]. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Z : array-like, shape (n_samples, n_neighbors, n_dim)
    reg : float, optional
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    """
    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    h = np.zeros(n_neighbors)
    x = np.zeros(n_neighbors)
    for i, A in enumerate(Z.transpose(0, 2, 1)): 
        neighbors = A.T

        if new:
            v_i = X[i]
            for j in range(n_neighbors):
                h[j] = neighbors[j][0] - v_i[0]
                x[j] = np.linalg.norm(neighbors[j][0:2] - v_i[0:2])
                # s = | h/x |
            s = np.absolute(np.divide(h, x, out=np.zeros(x.shape), where=x != 0))
            # normalization
            s = s / np.sum(s)
            temp = np.multiply(s.repeat(3).reshape(n_neighbors, 3), v_i - neighbors)
            temp = np.sum(temp, axis=0)
            D = (n_neighbors - 1) * X[i] - np.sum(neighbors, axis=0) -temp
        else:
            D = (n_neighbors - 1) * X[i] - np.sum(neighbors, axis=0)

        C = neighbors + D
        G = np.dot(C, C.T)

        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        # add regularization term to diagonal of G
        G.flat[::Z.shape[1] + 1] += R

        if new:
            g = np.multiply(gamma, np.identity(n_neighbors))
            w = solve(G + g, v, sym_pos=True)
            B[i, :] = w / np.sum(w) + s
        else:

            w = solve(G, v, sym_pos=True)
            B[i, :] = w / np.sum(w)

    if new:
        B = np.divide(1 - B, n_neighbors - 2)
    else:
        B = np.divide(1 - B, n_neighbors - 1)
    return B


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None, new=True, gamma=100):
    """Computes the barycenter weighted graph of k-Neighbors for points in X
    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.
    n_neighbors : int
        Number of neighbors for each sample.
    reg : float, optional
        Amount of regularization when solving the least-squares
        problem. Only relevant if mode='barycenter'. If None, use the
        default.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.
    See also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = knn._fit_X
    n_samples = X.shape[0]
    # index list of n_samples
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg, new=new, gamma=gamma)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples))


def null_space(M, k, k_skip=1, eigen_solver='arpack', tol=1E-6, max_iter=100,
               random_state=None):
    """
    Find the null space of a matrix M.

    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite

    k : integer
        Number of eigenvalues/vectors to return

    k_skip : integer, optional
        Number of low eigenvalues to skip.

    eigen_solver : string, {'auto', 'arpack', 'dense'}
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.

    tol : float, optional
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.

    max_iter : int
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``solver`` == 'arpack'.

    """
    if eigen_solver == 'auto':
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'

    if eigen_solver == 'arpack':
        random_state = check_random_state(random_state)
        # initialize with [-1,1] as in ARPACK
        v0 = random_state.uniform(-1, 1, M.shape[0])
        try:
            eigen_values, eigen_vectors = eigsh(M, k + k_skip, sigma=0.0,
                                                tol=tol, maxiter=max_iter,
                                                v0=v0)
        except RuntimeError as msg:
            raise ValueError("Error in determining null-space with ARPACK. "
                             "Error message: '%s'. "
                             "Note that method='arpack' can fail when the "
                             "weight matrix is singular or otherwise "
                             "ill-behaved.  method='dense' is recommended. "
                             "See online documentation for more information."
                             % msg)

        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])
    elif eigen_solver == 'dense':
        if hasattr(M, 'toarray'):
            M = M.toarray()
        eigen_values, eigen_vectors = eigh(
            M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True)
        index = np.argsort(np.abs(eigen_values))
        return eigen_vectors[:, index], np.sum(eigen_values)
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)


def TERME(X, n_neighbors=10, n_components=2, new=True, gamma=100):
    embedding, error = locally_linear_embedding(X, n_neighbors, n_components, new=new, gamma=gamma)
    return embedding


if __name__ == '__main__':
    data_name = "YC01_rel"
    file = "data/{}.csv".format(data_name)
    # d = pd.read_csv(file, header=None, sep=' ', index_col=False)
    X = np.loadtxt(fname=file, skiprows=1, delimiter=',')[:, :3]
    X.astype(np.float32)

    start_time = time.time()

    colors = np.zeros((3, X.shape[0]))
    for i in range(3):
        h = X[:, i]
        temp = (h - h.min()) / (h.max() - h.min())
        colors[i] = temp
    colors = colors.T

    normalization = 'max_min'
    for i in [0, 1, 2]:
        h = X[:, i]
        if normalization == 'max_min':
            temp = (h - h.min()) / (h.max() - h.min())
        elif normalization == 'mean_std':
            temp = (h - np.mean(h)) / np.std(h)
        elif normalization == 'nrom':
            temp = h / np.sum(h)
        else:
            temp = h
        X[:, i] = temp

    # ran = np.arange(100, 600, 200)
    ran = [300]

    m = TERME(X, 300, 2)
    end_time = time.time()
    print(end_time - start_time)
    # # TERME
    # for k in ran:
    #     m = TERME(X, k, 2)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o', c=colors, s=5)
    #     plt.savefig("data/manifold_data/TERME/manifold_shape/{}_TERME_{}.pdf".format(data_name, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/TERME/stable/{}_TERME_max_min_{}.csv".format(data_name, k), m, delimiter=',',
    #                fmt='%.64e')
    #     # plt.show()
    #     plt.cla()

    # for k in ran:
    #     mode = "LLE"
    #     print(mode)
    #     m = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2, method='standard').fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o', c=colors, s=5)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m, delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    # for k in ran:
    #     mode = "HLLE"
    #     print(mode)
    #     m = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2, method='hessian').fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o', c=colors, s=20)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m,
    #                delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    # for k in ran:
    #     mode = "MLLE"
    #     print(mode)
    #     m = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2, method='modified').fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o',c=colors, s=5)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m, delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    # for k in ran:
    #     mode = "LTSA"
    #     print(mode)
    #     m = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2, method='ltsa').fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o',c=colors, s=5)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m, delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    # for k in ran:
    #     mode = "T-SNE"
    #     print(mode)
    #     m = manifold.TSNE(n_components=2, init='pca', random_state=77).fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o',c=colors, s=5)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m, delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    #
    # for k in ran:
    #     mode = "Isomap"
    #     print(mode)
    #     m = manifold.Isomap(n_neighbors=k, n_components=2).fit_transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o',c=colors, s=5)
    #     plt.savefig("data/manifold_data/{}/manifold_shape/{}_{}_{}.pdf".format(mode, data_name, mode, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/{}/stable/{}_{}_max_min_{}.csv".format(mode, data_name, mode, k), m, delimiter=',', fmt='%.64e')
    #     # plt.show()
    #     plt.cla()
    #
    # for k in ran:
    #     print("UMAP")
    #     reducer = umap.UMAP(n_neighbors=k, n_components=2, metric='euclidean', random_state=42)
    #     reducer.fit(X)
    #     m = reducer.transform(X)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.scatter(m[:, 0], m[:, 1], marker='o',c=colors, s=5)
    #     plt.savefig("data/manifold_data/UMAP/manifold_shape/{}_UMAP_{}.pdf".format(data_name, k), quality=100, dpi=500, bbox_inches='tight', transparent=True, pad_inches=0)
    #     np.savetxt("data/manifold_data/UMAP/stable/{}_UMAP_max_min_{}.csv".format(data_name, k), m, delimiter=',',
    #                fmt='%.64e')
    #     # plt.show()
    #     plt.cla()



    # save and read
    # np.savetxt("data/left_TERME_{}.csv".format(k), m, delimiter=',', fmt='%.64e')
    # file = "data/left_TERME_30.csv"
    # # d = pd.read_csv(file, header=None, sep=' ', index_col=False)
    # q = np.loadtxt(fname=file, delimiter=',')[:, :3]
