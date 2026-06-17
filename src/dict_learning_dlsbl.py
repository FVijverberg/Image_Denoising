import numpy as np


def e_step(D, gamma, Y, sigma2, chunk=256):
    # Posterior mean and covariance of each sparse vector x_k.
    # given the current dictionary A and hyperparameters gamma, what do we estimate the codes X to be
    # Patches are processed in chunks so the small (m x m) inverse Phi_k can be batched 

    m, N = D.shape #patch dim and num atoms
    K = Y.shape[1] # num patches
    I = np.eye(m)
    DtY = D.T @ Y # Precompute

    M = np.zeros((N, K)) # for posterior mean
    diag_Sig = np.zeros((N, K)) # post cov. per patch
    Sigma_sum = np.zeros((N, N))# accumulate cov sum for dictionary update

    for s in range(0, K, chunk):
        g = gamma[:, s:s + chunk]# get chunk
        DG = D[None] * g.T[:, None, :]  #add batch dimension and reshape gamma -  is effectively A * diag(gamma_k) for each patch in the batch simultaneously
        Phi = np.linalg.inv(sigma2 * I + DG @ D.T)# (b, m, m)
        Sigma = (g.T[:, :, None] * np.eye(N) #posterior cov.
                 - np.swapaxes(DG, 1, 2) @ Phi @ DG) # transpose last two dims for matrix multiply
        v = DtY[:, s:s + chunk].T[:, :, None] # b x N x 1
        mu = (Sigma @ v)[:, :, 0] / sigma2  # post. mean (bxN)

        M[:, s:s + chunk] = mu.T
        diag_Sig[:, s:s + chunk] = np.diagonal(Sigma, axis1=1, axis2=2).T # for gamma update
        Sigma_sum += Sigma.sum(axis=0)

    Sigma_bar = Sigma_sum + M @ M.T #combined second moment for dict update
    return M, diag_Sig, Sigma_bar


def am_dictionary_update(A, Sigma_bar, YMt, max_sweeps=20, tol=1e-7):
    # Alternating-Minimization update of A
    # Minimises  g(A) = -Tr(M Y^T A) + 0.5 Tr(A (Sigma_bar - diag(Sigma_bar)) A^T)
    # over unit-norm columns, one col at a time 

    A = A.copy()
    N = A.shape[1]
    for _ in range(max_sweeps):
        A_old = A.copy()
        for i in range(N):
            # v_i = (Y M^T)_i - sum_{j != i} Sigma_bar[i, j] A_j
            v = YMt[:, i] - A @ Sigma_bar[:, i] + A[:, i] * Sigma_bar[i, i] # direction for atom i - grad - infleunce of small atoms
            nv = np.linalg.norm(v) #unit norm
            if nv > 1e-12:   # skip update if v_i == 0 
                A[:, i] = v / nv
        if np.linalg.norm(A - A_old) < tol:
            break
    return A


def train_dlsbl(Y, n_atoms, sigma_train, max_em=500, em_tol=1e-3, seed=42, verbose=True):
    #Learn a dictionary from patches Y with DL-SBL

    rng = np.random.default_rng(seed)
    m, K = Y.shape

    # remove dc
    means = Y.mean(axis=0, keepdims=True)
    Yc = Y - means

    sigma2 = sigma_train ** 2 # Noise

    # Initialise A from random data patches
    idx = rng.choice(K, size=n_atoms, replace=False)
    A = Yc[:, idx].copy()
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12 #normalize
    gamma = np.ones((n_atoms, K)) #gamma_k = 1

    for r in range(max_em):
        A_prev, gamma_prev = A.copy(), gamma.copy()

        # E-step
        M, diag_Sig, Sigma_bar = e_step(A, gamma, Yc, sigma2)

        # M-step
        gamma = M ** 2 + diag_Sig #hyperparams
        A = am_dictionary_update(A, Sigma_bar, Yc @ M.T) # dict update

        rel_change = (np.linalg.norm(A - A_prev) / (np.linalg.norm(A) + 1e-12)+ np.linalg.norm(gamma - gamma_prev) / (np.linalg.norm(gamma) + 1e-12))
        if verbose and (r % 10 == 0 or rel_change < em_tol):
            mse = np.mean((Yc - A @ M) ** 2)
            print(f"EM {r + 1:3d}/{max_em} | recon MSE {mse:8.4f} | "
                  f"rel_change {rel_change:.3e}")
        if rel_change < em_tol:
            break

    return A, M, gamma, means, sigma2


def sbl_recover(Y, A, sigma2, n_iter=30, chunk=512):
    #Per-signal SBL sparse recovery with a fixed dictionary (inference).

    m, N = A.shape
    P = Y.shape[1]
    I = np.eye(m)
    X = np.zeros((N, P))

    for s in range(0, P, chunk):
        y = Y[:, s:s + chunk] # (m, b)
        g = np.ones((N, y.shape[1]))# gamma per signal (init to 1)
        mu = np.zeros_like(g.T)
        for _ in range(n_iter): # iterate sbl fixed point eqns
            AG = A[None] * g.T[:, None, :]# b x m x n
            Phi = np.linalg.inv(sigma2 * I + AG @ A.T)# b x m x m
            Py = (Phi @ y.T[:, :, None])[:, :, 0]# (b,m) = Phi_k y_k
            mu = g.T * (Py @ A) #(b, N) = gamma A^T Phi y
            q = (A[None] * (Phi @ A[None])).sum(axis=1) # (b, N) = diag(A^T Phi A)
            sig = g.T - g.T ** 2 * q  # (b, N) = diag(Sigma_k)
            g = (mu ** 2 + sig).T # (N, b) update gamma
        X[:, s:s + chunk] = mu.T
    return X


if __name__ == "__main__":
    import time

    Y = np.load("../Data/Y_patches.npy") # training patches

    for sigma_train in [5, 10, 15, 25]:
        print(f"\n=== DL-SBL  sigma={sigma_train} ===")
        noise_rng = np.random.default_rng(sigma_train)
        Y_noisy = Y + noise_rng.normal(0, sigma_train, size=Y.shape)

        t0 = time.time()
        A, M, gamma, means, sigma2 = train_dlsbl(Y_noisy, n_atoms=128, sigma_train=float(sigma_train), max_em=500, seed=42)
        elapsed = time.time() - t0

        energy = M **2
        order = np.argsort(-energy, axis=0) # from highest to lowest energy for each patch
        cum = np.cumsum(np.take_along_axis(energy, order, axis=0), axis=0) # reorder according to order
        active = np.argmax(cum >= 0.99 * energy.sum(0, keepdims=True), axis=0) + 1
        print(f"Learned dictionary: {A.shape[1]} atoms, sigma2 = {sigma2:.1f}")
        print(f"Per-patch atoms for 99% energy -- mean {active.mean():.1f}, "
              f"median {np.median(active):.0f}, max {active.max()}")
        print(f"Training time: {elapsed:.1f} s ({elapsed/60:.1f} min)")

        np.save(f"../Data/dictionary_D_sbl_{sigma_train}.npy", A)
        np.save(f"../Data/sparse_X_sbl_{sigma_train}.npy", M)
        np.save(f"../Data/patch_means_sbl_{sigma_train}.npy", means)
        np.save(f"../Data/sigma2_sbl_{sigma_train}.npy", sigma2)
        print(f"Saved DL-SBL artifacts to ../Data/*_sbl_{sigma_train}.npy")