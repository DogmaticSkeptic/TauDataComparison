import numpy as np
import pandas as pd
import vector
from scipy.linalg import sqrtm
import numpy.linalg as la

def unit_vector(p):
    v = np.array([p.x, p.y, p.z], dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot normalize zero‐length vector")
    return v / norm

def compute_cos_observables_vector(row: pd.Series) -> dict:
    tl = vector.obj(pt=row["TauLeptonic_pt"], eta=row["TauLeptonic_eta"],
                    phi=row["TauLeptonic_phi"], mass=row["TauLeptonic_mass"])
    th = vector.obj(pt=row["TauHadronic_pt"], eta=row["TauHadronic_eta"],
                    phi=row["TauHadronic_phi"], mass=row["TauHadronic_mass"])
    mu = vector.obj(pt=row["Muon_pt"], eta=row["Muon_eta"],
                    phi=row["Muon_phi"], mass=row["Muon_mass"])
    # build rho either from explicit fields or π±+π0
    if {"Rho_pt","Rho_eta","Rho_phi","Rho_mass"}.issubset(row.index):
        rho = vector.obj(pt=row["Rho_pt"], eta=row["Rho_eta"],
                         phi=row["Rho_phi"], mass=row["Rho_mass"])
    else:
        cp = vector.obj(pt=row["ChargedPion_pt"], eta=row["ChargedPion_eta"],
                        phi=row["ChargedPion_phi"], mass=row["ChargedPion_mass"])
        np_ = vector.obj(pt=row["NeutralPion_pt"], eta=row["NeutralPion_eta"],
                         phi=row["NeutralPion_phi"], mass=row["NeutralPion_mass"])
        rho = cp + np_

    cm = tl + th
    tl_cm = tl.boostCM_of(cm)
    th_cm = th.boostCM_of(cm)

    # helicity basis
    k = tl_cm.to_pxpypz().unit()
    p = vector.Vector(x=0, y=0, z=1)
    r = (p - p.dot(k)*k).unit()
    n = r.cross(k).unit()

    lep_cm   = mu.boostCM_of(cm).boostCM_of(tl_cm)
    had_cm   = rho.boostCM_of(cm).boostCM_of(th_cm)
    l̂ = lep_cm.to_pxpypz().unit()
    ħ = had_cm.to_pxpypz().unit()

    return {
        "leptonic_cos_r": float(l̂.dot(r)),
        "leptonic_cos_n": float(l̂.dot(n)),
        "leptonic_cos_k": float(l̂.dot(k)),
        "hadronic_cos_r": float(ħ.dot(r)),
        "hadronic_cos_n": float(ħ.dot(n)),
        "hadronic_cos_k": float(ħ.dot(k)),
    }

def compute_normalized_histogram(data: np.ndarray, bin_edges: np.ndarray):
    counts, _ = np.histogram(data, bins=bin_edges)
    widths    = np.diff(bin_edges)
    area      = np.sum(counts * widths)
    if area == 0:
        return np.zeros_like(counts, float), np.zeros_like(counts, float)
    norm = counts / area
    err  = np.sqrt(counts) / area
    return norm, err

def compute_raw_histogram(data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Compute raw (unnormalized) histogram counts for `data` given `bin_edges`.
    Returns an array of counts (length = len(bin_edges)-1).
    """
    counts, _ = np.histogram(data, bins=bin_edges)
    return counts

def compute_density_matrix(BA, BB, C):
    sigma_x = np.array([[0, 1],[1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],[0, -1]], dtype=complex)
    paulis = {"r": sigma_x, "n": sigma_y, "k": sigma_z}
    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)
    rho = I4.copy()
    for i, key in enumerate(["r", "n", "k"]):
        rho += BA[i] * np.kron(paulis[key], I2)
        rho += BB[i] * np.kron(I2, paulis[key])
        for j, key2 in enumerate(["r", "n", "k"]):
            rho += C[i, j] * np.kron(paulis[key], paulis[key2])
    return rho / 4.0

def compute_concurrence(rho):
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    R = rho @ np.kron(sigma_y, sigma_y) @ np.conjugate(rho) @ np.kron(sigma_y, sigma_y)
    eigvals = np.sort(np.sqrt(np.abs(la.eigvals(R))))[::-1]
    return max(0, eigvals[0] - np.sum(eigvals[1:]))

# ----------------------------------------------------------------------
# full‐density‐matrix eigenvalue calculator (from your snippet)
def compute_full_density_matrix(C: dict[str, float],
                                B: dict[str, float]) -> np.ndarray:
    # Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I2      = np.eye(2, dtype=complex)

    def kron(a, b):
        return np.kron(a, b)

    # reconstruct ρ
    rho = (1/4) * (
        np.eye(4, dtype=complex)
        + B['Bk'] * kron(pauli_x, I2) + B['Br'] * kron(pauli_y, I2) + B['Bn'] * kron(pauli_z, I2)
        + B['Ak'] * kron(I2, pauli_x) + B['Ar'] * kron(I2, pauli_y) + B['An'] * kron(I2, pauli_z)
        + C['kk'] * kron(pauli_x, pauli_x) + C['rr'] * kron(pauli_y, pauli_y) + C['nn'] * kron(pauli_z, pauli_z)
        + C['kr'] * kron(pauli_x, pauli_y) + C['kn'] * kron(pauli_x, pauli_z) + C['rn'] * kron(pauli_y, pauli_z)
        + C['rk'] * kron(pauli_y, pauli_x) + C['nr'] * kron(pauli_z, pauli_x) + C['nk'] * kron(pauli_z, pauli_y)
    )

    # spin‐flipped
    pauli_y2 = kron(pauli_y, pauli_y)
    rho_tilde = pauli_y2 @ np.conjugate(rho) @ pauli_y2

    # R = √ρ ρ˜ √ρ
    sqrt_rho = sqrtm(rho)
    R        = sqrt_rho @ rho_tilde @ sqrt_rho

    # eigenvalues λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄
    vals, _ = la.eig(R)
    vals     = np.real(vals)
    vals.sort()
    return vals[::-1]

def compute_best_bell_violation(C_vals, C_unc):
    best_violation = -np.inf
    best_pair = None
    best_sign = None
    best_unc = None
    for i in range(3):
        for j in range(i+1, 3):
            for sign in [1, -1]:
                value = abs(C_vals[i, i] + sign * C_vals[j, j]) - np.sqrt(2)
                unc = np.sqrt(C_unc[i, i]**2 + C_unc[j, j]**2)
                if value > best_violation:
                    best_violation = value
                    best_pair = (i, j)
                    best_sign = '+' if sign == 1 else '-'
                    best_unc = unc
    return best_pair, best_sign, best_violation, best_unc

def assemble_C_numeric_and_uncertainty(c_results, dataset, method, product_observables):
    C_vals = np.zeros((3, 3))
    C_unc = np.zeros((3, 3))
    idx_map = {"r": 0, "n": 1, "k": 2}
    for _, col in product_observables:
        letters = col.split("_")[-1]
        i = idx_map[letters[0]]
        j = idx_map[letters[1]]
        try:
            val, unc = c_results[col][dataset][method]
        except KeyError:
            val, unc = np.nan, np.nan
        C_vals[i, j] = val
        C_unc[i, j] = unc
    return C_vals, C_unc


