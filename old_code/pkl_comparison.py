import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
import vector

# Enable tqdm on pandas
tqdm.pandas()

# =============================================================================
# Configuration
# =============================================================================
CSV_FILE = "./data/aaron_code_data.csv"
PKL_FILE = "./data/yulei_code_data.pkl"
OUTDIR = "cos_comparison_plots"
BINS = np.linspace(-1.0, 1.0, 51)
MAX_EVENTS = 10000

# Analyzing powers
KAPPA_LEP = -0.34   # for the muonic (leptonic) tau
KAPPA_HAD = 0.41    # for the hadronic tau

# B‑observables: (xlabel, column name)
B_OBSERVABLES = [
    (r"$\cos\theta_{r}^{\mathrm{lep}}$", "leptonic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{lep}}$", "leptonic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{lep}}$", "leptonic_cos_k"),
    (r"$\cos\theta_{r}^{\mathrm{had}}$", "hadronic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{had}}$", "hadronic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{had}}$", "hadronic_cos_k")
]

# Product (C) observables
LEP_KEYS = ["leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k"]
HAD_KEYS = ["hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"]
PRODUCT_OBSERVABLES = []
for lep in LEP_KEYS:
    for had in HAD_KEYS:
        col = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
        label = r"$C_{" + f"{lep.split('_')[-1]}{had.split('_')[-1]}" + "}$"
        PRODUCT_OBSERVABLES.append((label, col))

# =============================================================================
# Utility Functions
# =============================================================================
def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def compute_cos_observables_vector(row: pd.Series) -> dict:
    tau_lep = vector.obj(pt=row["TauLeptonic_pt"], eta=row["TauLeptonic_eta"],
                         phi=row["TauLeptonic_phi"], mass=row["TauLeptonic_mass"])
    tau_had = vector.obj(pt=row["TauHadronic_pt"], eta=row["TauHadronic_eta"],
                         phi=row["TauHadronic_phi"], mass=row["TauHadronic_mass"])
    lepton = vector.obj(pt=row["Muon_pt"], eta=row["Muon_eta"],
                        phi=row["Muon_phi"], mass=row["Muon_mass"])
    pion_ch = vector.obj(pt=row["ChargedPion_pt"], eta=row["ChargedPion_eta"],
                         phi=row["ChargedPion_phi"], mass=row["ChargedPion_mass"])
    pion0 = vector.obj(pt=row["NeutralPion_pt"], eta=row["NeutralPion_eta"],
                       phi=row["NeutralPion_phi"], mass=row["NeutralPion_mass"])
    rho = pion_ch + pion0
    ditau = tau_lep + tau_had

    # -- Leptonic tau --
    lep_cm = tau_lep.boostCM_of(ditau)
    k_hat = unit_vector(np.array([lep_cm.x, lep_cm.y, lep_cm.z]))
    z = np.array([0, 0, 1])
    r_hat = unit_vector(z - np.dot(z, k_hat) * k_hat)
    n_hat = np.cross(k_hat, r_hat)
    lep_rest = lepton.boostCM_of(tau_lep)
    lep_dir = unit_vector(np.array([lep_rest.x, lep_rest.y, lep_rest.z]))
    leptonic_cos_r = np.dot(lep_dir, r_hat)
    leptonic_cos_n = np.dot(lep_dir, n_hat)
    leptonic_cos_k = np.dot(lep_dir, k_hat)

    # -- Hadronic tau --
    had_cm = tau_had.boostCM_of(ditau)
    k2 = unit_vector(np.array([had_cm.x, had_cm.y, had_cm.z]))
    r2 = unit_vector(z - np.dot(z, k2) * k2)
    n2 = np.cross(k2, r2)
    rho_rest = rho.boostCM_of(tau_had)
    rho_dir = unit_vector(np.array([rho_rest.x, rho_rest.y, rho_rest.z]))
    hadronic_cos_r = np.dot(rho_dir, r2)
    hadronic_cos_n = np.dot(rho_dir, n2)
    hadronic_cos_k = np.dot(rho_dir, k2)

    return {
        "leptonic_cos_r": leptonic_cos_r,
        "leptonic_cos_n": leptonic_cos_n,
        "leptonic_cos_k": leptonic_cos_k,
        "hadronic_cos_r": hadronic_cos_r,
        "hadronic_cos_n": hadronic_cos_n,
        "hadronic_cos_k": hadronic_cos_k
    }

def compute_normalized_histogram(data: np.ndarray, bins: np.ndarray):
    counts, _ = np.histogram(data, bins=bins)
    widths = np.diff(bins)
    area = np.sum(counts * widths)
    if area == 0:
        return np.zeros_like(counts), np.zeros_like(counts)
    norm = counts / area
    err = np.sqrt(counts) / area
    return norm, err

def model_B(x: np.ndarray, B: float, kappa: float) -> np.ndarray:
    return 0.5 * (1 + kappa * B * x)

def model_C(x: np.ndarray, C: float, k0: float, k1: float) -> np.ndarray:
    eps = 1e-6
    return -0.5 * (1 + k0 * k1 * C * x) * np.log(np.abs(x) + eps)

def overlay_plot_B(bins, n1, e1, n2, e2, xlabel, path, kappa):
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(6,4))
    plt.bar(centers, n1, width=np.diff(bins), alpha=0.5,
            label="CSV", color="blue", edgecolor="black")
    plt.errorbar(centers, n1, yerr=e1, fmt="none", color="black", capsize=2)
    plt.bar(centers, n2, width=np.diff(bins), alpha=0.5,
            label="PKL", color="red", edgecolor="black")
    plt.errorbar(centers, n2, yerr=e2, fmt="none", color="black", capsize=2)

    try:
        popt1, _ = curve_fit(lambda x,B: model_B(x,B,kappa),
                             centers, n1, p0=[0.0], sigma=e1,
                             absolute_sigma=True)
        plt.plot(centers, model_B(centers, popt1[0], kappa),
                 "b-", label=f"Aaron Analysis Data B={popt1[0]:.3f}")
    except:
        pass

    try:
        popt2, _ = curve_fit(lambda x,B: model_B(x,B,kappa),
                             centers, n2, p0=[0.0], sigma=e2,
                             absolute_sigma=True)
        plt.plot(centers, model_B(centers, popt2[0], kappa),
                 "r-", label=f"Yulei Code Data B={popt2[0]:.3f}")
    except:
        pass

    plt.xlabel(xlabel)
    plt.ylabel("Normalized counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def overlay_plot_C(bins, n1, e1, n2, e2, xlabel, path):
    centers = 0.5 * (bins[:-1] + bins[1:])
    plt.figure(figsize=(6,4))
    plt.bar(centers, n1, width=np.diff(bins), alpha=0.5,
            label="CSV", color="blue", edgecolor="black")
    plt.errorbar(centers, n1, yerr=e1, fmt="none", color="black", capsize=2)
    plt.bar(centers, n2, width=np.diff(bins), alpha=0.5,
            label="PKL", color="red", edgecolor="black")
    plt.errorbar(centers, n2, yerr=e2, fmt="none", color="black", capsize=2)

    try:
        popt1, _ = curve_fit(lambda x,C: model_C(x,C,KAPPA_LEP,KAPPA_HAD),
                             centers, n1, p0=[0.0], sigma=e1,
                             absolute_sigma=True)
        plt.plot(centers, model_C(centers, popt1[0], KAPPA_LEP, KAPPA_HAD),
                 "b-", label=f"CSV fit C={popt1[0]:.3f}")
    except:
        pass

    try:
        popt2, _ = curve_fit(lambda x,C: model_C(x,C,KAPPA_LEP,KAPPA_HAD),
                             centers, n2, p0=[0.0], sigma=e2,
                             absolute_sigma=True)
        plt.plot(centers, model_C(centers, popt2[0], KAPPA_LEP, KAPPA_HAD),
                 "r-", label=f"PKL fit C={popt2[0]:.3f}")
    except:
        pass

    plt.xlabel(xlabel)
    plt.ylabel("Normalized counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load and truncate CSV
    df_csv = pd.read_csv(CSV_FILE).head(MAX_EVENTS)

    # Compute cos‑observables for CSV
    cos_csv = df_csv.progress_apply(
        lambda r: pd.Series(compute_cos_observables_vector(r)), axis=1
    )
    df_csv[list(cos_csv.columns)] = cos_csv

    # Load and truncate PKL
    df_pkl = pd.read_pickle(PKL_FILE).head(MAX_EVENTS)
    # Map leptonic ← cos_theta_B_*, hadronic ← cos_theta_A_*
    df_pkl = df_pkl.rename(columns={
        "cos_theta_B_r": "leptonic_cos_r",
        "cos_theta_B_n": "leptonic_cos_n",
        "cos_theta_B_k": "leptonic_cos_k",
        "cos_theta_A_r": "hadronic_cos_r",
        "cos_theta_A_n": "hadronic_cos_n",
        "cos_theta_A_k": "hadronic_cos_k"
    })

    # Compare B observables
    for xlabel, col in B_OBSERVABLES:
        if col not in df_csv or col not in df_pkl:
            continue
        data_csv = df_csv[col].dropna().to_numpy()
        data_pkl = df_pkl[col].dropna().to_numpy()
        n1, e1 = compute_normalized_histogram(data_csv, BINS)
        n2, e2 = compute_normalized_histogram(data_pkl, BINS)
        path = os.path.join(OUTDIR, f"{col}_B_comparison.png")
        kappa = KAPPA_LEP if col.startswith("leptonic") else KAPPA_HAD
        overlay_plot_B(BINS, n1, e1, n2, e2, xlabel, path, kappa)

    # Compute and compare C observables
    for lep in LEP_KEYS:
        for had in HAD_KEYS:
            col = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
            df_csv[col] = df_csv[lep] * df_csv[had]
            df_pkl[col] = df_pkl[lep] * df_pkl[had]

    for xlabel, col in PRODUCT_OBSERVABLES:
        if col not in df_csv or col not in df_pkl:
            continue
        data_csv = df_csv[col].dropna().to_numpy()
        data_pkl = df_pkl[col].dropna().to_numpy()
        n1, e1 = compute_normalized_histogram(data_csv, BINS)
        n2, e2 = compute_normalized_histogram(data_pkl, BINS)
        path = os.path.join(OUTDIR, f"{col}_C_comparison.png")
        overlay_plot_C(BINS, n1, e1, n2, e2, xlabel, path)

if __name__ == "__main__":
    main()

