import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_summary_pdf(b_results, c_results, kappa0, kappa1, B_OBSERVABLES, PRODUCT_OBSERVABLES, outdir="plots"):
    """
    Generates a summary PDF containing spin observables and Bell violation information.
    """

    def assemble_B_vector(dataset, method):
        BA, BB = [], []
        for _, col in B_OBSERVABLES:
            val, unc = b_results[col][dataset][method]
            if col.startswith("leptonic"):
                BA.append(f"{val:.3f} ± {unc:.3f}")
            else:
                BB.append(f"{val:.3f} ± {unc:.3f}")
        return BA, BB

    def assemble_B_numeric(dataset, method):
        BA_vals, BB_vals = [], []
        for _, col in B_OBSERVABLES:
            val = b_results[col][dataset][method][0]
            if col.startswith("leptonic"):
                BA_vals.append(val)
            else:
                BB_vals.append(val)
        return np.array(BA_vals), np.array(BB_vals)

    from cos_utils import compute_density_matrix, compute_concurrence, compute_best_bell_violation, assemble_C_numeric_and_uncertainty, compute_full_density_matrix
    import numpy as np
    methods = ["asym", "avg", "fit"]
    datasets = ["Miles", "Yulei", "Yulei_pre"]
    summary_lines = []
    summary_lines.append("Spin Observables Summary\n")
    summary_lines.append("=" * 80 + "\n")
    for ds in datasets:
        # pretty‐name mapping
        label = "Precomputed Yulei" if ds=="Yulei_pre" else ds
        summary_lines.append(f"Dataset: {label}\n" + "-" * 80)
        for m in methods:
            BA_str, BB_str = assemble_B_vector(ds, m)
            BA_num, BB_num = assemble_B_numeric(ds, m)
            C_vals, C_unc = assemble_C_numeric_and_uncertainty(c_results, ds, m, PRODUCT_OBSERVABLES)
            C_vals        = -C_vals
            rho           = compute_density_matrix(BA_num, BB_num, C_vals)

            # --- compute and print R eigenvalues via full density‐matrix method
            dirs   = ['r','n','k']
            B_dict = {
                'Br': BA_num[0], 'Bn': BA_num[1], 'Bk': BA_num[2],
                'Ar': BB_num[0], 'An': BB_num[1], 'Ak': BB_num[2]
            }
            C_dict = {
                f"{d1}{d2}": C_vals[i,j]
                for i, d1 in enumerate(dirs)
                for j, d2 in enumerate(dirs)
            }
            eigs = compute_full_density_matrix(C_dict, B_dict)
            summary_lines.append(
                "  R eigenvalues: " +
                ", ".join(f"{λ:.3f}" for λ in eigs)
            )

            # compute central value
            conc = compute_concurrence(rho)
            # estimate uncertainty via Monte-Carlo on C uncertainties
            n_samples = 1000
            conc_samples = []
            for _ in range(n_samples):
                # perturb each C_{ij} by its 1σ (use absolute uncertainties to ensure non-negative scale)
                delta = np.random.normal(loc=0.0, scale=np.abs(C_unc))
                C_pert = C_vals + delta
                rho_pert = compute_density_matrix(BA_num, BB_num, C_pert)
                conc_samples.append(compute_concurrence(rho_pert))
            conc_unc = float(np.std(conc_samples))
            pair, sign, bell_val, bell_unc = compute_best_bell_violation(C_vals, C_unc)
            summary_lines.append(f"Method: {m}")
            summary_lines.append("  Leptonic (BA): " + ", ".join(BA_str))
            summary_lines.append("  Hadronic (BB): " + ", ".join(BB_str))
            summary_lines.append("  Spin Correlation (C) matrix:")
            for row in C_vals:
                summary_lines.append("    " + "  ".join(f"{x:.3f}" for x in row))
            summary_lines.append("  Density Matrix (ρ):")
            for row in rho:
                summary_lines.append("    " + "  ".join(f"{elem.real:+.3f}{elem.imag:+.3f}j" for elem in row))
            summary_lines.append(f"  Concurrence: {conc:.3f} ± {conc_unc:.3f}")
            summary_lines.append("  Bell Violation Measure:")
            if pair is not None:
                i0, j0 = pair
                basis = {0: 'r', 1: 'n', 2: 'k'}
                summary_lines.append(f"    Best pair: ({basis[i0]}, {basis[j0]}) sign '{sign}'")
                summary_lines.append(f"    Violation: {bell_val:.3f} ± {bell_unc:.3f}")
            else:
                summary_lines.append("    Not available")
            summary_lines.append("-" * 80 + "\n")

    summary_text = "\n".join(summary_lines)
    wrapped_text = "\n".join(textwrap.fill(line, width=90) for line in summary_text.split("\n"))
    summary_pdf = os.path.join(outdir, "Summary.pdf")
    pp = PdfPages(summary_pdf)
    fig = plt.figure(figsize=(11, 17))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.text(0.01, 0.99, wrapped_text, va="top", ha="left",
             fontfamily="monospace", fontsize=8, wrap=True)
    pp.savefig(fig)
    plt.close(fig)
    pp.close()
    print(f"Summary PDF saved to {summary_pdf}")
