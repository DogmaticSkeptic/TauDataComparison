import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vector

# Name of the Yulei CSV file.
YULEI_FILE = "yulei_events.csv"
# Directory to save the plots.
OUTPUT_DIR = "plots"

def compute_Z_from_taus_yulei_vector(df):
    """
    Compute Z boson observables for Yulei data using the vector package.
    Assumes the following columns exist in df:
      TauLeptonic_pt, TauLeptonic_eta, TauLeptonic_phi, TauLeptonic_mass,
      TauHadronic_pt, TauHadronic_eta, TauHadronic_phi, TauHadronic_mass.
    Returns a dictionary with keys: "Z_pt", "Z_eta", "Z_phi", "Z_mass".
    """
    tau_lep = vector.array({
        "pt": df["TauLeptonic_pt"].values,
        "eta": df["TauLeptonic_eta"].values,
        "phi": df["TauLeptonic_phi"].values,
        "mass": df["TauLeptonic_mass"].values
    })
    tau_had = vector.array({
        "pt": df["TauHadronic_pt"].values,
        "eta": df["TauHadronic_eta"].values,
        "phi": df["TauHadronic_phi"].values,
        "mass": df["TauHadronic_mass"].values
    })

    Z = tau_lep + tau_had
    return {"Z_pt": Z.pt, "Z_eta": Z.eta, "Z_phi": Z.phi, "Z_mass": Z.mass}

def plot_residual_histogram(residuals, xlabel, title, output_path, bins=50):
    """
    Plot a histogram of residuals.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=bins, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved residual plot: {output_path}")

def main():
    # Load the Yulei CSV file.
    try:
        df = pd.read_csv(YULEI_FILE)
    except Exception as e:
        print(f"Error reading {YULEI_FILE}: {e}")
        return

    # Compute the Z observables from tau four-vectors using vector.
    computed_z = compute_Z_from_taus_yulei_vector(df)

    # List of Z observables to compare.
    z_obs = ["Z_pt", "Z_eta", "Z_phi", "Z_mass"]

    # Ensure output directory exists.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # For each Z observable, compute residual = computed - CSV and plot histogram.
    for obs in z_obs:
        if obs not in df.columns:
            print(f"Column {obs} not found in {YULEI_FILE}; skipping.")
            continue

        # CSV values (drop NaNs) and computed values.
        csv_values = df[obs].dropna().to_numpy()
        comp_values = computed_z[obs]
        
        # Ensure the computed array is of the same length as csv_values.
        # For simplicity, we assume the order of events in the CSV and the computed values match.
        # If there is a mismatch, further alignment of events is needed.
        if csv_values.shape[0] != comp_values.shape[0]:
            print(f"Warning: Mismatch in number of entries for {obs}. Using minimum length for residuals.")
        n = min(csv_values.shape[0], comp_values.shape[0])
        csv_values = csv_values[:n]
        comp_values = comp_values[:n]

        residuals = comp_values - csv_values

        # Define appropriate x-axis label and title for the residual histogram.
        xlabel = f"Residual {obs} [Computed - CSV]"
        title = f"Residuals for {obs}"

        output_path = os.path.join(OUTPUT_DIR, f"{obs}_residual.png")
        plot_residual_histogram(residuals, xlabel, title, output_path)

if __name__ == "__main__":
    main()

