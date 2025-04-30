#!/usr/bin/env python3
"""
pkl_to_csv.py

Convert Yulei’s PKL into a CSV matching the Aaron‐CSV schema.
"""
import argparse
import pandas as pd
import numpy as np

# 1) Define the rename map from Yulei → Aaron
RENAME_MAP = {
    # leptonic τ
    "tau_m_PT":   "TauLeptonic_pt",
    "tau_m_Eta":  "TauLeptonic_eta",
    "tau_m_Phi":  "TauLeptonic_phi",
    "tau_m_Mass": "TauLeptonic_mass",
    # hadronic τ (ρ → π±+π0)
    "tau_p_PT":   "TauHadronic_pt",
    "tau_p_Eta":  "TauHadronic_eta",
    "tau_p_Phi":  "TauHadronic_phi",
    "tau_p_Mass": "TauHadronic_mass",
    # muon
    "mu_m_PT":    "Muon_pt",
    "mu_m_Eta":   "Muon_eta",
    "mu_m_Phi":   "Muon_phi",
    "mu_m_Mass":  "Muon_mass",
    # ρ meson
    "rho_p_PT":   "Rho_pt",
    "rho_p_Eta":  "Rho_eta",
    "rho_p_Phi":  "Rho_phi",
    "rho_p_Mass": "Rho_mass",
    # Z boson (uppercase → mixed‐case)
    "Z_PT":       "Z_pt",
    "Z_Eta":      "Z_eta",
    "Z_Phi":      "Z_phi",
    "Z_Mass":     "Z_mass",
}

# 2) The full Aaron‐CSV column list
AARON_COLUMNS = [
    "TauLeptonic_pt",
    "TauLeptonic_eta",
    "TauLeptonic_phi",
    "TauLeptonic_mass",
    "TauHadronic_pt",
    "TauHadronic_eta",
    "TauHadronic_phi",
    "TauHadronic_mass",
    "ChargedPion_pt",
    "ChargedPion_eta",
    "ChargedPion_phi",
    "ChargedPion_mass",
    "NeutralPion_pt",
    "NeutralPion_eta",
    "NeutralPion_phi",
    "NeutralPion_mass",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
    "nu_mu_fromLeptonicTau_pt",
    "nu_mu_fromLeptonicTau_eta",
    "nu_mu_fromLeptonicTau_phi",
    "nu_mu_fromLeptonicTau_mass",
    "nu_tau_fromLeptonicTau_pt",
    "nu_tau_fromLeptonicTau_eta",
    "nu_tau_fromLeptonicTau_phi",
    "nu_tau_fromLeptonicTau_mass",
    "nu_tau_fromHadronicTau_pt",
    "nu_tau_fromHadronicTau_eta",
    "nu_tau_fromHadronicTau_phi",
    "nu_tau_fromHadronicTau_mass",
    "Z_pt",
    "Z_eta",
    "Z_phi",
    "Z_mass",
    "cos_theta_A_n",
    "cos_theta_A_r",
    "cos_theta_A_k",
    "cos_theta_B_n",
    "cos_theta_B_r",
    "cos_theta_B_k",
]

def main():
    parser = argparse.ArgumentParser(
        description="Convert Yulei PKL to a CSV matching the Aaron schema"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input Yulei .pkl file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to write output CSV file"
    )
    args = parser.parse_args()

    # Load the pickle
    df = pd.read_pickle(args.input)

    # Rename known columns
    df = df.rename(columns=RENAME_MAP)

    # Insert any missing Aaron columns as NaN
    for col in AARON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder to exactly Aaron’s column order
    df = df[AARON_COLUMNS]

    # Write out
    df.to_csv(args.output, index=False)
    print(f"Wrote standardized CSV to: {args.output}")

if __name__ == "__main__":
    main()

