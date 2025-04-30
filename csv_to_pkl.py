#!/usr/bin/env python3
"""
csv_to_pkl.py

Convert a CSV dataset into a pandas‐pickle with column names matching
an existing PKL schema, so that comparison scripts need no further renaming.
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV to PKL with standardized column names"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output pickle file"
    )
    args = parser.parse_args()

    # 1. Read CSV
    df = pd.read_csv(args.input)

    # 2. Rename columns to match the PKL schema
    rename_map = {
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
        # muon from leptonic τ
        "mu_m_PT":    "Muon_pt",
        "mu_m_Eta":   "Muon_eta",
        "mu_m_Phi":   "Muon_phi",
        "mu_m_Mass":  "Muon_mass",
        # ρ meson (if pre-built in CSV)
        "rho_p_PT":   "Rho_pt",
        "rho_p_Eta":  "Rho_eta",
        "rho_p_Phi":  "Rho_phi",
        "rho_p_Mass": "Rho_mass"
        # if your CSV already includes ChargedPion_*/NeutralPion_*, those need no renaming
    }
    df.rename(columns=rename_map, inplace=True)

    # 3. Serialize to pickle
    df.to_pickle(args.output)
    print(f"Wrote standardized PKL to: {args.output}")

if __name__ == "__main__":
    main()

