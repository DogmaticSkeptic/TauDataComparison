#!/usr/bin/env python3
"""
convert_yulei_to_aaron.py

Reads:  ./data/yulei_code_data.pkl
Writes: ./data/aaron_code_data_converted.csv

Maps Yulei’s columns into Aaron’s schema, inserting NaNs where necessary,
and includes cos_theta_* variables at the end.
"""

import os
import pandas as pd
import numpy as np

# Paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
input_pkl = os.path.join(data_dir, 'yulei_code_data.pkl')
output_csv = os.path.join(data_dir, 'yulei_code_data.csv')

# Load source DataFrame
df_in = pd.read_pickle(input_pkl)

# Aaron’s base schema
aaron_cols = [
    'TauLeptonic_pt', 'TauLeptonic_eta', 'TauLeptonic_phi', 'TauLeptonic_mass',
    'TauHadronic_pt', 'TauHadronic_eta', 'TauHadronic_phi', 'TauHadronic_mass',
    'ChargedPion_pt', 'ChargedPion_eta', 'ChargedPion_phi', 'ChargedPion_mass',
    'NeutralPion_pt', 'NeutralPion_eta', 'NeutralPion_phi', 'NeutralPion_mass',
    'Muon_pt', 'Muon_eta', 'Muon_phi', 'Muon_mass',
    'nu_mu_fromLeptonicTau_pt', 'nu_mu_fromLeptonicTau_eta',
    'nu_mu_fromLeptonicTau_phi', 'nu_mu_fromLeptonicTau_mass',
    'nu_tau_fromLeptonicTau_pt', 'nu_tau_fromLeptonicTau_eta',
    'nu_tau_fromLeptonicTau_phi', 'nu_tau_fromLeptonicTau_mass',
    'nu_tau_fromHadronicTau_pt', 'nu_tau_fromHadronicTau_eta',
    'nu_tau_fromHadronicTau_phi', 'nu_tau_fromHadronicTau_mass',
    'Z_pt', 'Z_eta', 'Z_phi', 'Z_mass'
]

# Additional cos_theta columns to include
cos_cols = [
    'cos_theta_A_n', 'cos_theta_A_r', 'cos_theta_A_k',
    'cos_theta_B_n', 'cos_theta_B_r', 'cos_theta_B_k'
]

# Build full output schema
output_cols = aaron_cols + cos_cols
df_out = pd.DataFrame(index=df_in.index, columns=output_cols)

# 1. Leptonic tau ← tau_m
df_out['TauLeptonic_pt']   = df_in.get('tau_m_PT')
df_out['TauLeptonic_eta']  = df_in.get('tau_m_Eta')
df_out['TauLeptonic_phi']  = df_in.get('tau_m_Phi')
df_out['TauLeptonic_mass'] = df_in.get('tau_m_Mass')

# 2. Hadronic tau ← tau_p
df_out['TauHadronic_pt']   = df_in.get('tau_p_PT')
df_out['TauHadronic_eta']  = df_in.get('tau_p_Eta')
df_out['TauHadronic_phi']  = df_in.get('tau_p_Phi')
df_out['TauHadronic_mass'] = df_in.get('tau_p_Mass')

# 3. Pions not provided → NaN
pion_cols = [
    'ChargedPion_pt', 'ChargedPion_eta', 'ChargedPion_phi', 'ChargedPion_mass',
    'NeutralPion_pt', 'NeutralPion_eta', 'NeutralPion_phi', 'NeutralPion_mass'
]
df_out[pion_cols] = np.nan

# 4. Muon ← mu_m
df_out['Muon_pt']   = df_in.get('mu_m_PT')
df_out['Muon_eta']  = df_in.get('mu_m_Eta')
df_out['Muon_phi']  = df_in.get('mu_m_Phi')
df_out['Muon_mass'] = df_in.get('mu_m_Mass')

# 5. Neutrinos not provided → NaN
nu_cols = [
    'nu_mu_fromLeptonicTau_pt', 'nu_mu_fromLeptonicTau_eta',
    'nu_mu_fromLeptonicTau_phi', 'nu_mu_fromLeptonicTau_mass',
    'nu_tau_fromLeptonicTau_pt', 'nu_tau_fromLeptonicTau_eta',
    'nu_tau_fromLeptonicTau_phi', 'nu_tau_fromLeptonicTau_mass',
    'nu_tau_fromHadronicTau_pt', 'nu_tau_fromHadronicTau_eta',
    'nu_tau_fromHadronicTau_phi', 'nu_tau_fromHadronicTau_mass'
]
df_out[nu_cols] = np.nan

# 6. Z boson ← Z_PT, Z_Eta, Z_Phi, Z_Mass
df_out['Z_pt']   = df_in.get('Z_PT')
df_out['Z_eta']  = df_in.get('Z_Eta')
df_out['Z_phi']  = df_in.get('Z_Phi')
df_out['Z_mass'] = df_in.get('Z_Mass')

# 7. cos_theta values
for col in cos_cols:
    df_out[col] = df_in.get(col)

# 8. Write to CSV
df_out.to_csv(output_csv, index=False)
print(f'Conversion complete. Output file: {output_csv}')

