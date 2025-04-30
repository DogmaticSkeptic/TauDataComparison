#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Open ROOT file and retrieve TTree.
    TFile* file = TFile::Open("/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/data/sig/lep_rho.root");
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening ROOT file." << std::endl;
        return 1;
    }
    TTree* tree = static_cast<TTree*>(file->Get("qe"));
    if (!tree) {
        std::cerr << "TTree 'qe' not found." << std::endl;
        return 1;
    }

    // Z boson.
    int z_valid;
    float z_pt, z_eta, z_phi, z_mass;
    tree->SetBranchAddress("parton_Z_Valid", &z_valid);
    tree->SetBranchAddress("parton_Z_PT",    &z_pt);
    tree->SetBranchAddress("parton_Z_Eta",   &z_eta);
    tree->SetBranchAddress("parton_Z_Phi",   &z_phi);
    tree->SetBranchAddress("parton_Z_Mass",  &z_mass);

    // Leptonic taus (μ and τ).
    int tau_p_valid, tau_m_valid;
    float tau_p_pt, tau_p_eta, tau_p_phi, tau_p_mass;
    float tau_m_pt, tau_m_eta, tau_m_phi, tau_m_mass;
    tree->SetBranchAddress("parton_tau_p_Valid", &tau_p_valid);
    tree->SetBranchAddress("parton_tau_p_PT",    &tau_p_pt);
    tree->SetBranchAddress("parton_tau_p_Eta",   &tau_p_eta);
    tree->SetBranchAddress("parton_tau_p_Phi",   &tau_p_phi);
    tree->SetBranchAddress("parton_tau_p_Mass",  &tau_p_mass);
    tree->SetBranchAddress("parton_tau_m_Valid", &tau_m_valid);
    tree->SetBranchAddress("parton_tau_m_PT",    &tau_m_pt);
    tree->SetBranchAddress("parton_tau_m_Eta",   &tau_m_eta);
    tree->SetBranchAddress("parton_tau_m_Phi",   &tau_m_phi);
    tree->SetBranchAddress("parton_tau_m_Mass",  &tau_m_mass);

    // Muons.
    int mu_p_valid, mu_m_valid;
    float mu_p_pt, mu_p_eta, mu_p_phi, mu_p_mass;
    float mu_m_pt, mu_m_eta, mu_m_phi, mu_m_mass;
    tree->SetBranchAddress("parton_mu_p_Valid", &mu_p_valid);
    tree->SetBranchAddress("parton_mu_p_PT",    &mu_p_pt);
    tree->SetBranchAddress("parton_mu_p_Eta",   &mu_p_eta);
    tree->SetBranchAddress("parton_mu_p_Phi",   &mu_p_phi);
    tree->SetBranchAddress("parton_mu_p_Mass",  &mu_p_mass);
    tree->SetBranchAddress("parton_mu_m_Valid", &mu_m_valid);
    tree->SetBranchAddress("parton_mu_m_PT",    &mu_m_pt);
    tree->SetBranchAddress("parton_mu_m_Eta",   &mu_m_eta);
    tree->SetBranchAddress("parton_mu_m_Phi",   &mu_m_phi);
    tree->SetBranchAddress("parton_mu_m_Mass",  &mu_m_mass);

    // Hadronic taus: positively charged rho, π±, π0.
    int rho_p_valid, rho_m_valid;
    float rho_p_pt, rho_p_eta, rho_p_phi, rho_p_mass;
    tree->SetBranchAddress("parton_rho_p_Valid", &rho_p_valid);
    tree->SetBranchAddress("parton_rho_p_PT",    &rho_p_pt);
    tree->SetBranchAddress("parton_rho_p_Eta",   &rho_p_eta);
    tree->SetBranchAddress("parton_rho_p_Phi",   &rho_p_phi);
    tree->SetBranchAddress("parton_rho_p_Mass",  &rho_p_mass);
    // (existing π± and π0 branches)
    int pi_p_valid, pi_m_valid;
    float pi_p_pt, pi_p_eta, pi_p_phi, pi_p_mass;
    float pi_m_pt, pi_m_eta, pi_m_phi, pi_m_mass;
    int pi0_p_valid, pi0_m_valid;
    float pi0_p_pt, pi0_p_eta, pi0_p_phi, pi0_p_mass;
    float pi0_m_pt, pi0_m_eta, pi0_m_phi, pi0_m_mass;
    tree->SetBranchAddress("parton_pi_p_1_Valid", &pi_p_valid);
    tree->SetBranchAddress("parton_pi_p_1_PT",    &pi_p_pt);
    tree->SetBranchAddress("parton_pi_p_1_Eta",   &pi_p_eta);
    tree->SetBranchAddress("parton_pi_p_1_Phi",   &pi_p_phi);
    tree->SetBranchAddress("parton_pi_p_1_Mass",  &pi_p_mass);
    tree->SetBranchAddress("parton_pi0_p_Valid",  &pi0_p_valid);
    tree->SetBranchAddress("parton_pi0_p_PT",     &pi0_p_pt);
    tree->SetBranchAddress("parton_pi0_p_Eta",    &pi0_p_eta);
    tree->SetBranchAddress("parton_pi0_p_Phi",    &pi0_p_phi);
    tree->SetBranchAddress("parton_pi0_p_Mass",   &pi0_p_mass);
    tree->SetBranchAddress("parton_rho_m_Valid",  &rho_m_valid);
    tree->SetBranchAddress("parton_pi_m_1_Valid", &pi_m_valid);
    tree->SetBranchAddress("parton_pi_m_1_PT",    &pi_m_pt);
    tree->SetBranchAddress("parton_pi_m_1_Eta",   &pi_m_eta);
    tree->SetBranchAddress("parton_pi_m_1_Phi",   &pi_m_phi);
    tree->SetBranchAddress("parton_pi_m_1_Mass",  &pi_m_mass);
    tree->SetBranchAddress("parton_pi0_m_Valid",  &pi0_m_valid);
    tree->SetBranchAddress("parton_pi0_m_PT",     &pi0_m_pt);
    tree->SetBranchAddress("parton_pi0_m_Eta",    &pi0_m_eta);
    tree->SetBranchAddress("parton_pi0_m_Phi",    &pi0_m_phi);
    tree->SetBranchAddress("parton_pi0_m_Mass",   &pi0_m_mass);

    // Neutrinos from τ and μ decays.
    int nu_mu_p_valid, nu_mu_m_valid;
    float nu_mu_p_pt, nu_mu_p_eta, nu_mu_p_phi, nu_mu_p_mass;
    float nu_mu_m_pt, nu_mu_m_eta, nu_mu_m_phi, nu_mu_m_mass;
    tree->SetBranchAddress("parton_nu_mu_p_Valid", &nu_mu_p_valid);
    tree->SetBranchAddress("parton_nu_mu_p_PT",    &nu_mu_p_pt);
    tree->SetBranchAddress("parton_nu_mu_p_Eta",   &nu_mu_p_eta);
    tree->SetBranchAddress("parton_nu_mu_p_Phi",   &nu_mu_p_phi);
    tree->SetBranchAddress("parton_nu_mu_p_Mass",  &nu_mu_p_mass);
    tree->SetBranchAddress("parton_nu_mu_m_Valid", &nu_mu_m_valid);
    tree->SetBranchAddress("parton_nu_mu_m_PT",    &nu_mu_m_pt);
    tree->SetBranchAddress("parton_nu_mu_m_Eta",   &nu_mu_m_eta);
    tree->SetBranchAddress("parton_nu_mu_m_Phi",   &nu_mu_m_phi);
    tree->SetBranchAddress("parton_nu_mu_m_Mass",  &nu_mu_m_mass);

    int nu_tau_p_valid, nu_tau_m_valid;
    float nu_tau_p_pt, nu_tau_p_eta, nu_tau_p_phi, nu_tau_p_mass;
    float nu_tau_m_pt, nu_tau_m_eta, nu_tau_m_phi, nu_tau_m_mass;
    tree->SetBranchAddress("parton_nu_tau_p_Valid", &nu_tau_p_valid);
    tree->SetBranchAddress("parton_nu_tau_p_PT",    &nu_tau_p_pt);
    tree->SetBranchAddress("parton_nu_tau_p_Eta",   &nu_tau_p_eta);
    tree->SetBranchAddress("parton_nu_tau_p_Phi",   &nu_tau_p_phi);
    tree->SetBranchAddress("parton_nu_tau_p_Mass",  &nu_tau_p_mass);
    tree->SetBranchAddress("parton_nu_tau_m_Valid", &nu_tau_m_valid);
    tree->SetBranchAddress("parton_nu_tau_m_PT",    &nu_tau_m_pt);
    tree->SetBranchAddress("parton_nu_tau_m_Eta",   &nu_tau_m_eta);
    tree->SetBranchAddress("parton_nu_tau_m_Phi",   &nu_tau_m_phi);
    tree->SetBranchAddress("parton_nu_tau_m_Mass",  &nu_tau_m_mass);

    // Prepare CSV output.
    std::ofstream out("lep_rho_z_valid.csv");
    if (!out) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    // Header now includes Rho kinematics.
    out << "TauLeptonic_pt,TauLeptonic_eta,TauLeptonic_phi,TauLeptonic_mass,"
           "TauHadronic_pt,TauHadronic_eta,TauHadronic_phi,TauHadronic_mass,"
           "Rho_pt,Rho_eta,Rho_phi,Rho_mass,"
           "ChargedPion_pt,ChargedPion_eta,ChargedPion_phi,ChargedPion_mass,"
           "NeutralPion_pt,NeutralPion_eta,NeutralPion_phi,NeutralPion_mass,"
           "Muon_pt,Muon_eta,Muon_phi,Muon_mass,"
           "nu_mu_fromLeptonicTau_pt,nu_mu_fromLeptonicTau_eta,nu_mu_fromLeptonicTau_phi,nu_mu_fromLeptonicTau_mass,"
           "nu_tau_fromLeptonicTau_pt,nu_tau_fromLeptonicTau_eta,nu_tau_fromLeptonicTau_phi,nu_tau_fromLeptonicTau_mass,"
           "nu_tau_fromHadronicTau_pt,nu_tau_fromHadronicTau_eta,nu_tau_fromHadronicTau_phi,nu_tau_fromHadronicTau_mass,"
           "Z_pt,Z_eta,Z_phi,Z_mass\n";

    // Loop and fill.
    Long64_t nentries = tree->GetEntries();
    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);
        if (tau_m_valid && mu_m_valid == 1 && tau_p_valid == 1 && rho_p_valid == 1 && z_valid) {
            // Build TLorentzVectors.
            TLorentzVector tau_lep, tau_had, rho, pion_ch, pion0, muon;
            tau_lep.SetPtEtaPhiM(tau_m_pt,   tau_m_eta,   tau_m_phi,   tau_m_mass);
            tau_had.SetPtEtaPhiM(tau_p_pt,   tau_p_eta,   tau_p_phi,   tau_p_mass);
            rho.SetPtEtaPhiM( rho_p_pt,     rho_p_eta,    rho_p_phi,    rho_p_mass);
            pion_ch.SetPtEtaPhiM(pi_p_pt,    pi_p_eta,     pi_p_phi,     pi_p_mass);
            pion0.SetPtEtaPhiM( pi0_p_pt,    pi0_p_eta,    pi0_p_phi,    pi0_p_mass);
            muon.SetPtEtaPhiM(   mu_m_pt,    mu_m_eta,     mu_m_phi,     mu_m_mass);

            TLorentzVector nu_mu_fromLep, nu_tau_fromLep, nu_tau_fromHad;
            nu_mu_fromLep.SetPtEtaPhiM(nu_mu_m_pt,   nu_mu_m_eta,   nu_mu_m_phi,   nu_mu_m_mass);
            nu_tau_fromLep.SetPtEtaPhiM(nu_tau_m_pt,  nu_tau_m_eta,  nu_tau_m_phi,  nu_tau_m_mass);
            nu_tau_fromHad.SetPtEtaPhiM(nu_tau_p_pt,  nu_tau_p_eta,  nu_tau_p_phi,  nu_tau_p_mass);

            // Write CSV line.
            out << tau_lep.Pt() << "," << tau_lep.Eta() << "," << tau_lep.Phi() << "," << tau_lep.M() << ","
                << tau_had.Pt() << "," << tau_had.Eta() << "," << tau_had.Phi() << "," << tau_had.M() << ","
                << rho.Pt()     << "," << rho.Eta()     << "," << rho.Phi()     << "," << rho.M()     << ","
                << pion_ch.Pt() << "," << pion_ch.Eta() << "," << pion_ch.Phi() << "," << pion_ch.M() << ","
                << pion0.Pt()   << "," << pion0.Eta()   << "," << pion0.Phi()   << "," << pion0.M()   << ","
                << muon.Pt()    << "," << muon.Eta()    << "," << muon.Phi()    << "," << muon.M()    << ","
                << nu_mu_fromLep.Pt() << "," << nu_mu_fromLep.Eta() << "," << nu_mu_fromLep.Phi() << "," << nu_mu_fromLep.M() << ","
                << nu_tau_fromLep.Pt() << "," << nu_tau_fromLep.Eta() << "," << nu_tau_fromLep.Phi() << "," << nu_tau_fromLep.M() << ","
                << nu_tau_fromHad.Pt() << "," << nu_tau_fromHad.Eta() << "," << nu_tau_fromHad.Phi() << "," << nu_tau_fromHad.M() << ","
                << z_pt << "," << z_eta << "," << z_phi << "," << z_mass << "\n";
        }
    }

    out.close();
    file->Close();
    delete file;

    return 0;
}

