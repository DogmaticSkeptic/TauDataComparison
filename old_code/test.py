import matplotlib.pyplot as plt
import pickle

a = pickle.load(open("./data/yulei_code_data.pkl", "rb"))
b = a.query("rho_p_PT > 25 and mu_m_PT > 12 and m_tautau > 80 and m_tautau < 100 and theta_tau_cm > 0.6 and theta_tau_cm < 1.0")
plt.hist(b['cos_theta_A_k'], bins=100); plt.show()
plt.hist(b['cos_theta_B_k'], bins=100); plt.show()
