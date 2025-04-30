import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector

def unit_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ——— helicity basis and Core, verbatim —————————————————————————————

def helicity_basis(particle: vector.Vector) -> dict[str, vector.Vector]:
    """
    Helicity basis: https://arxiv.org/pdf/2305.07075
    Returns the helicity basis for a given particle
    """
    k_hat = particle.to_pxpypz().unit()
    # Define beam direction
    p_hat = vector.Vector(x=0, y=0, z=1)
    y_p = p_hat.dot(k_hat)
    r_p = np.sqrt(1 - y_p ** 2)
    r_hat = 1 / r_p * (p_hat - y_p * k_hat)
    r_hat = r_hat.unit()
    n_hat = r_hat.cross(k_hat)
    n_hat = n_hat.unit()
    return {"k": k_hat, "r": r_hat, "n": n_hat}


def _get_particle_in_frame(particle_dict, frame: str) -> vector.Vector4D:
    if frame not in particle_dict:
        raise ValueError(f"Frame '{frame}' not available.")
    if particle_dict[frame] is None:
        raise ValueError(f"Particle not yet transformed to frame '{frame}'.")
    return particle_dict[frame]


class Core:
    def __init__(
        self,
        main_particle_1: vector.Vector4D, main_particle_2: vector.Vector4D,
        child1: vector.Vector4D, child2: vector.Vector4D,
    ):
        self._m1 = {'lab frame': main_particle_1}
        self._m2 = {'lab frame': main_particle_2}
        self._c1 = {'lab frame': child1}
        self._c2 = {'lab frame': child2}
        self._all_particles = {'m1': self._m1, 'm2': self._m2,
                               'c1': self._c1, 'c2': self._c2}

    def m1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m1, frame)
    def m2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m2, frame)
    def c1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c1, frame)
    def c2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c2, frame)

    def _transform_to_frame(self, boost: vector.Vector3D,
                            source_frame: str, target_frame: str) -> None:
        for pdict in self._all_particles.values():
            pdict[target_frame] = pdict[source_frame].boost(boost)

    def analyze(self) -> dict[str, np.ndarray]:
        # Center-of-mass frame
        v1 = self.m1('lab frame')
        v2 = self.m2('lab frame')
        # manually sum components into a new Vector4D
        m0 = vector.Vector4D(
            px = v1.x + v2.x,
            py = v1.y + v2.y,
            pz = v1.z + v2.z,
            E  = v1.t + v2.t
        )
        self._transform_to_frame(-m0.to_beta3(), 'lab frame', 'cm frame')
        # Rest frames of each tau
        self._transform_to_frame(-self.m1('cm frame').to_beta3(),
                                 'cm frame', 'm1 cm frame')
        self._transform_to_frame(-self.m2('cm frame').to_beta3(),
                                 'cm frame', 'm2 cm frame')
        helicity_m1 = helicity_basis(self.m1('cm frame'))

        return {
            'cos_theta_A_n': self.c1('m1 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['n']),
            'cos_theta_A_r': self.c1('m1 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['r']),
            'cos_theta_A_k': self.c1('m1 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['k']),
            'cos_theta_B_n': self.c2('m2 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['n']),
            'cos_theta_B_r': self.c2('m2 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['r']),
            'cos_theta_B_k': self.c2('m2 cm frame')
                                    .to_pxpypz().unit()
                                    .dot(helicity_m1['k']),
        }

# ——— helper: build Vector4D from (pt, eta, phi, m) ——————————————————

def vec4(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E  = np.sqrt(px*px + py*py + pz*pz + m*m)
    return vector.Vector4D(px=px, py=py, pz=pz, E=E)

# ——— load data ————————————————————————————————————————————————

df = pd.read_pickle("./data/yulei_code_data.pkl")

# allocate storage for computed cosines
computed = {key: [] for key in [
    'cos_theta_A_n','cos_theta_A_r','cos_theta_A_k',
    'cos_theta_B_n','cos_theta_B_r','cos_theta_B_k'
]}

# loop over events
for _, row in df.iterrows():
    # m1 = tau_m, c1 = mu_m ;  m2 = tau_p, c2 = rho_p
    m1 = vec4(row.tau_m_PT, row.tau_m_Eta, row.tau_m_Phi, row.tau_m_Mass)
    c1 = vec4(row.mu_m_PT,  row.mu_m_Eta,  row.mu_m_Phi,  row.mu_m_Mass)
    m2 = vec4(row.tau_p_PT, row.tau_p_Eta, row.tau_p_Phi, row.tau_p_Mass)
    c2 = vec4(row.rho_p_PT, row.rho_p_Eta, row.rho_p_Phi, row.rho_p_Mass)

    core = Core(main_particle_1=m1, main_particle_2=m2,
                child1=c1, child2=c2)
    out  = core.analyze()
    for key in computed:
        computed[key].append(float(out[key]))

# convert to arrays
for key in computed:
    computed[key] = np.array(computed[key])

# loop over events
for _, row in df.iterrows():
    # build 4-vectors via vector.obj
    tau_lep = vector.obj(
        pt=row["TauLeptonic_pt"], eta=row["TauLeptonic_eta"],
        phi=row["TauLeptonic_phi"], mass=row["TauLeptonic_mass"]
    )
    tau_had = vector.obj(
        pt=row["TauHadronic_pt"], eta=row["TauHadronic_eta"],
        phi=row["TauHadronic_phi"], mass=row["TauHadronic_mass"]
    )
    lepton = vector.obj(
        pt=row["Muon_pt"], eta=row["Muon_eta"],
        phi=row["Muon_phi"], mass=row["Muon_mass"]
    )
    pion_ch = vector.obj(
        pt=row["ChargedPion_pt"], eta=row["ChargedPion_eta"],
        phi=row["ChargedPion_phi"], mass=row["ChargedPion_mass"]
    )
    pion0 = vector.obj(
        pt=row["NeutralPion_pt"], eta=row["NeutralPion_eta"],
        phi=row["NeutralPion_phi"], mass=row["NeutralPion_mass"]
    )
    rho   = pion_ch + pion0
    ditau = tau_lep + tau_had

    # Leptonic tau angles
    tau_lep_cm = tau_lep.boostCM_of(ditau)
    k_hat       = unit_vector(np.array([tau_lep_cm.x, tau_lep_cm.y, tau_lep_cm.z]))
    z_axis      = np.array([0, 0, 1])
    r_hat       = unit_vector(z_axis - np.dot(z_axis, k_hat) * k_hat)
    n_hat       = np.cross(k_hat, r_hat)
    lepton_rest = lepton.boostCM_of(tau_lep)
    lepton_mom  = unit_vector(np.array([lepton_rest.x, lepton_rest.y, lepton_rest.z]))
    leptonic_cos_r = np.dot(lepton_mom, r_hat)
    leptonic_cos_n = np.dot(lepton_mom, n_hat)
    leptonic_cos_k = np.dot(lepton_mom, k_hat)

    # Hadronic tau angles
    tau_had_cm2 = tau_had.boostCM_of(ditau)
    k_hat2      = unit_vector(np.array([tau_had_cm2.x, tau_had_cm2.y, tau_had_cm2.z]))
    r_hat2      = unit_vector(z_axis - np.dot(z_axis, k_hat2) * k_hat2)
    n_hat2      = np.cross(k_hat2, r_hat2)
    rho_rest    = rho.boostCM_of(tau_had)
    rho_mom     = unit_vector(np.array([rho_rest.x, rho_rest.y, rho_rest.z]))
    hadronic_cos_r = np.dot(rho_mom, r_hat2)
    hadronic_cos_n = np.dot(rho_mom, n_hat2)
    hadronic_cos_k = np.dot(rho_mom, k_hat2)

    # append results
    computed["leptonic_cos_r"].append(leptonic_cos_r)
    computed["leptonic_cos_n"].append(leptonic_cos_n)
    computed["leptonic_cos_k"].append(leptonic_cos_k)
    computed["hadronic_cos_r"].append(hadronic_cos_r)
    computed["hadronic_cos_n"].append(hadronic_cos_n)
    computed["hadronic_cos_k"].append(hadronic_cos_k)

# ——— compute residuals: calc − df ——————————————————————————————

residuals = {}
for key in computed:
    residuals[key] = computed[key] - df[key].values

# ——— residual histograms —————————————————————————————————————

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
axes = axes.flatten()

for ax, key in zip(axes, [
    'cos_theta_A_n','cos_theta_A_r','cos_theta_A_k',
    'cos_theta_B_n','cos_theta_B_r','cos_theta_B_k'
]):
    ax.hist(residuals[key], bins=50, histtype='stepfilled', alpha=0.7)
    ax.set_title(f"Residuals of {key}", fontsize=10)
    ax.set_xlabel(r"$\Delta\cos\theta$")
    ax.set_ylabel("Entries")
    ax.grid(True)

plt.show()

