import numpy as np
import matplotlib.pyplot as plt

# Konstanta
k = 8.617e-5  # eV/K
q = 1.602e-19  # C
I_0 = 1e-12  # A
E_sun = 2.5  # eV (energi matahari rata-rata)

# Model efisiensi non-linier vs suhu (naik-turun)
def efficiency_vs_temp(T, E_g=1.1, eta0=0.32, beta=0.004, gamma=0.1, delta=0.001):
    T0 = 300
    linear_drop = 1 - beta * (T - T0)
    hump = gamma * np.exp(-delta * (T - T0)**2)  # Hump di sekitar T0
    eta_classic = eta0 * linear_drop * (1 + hump)
    eta_quantum = eta_classic * 1.15  # Quantum boost
    return np.maximum(eta_classic, 0), np.maximum(eta_quantum, 0)  # Hindari negatif

# Model efisiensi vs band gap (SQ-like: parabolic peak)
def efficiency_vs_bandgap(E_g, T=300):
    # SQ approx: eta max ~33% di E_g~1.3 eV, turun di sisi
    eta_classic = 0.33 * (1 - ((E_g - 1.3)/0.8)**2) * (1 - 0.004*(T-300))  # Parabolic
    eta_classic = np.maximum(eta_classic, 0)
    eta_quantum = eta_classic * 1.2
    return eta_classic, eta_quantum

# Model I-V (dari sebelumnya, tapi dengan T variabel)
def iv_classic(V, T, I_L):
    return I_L - I_0 * (np.exp(q * V / (1 * k * T)) - 1)  # Koreksi: k*T in eV

def photocurrent(E_photon_ev=2.5, phi_ev=2.0, intensity=1.0, quantum_factor=1.0):
    # Gunakan num_free_electrons dari kode sebelumnya (asumsikan tersedia)
    num_e = max(0, intensity * (E_photon_ev - phi_ev) / phi_ev)  # Sederhana jika belum
    return q * num_e * 1e19 * quantum_factor

I_L_base = photocurrent()  # Dasar

# Parameter
T_range = np.linspace(200, 400, 100)  # Suhu luas untuk hump
E_g_range = np.linspace(0.5, 2.5, 100)
V = np.linspace(0, 0.7, 100)

# Plot 1: Efisiensi vs Suhu (naik-turun)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
eta_c_range = [efficiency_vs_temp(t, E_g=1.1)[0] for t in T_range]
eta_q_range = [efficiency_vs_temp(t, E_g=1.1)[1] for t in T_range]
plt.plot(T_range, np.array(eta_c_range) * 100, label='Klasik', color='blue')
plt.plot(T_range, np.array(eta_q_range) * 100, label='Kuantum', color='red')
plt.xlabel('Suhu T (K)')
plt.ylabel('Efisiensi η (%)')
plt.title('Efisiensi vs Suhu: Kurva Naik-Turun (Efek Termal)')
plt.legend()
plt.grid(True)

# Plot 2: Efisiensi vs Band Gap (parabolic peak)
plt.subplot(1, 2, 2)
eta_c_bg = [efficiency_vs_bandgap(eg)[0] for eg in E_g_range]
eta_q_bg = [efficiency_vs_bandgap(eg)[1] for eg in E_g_range]
plt.plot(E_g_range, np.array(eta_c_bg) * 100, label='Klasik', color='blue')
plt.plot(E_g_range, np.array(eta_q_bg) * 100, label='Kuantum', color='red')
plt.xlabel('Band Gap E_g (eV)')
plt.ylabel('Efisiensi η (%)')
plt.title('Efisiensi vs Band Gap: Puncak Optimal ~1.3 eV')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot I-V multi-suhu (variasi kurva)
plt.figure(figsize=(10, 6))
T_vary = [250, 300, 350]  # Suhu berbeda
colors = ['green', 'blue', 'orange']
for i, t in enumerate(T_vary):
    I_L_t = I_L_base * (1 - 0.002 * (t - 300))  # I_L turun dengan T
    I_classic_t = iv_classic(V, t, I_L_t)
    plt.plot(V, I_classic_t * 1e3, label=f'Klasik T={t}K', color=colors[i], linestyle='-', alpha=0.8)
    I_quantum_t = iv_classic(V, t, I_L_t * 1.2)  # Quantum: I_L lebih tinggi
    plt.plot(V, I_quantum_t * 1e3, label=f'Kuantum T={t}K', color=colors[i], linestyle='--')
plt.xlabel('Tegangan V (V)')
plt.ylabel('Arus I (mA)')
plt.title('Kurva I-V Variasi Suhu: Shift Non-Linier')
plt.legend()
plt.grid(True)
plt.show()

# Output
print(f"Efisiensi klasik max (T~300K): {max(eta_c_range)*100:.1f}%")
print(f"Puncak efisiensi band gap: E_g={E_g_range[np.argmax(eta_c_bg)]:.2f} eV, η={max(eta_c_bg)*100:.1f}%")
