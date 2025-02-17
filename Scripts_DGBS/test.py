import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
import matplotlib.pyplot as plt



def qkd_wfhd(alpha_amp, phi, reflectivity, cutoff=2):
    qkd = sf.Program(2)
    with qkd.context as q:
        # prepare the input single photon and coherent state
        Fock(1) | q[0]
        # Dgate(r=alpha_amp, phi=0) | q[0]
        # Dgate(r=alpha_amp, phi=0) | q[1]
        Fock(1) | q[1]

        # linear interferometer
        Rgate(phi) | q[1]  
        BSgate(theta=np.arcsin(reflectivity), phi=np.pi/2) | (q[0], q[1]) 
         

    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim":cutoff})
    results = eng.run(qkd)
    state = results.state
    return state.all_fock_probs()

if __name__ == "__main__":
    alpha_amp=0.0001
    reflectivity = 0.5
    phis = np.linspace(0, 2*np.pi, 100)
    probs_mode_0 = []
    probs_mode_1 = []
    probs_mode_cc = []

    for phi in phis:
        probs = qkd_wfhd(alpha_amp, phi, reflectivity,cutoff=20)
        probs_mode_0.append(np.sum(probs[1,:]))
        probs_mode_1.append(np.sum(probs[:,1]))
        probs_mode_cc.append(np.sum(probs[1:,1:]))

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(phis, probs_mode_0, label='Mode 0')
    plt.xlabel('Phi')
    plt.ylabel('Probability')
    plt.title('Probability of Single Count in Mode 0')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(phis, probs_mode_1, label='Mode 1')
    plt.xlabel('Phi')
    plt.ylabel('Probability')
    plt.title('Probability of Single Count in Mode 1')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(phis, probs_mode_cc, label='Mode 1')
    plt.xlabel('Phi')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()