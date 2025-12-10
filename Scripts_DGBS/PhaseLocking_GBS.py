from strawberryfields import Program
from strawberryfields.ops import Sgate, Interferometer, Dgate, LossChannel
import strawberryfields as sf
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
 
def probability_array_DGBS(rescale_fac, gamma_val, Adj, loss, phase_array, cutoff=5, fock_prob=True, indices=None):
    """Return the tensor of the probabilities for DGBS with strawberryfields or a specific value if indices are provided.
    
    Parameters:
    c (float): The constant c.
    gamma_val (float): The loop strength.
    Adj (np.ndarray): The adjacency matrix.
    loss (float): The loss parameter.
    phase_array (list): The phase array for the squeezed states.
    cutoff (int): The cutoff for the dimension of Fock space for each mode.
    fock_prob (bool): Whether to return Fock probabilities or the state.
    indices (list): The indices of the specific value to return from the tensor.
    
    Returns:
    np.ndarray or float: The tensor of probabilities or a specific value if indices are provided.
    """
    m = Adj.shape[0]
    rl, _ = sf.decompositions.takagi(Adj)
    rmax= np.max(rl)
    rescaled_Adj= Adj / rmax*rescale_fac
    Id = np.eye(Adj.shape[0])
    gamma = gamma_val * np.ones(2 * Adj.shape[0])
    Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]])
    Sigma_Q_tot = inv(Sigma_Qinv)
    gamma = gamma_val * np.ones(2 * Adj.shape[0])
    d_alpha = (Sigma_Q_tot @ gamma)
    alpha = d_alpha[:m]
    
    rl, U = sf.decompositions.takagi(rescaled_Adj)
    # create the m mode Strawberry Fields program
    gbs = sf.Program(m)
    
    r = np.arctanh(rl)
    
    with gbs.context as q:
        # prepare the input squeezed states
        for n in range(m):
            Sgate(-r[n], phi=phase_array[n]) | q[n]
        
        # linear interferometer
        Interferometer(U) | q
        
        # Displacement operation    
        for n in range(m):
            if alpha[n] < 0:
                phase = np.pi
            else:
                phase = 0
            Dgate(np.abs(alpha[n]), phase) | q[n]
        for n in range(m):
            LossChannel(1.0 - loss) | q[n]
    
    eng = sf.Engine(backend="gaussian")
    results = eng.run(gbs)
    state = results.state
    if fock_prob:
        probabilities = state.all_fock_probs(cutoff=cutoff)
        if indices is not None:
            # Return the specific value at the given indices
            return probabilities[tuple(indices)]
        return probabilities
    else:
        return state

def phase_noise(var, nmodes, noise_bandwidth, t_end, dt, freq_range=None):
    """Return the phase noise for a given number of modes and time parameters, with spectral control.
    
    Parameters:
    var (float): The variance of the phase noise.
    nmodes (int): The number of modes.
    noise_bandwidth (float): The bandwidth of the noise.
    t_end (float): The end time for the simulation.
    dt (float): The time step for the simulation.
    freq_range (tuple): The frequency range (low, high) for the grey noise filter.
    
    Returns:
    np.ndarray: The phase noise array.
    """
    t = np.arange(0, t_end, dt)
    phase_noise = np.zeros((nmodes, len(t)))
    for i in range(nmodes):
        # Generate white noise
        noise = np.random.normal(0, var, len(t))
        
        # Transform to frequency domain
        noise_fft = np.fft.fft(noise)
        freqs = np.fft.fftfreq(len(t), dt)
        
        # Apply band-pass filter for grey noise
        if freq_range:
            low, high = freq_range
            filter_mask = (freqs >= low) & (freqs <= high)
            filter_mask |= (freqs <= -low) & (freqs >= -high)
            noise_fft[~filter_mask] = 0
        
        # Transform back to time domain
        filtered_noise = np.fft.ifft(noise_fft).real
        
        # Normalize and scale
        filtered_noise = np.cumsum(filtered_noise) * dt
        filtered_noise -= np.mean(filtered_noise)
        filtered_noise /= np.std(filtered_noise)
        filtered_noise *= noise_bandwidth
        
        phase_noise[i] = filtered_noise
    return phase_noise

def phase_lock(nmodes,t_end,dt,freq_array=None,amplitude_array=None):
    """Return the phase locking array for a given number of modes and time parameters.
    
    Parameters:
    nmodes (int): The number of modes.
    t_end (float): The end time for the simulation.
    dt (float): The time step for the simulation.
    freq_array (np.ndarray): The frequency array for the modes.
    amplitude_array (np.ndarray): The amplitude array for the modes.
    
    Returns:
    np.ndarray: The phase locking array.
    """
    t = np.arange(0, t_end, dt)
    phase_locking = np.zeros((nmodes, len(t)))
    
    if freq_array is None:
        freq_array = np.random.uniform(0.1, 20, nmodes)
    
    if amplitude_array is None:
        amplitude_array = np.random.uniform(0.1, 1, nmodes)
    
    for i in range(nmodes):
        phase_locking[i] = amplitude_array[i] * np.sin(2*np.pi*freq_array[i] * t)
    
    return phase_locking
def lock_amp(coincidence_prob,cutoff_freq,signal_ref):
    """Return the amplitude of the phase locking signal.
    
    Parameters:
    coincidence_prob (float): The probability of coincidence.
    filter_bandwidth (float): The bandwidth of the filter.
    signal_ref (float): The reference signal.
    
    Returns:
    float: The amplitude of the phase locking signal.
    """
    fft_result=np.fft.fft(coincidence_prob*signal_ref)
    frequencies = np.fft.fftfreq(len(coincidence_prob),d=dt)
    fft_result[np.abs(frequencies) > cutoff_freq] = 0
    filtered_signal = np.fft.ifft(fft_result).real
    return filtered_signal
if __name__ == "__main__":
    # Example adjacency matrix
    Adj = np.array([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0]
    ])
    # Adj=np.ones((6,6))
    # Parameters
    t_end = 10  # Total simulation time
    dt = 0.01   # Time step
    nmodes = Adj.shape[0]
    freq_range = (0.1, 20)
    phase_noise_array = phase_noise(var=np.pi/10, nmodes=nmodes, noise_bandwidth=np.pi/10, t_end=t_end, dt=dt, freq_range=freq_range)
    freq_array = np.array([10, 12, 14, 16, 18, 20])
    # freq_array = np.array([10,0,0,0,0,0])
    phase_locking_array = phase_lock(nmodes=nmodes, t_end=t_end, dt=dt,freq_array=freq_array, amplitude_array=np.pi/2*np.ones(nmodes))
    total_phase_array = phase_locking_array + phase_noise_array
    total_phase_array=phase_locking_array
    # Time array
    time = np.arange(0, t_end, dt)
    
    # Compute probabilities for one coincidence
    coincidence_probabilities = []
    for t_idx in range(len(time)):
        # Update phase array with the current phase model
        phase_array = total_phase_array[:, t_idx]
        # Compute the probability tensor and extract the probability of one coincidence
        indices = [1,0,0,0,0,1]
        prob = probability_array_DGBS(0.8, 0, Adj, 0, phase_array, cutoff=2, fock_prob=True, indices=indices)
        coincidence_probabilities.append(prob)
    
   
# Compute Fourier Transform of the coincidence probabilities
    coincidence_probabilities = np.array(coincidence_probabilities)
    fft_result = np.fft.fft(coincidence_probabilities)
    frequencies = np.fft.fftfreq(len(time), d=dt)

    # Compute the Nyquist frequency
    nyquist_freq = 1 / (2 * dt)
    
    # Only keep the positive frequencies below the Nyquist frequency
    positive_freqs = frequencies[(frequencies >= 0) & (frequencies <= nyquist_freq)]
    positive_fft = np.abs(fft_result[(frequencies >= 0) & (frequencies <= nyquist_freq)])
    phase_fft = np.angle(fft_result[(frequencies >= 0) & (frequencies <= nyquist_freq)])
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Probability of one coincidence
    axs[0].plot(time, coincidence_probabilities, label="One Coincidence Probability", color="blue")
    axs[0].set_title("Probability of One Coincidence as a Function of Time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Probability")
    axs[0].legend()
    axs[0].grid()
    
    # Plot 2: Fourier Transform of coincidence probability
    axs[1].plot(positive_freqs, positive_fft, label="Fourier Transform", color="red")
    axs[1].set_title("Fourier Transform of Coincidence Probability")
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(positive_freqs, phase_fft, label="Phase of Fourier Transform", color="green")
    axs[2].set_title("Fourier Transform of Coincidence Probability")
    axs[2].set_xlabel("Frequency")
    axs[2].set_ylabel("Phase")
    axs[2].legend()
    axs[2].grid()


    # Plot 2: Phase noise
    # for i in range(nmodes):
    #     axs[1].plot(time, phase_noise_array[i], label=f"Mode {i+1}")
    # axs[1].set_title("Phase Noise with Controlled Spectral Bandwidth")
    # axs[1].set_xlabel("Time")
    # axs[1].set_ylabel("Phase Noise (in multiples of π)")

    # for i in range(nmodes):
    #     axs[2].plot(time, total_phase_array[i], label=f"Mode {i+1}")
    # axs[2].set_title("Phase Noise with Controlled Spectral Bandwidth")
    # axs[2].set_xlabel("Time")
    # axs[2].set_ylabel("Phase Noise (in multiples of π)")
    
    # Format y-axis ticks as a factor of π
    def pi_formatter(x, pos):
        return f'{x/np.pi:.2f}π'
    
    axs[1].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axs[1].legend()
    axs[1].grid()

    
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    

