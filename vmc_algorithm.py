# import python library for numercial algebra
import numpy as np

from quantum_force import quantum_force
from wavefunction import wavefunction

# Define Metropolis Steps using 
def metropolis(r_old, nuclei_coords, exponent, delta_t):
    # Define random diffusion vector
    chi = np.random.normal(0.0, 1.0, size=3)
    # calculate force
    F_old = quantum_force(r_old, nuclei_coords, exponent)
    # calculate new positions based on old force
    r_new = r_old + delta_t * F_old + np.sqrt(delta_t) * chi
    # calculate new force based on new positions 
    F_new = quantum_force(r_new, nuclei_coords, exponent)

    # calculate transition probabilities ratio
    delta_F = F_old + F_new   
    chi_sq = np.dot(chi, chi)
    exponent_factor = - 0.5 * delta_t * np.dot(delta_F, delta_F) + np.sqrt(delta_t) * np.dot(chi, delta_F)
    T_ratio = np.exp(exponent_factor)

    # wavefunction evaluation at old positions
    psi_old = wavefunction(r_old, nuclei_coords, exponent)
    # wavefunction evaluation at new position 
    psi_new = wavefunction(r_new, nuclei_coords, exponent)
    psi_ratio = (psi_new / psi_old)**2

    # Define acceptance ratio
    A = T_ratio * psi_ratio
     
    # Accept the new move if is less than or equal to acceptance ratio
    if np.random.rand() <= A:
        return r_new, True
    # Otherwise return to old position
    else:
        return r_old, False

