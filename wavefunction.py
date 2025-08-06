import numpy as np 

# Define Wavefunction
def wavefunction(r, nuclei_coods, exponent):
    # initialise the wavefunction
    psi = 0.0
    for R, zeta in zip(nuclei_coods, exponent):
        # calculate distance r-R
        dist = np.linalg.norm(r - R)
        # Calculate value of psi
        psi += np.exp(-zeta * dist)
    return psi
