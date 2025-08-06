# import python library for linear algebra
import numpy as np

# Function for Quantum Force 
def quantum_force(r, nuclei_coords, exponent):
    """
Compute the quantum force (drift vector) at position r due to nuclei.

Parameters:
    r : np.ndarray
        Electron position (shape: (3,))
    nuclei_coords : List[np.ndarray]
        List of nuclear coordinates
    exponent : List[float]
        List of orbital exponents for each nucleus

Returns:
    Fq : np.ndarray
        Quantum force vector (shape: (3,))
"""

    # initialize quantum force 
    Fq = np.zeros(3)
    # initialize total wavefunction 
    total_psi = 0.0
    # initialize gradiant
    total_grad = np.zeros(3)

    for R, zeta in zip(nuclei_coords, exponent):
        diff = r - R
        dist = np.linalg.norm(diff)
        # avoid dividion by zero 
        if dist < 1e-8:
            continue
        psi = np.exp(-zeta * dist)
        grad = - zeta * diff / dist * psi

        total_grad += grad
        total_psi += psi 

    Fq = 2 * total_grad / total_psi

    return Fq




