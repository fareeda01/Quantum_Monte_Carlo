# Import numercial python library 
import numpy as np

# Function for Potential Energy calculation
def potential_energy(electron_positions, nuclei_coords, charge):
    # initialise potential energy 
    V = 0.0 

    # Calculations for Electron_Nuclear attraction term
    # Loop over all electrons 
    for r in electron_positions:       
        for R, Z in zip(nuclei_coords, charge):       # charge (nuclear) 
            dist = np.linalg.norm(r - R)
            V -= Z / dist   # interaction is attractive, so we subtract here 
    
    # Calculations for Electon_Electron Repulsion Term
    # No. of electrons
    N = len(electron_positions) 
    for i in range (N):
        for j in range(i+1, N):
            elec_elec_dist = np.linalg.norm(electron_positions[i] -  electron_positions[j])
            V += 1.0 / elec_elec_dist # Repulsion energy contribution is positive, so added
    return V


# Function for Kinetic Energy Calculation
def kinetic_energy(electron_positions, nuclei_coords, exponent):
    T = 0.0   # Initialize Kinetic Energy
    
    for r in electron_positions:
        # initialise laplacian
        total_laplacian = 0.0
        # initialise psi
        total_psi = 0.0
    for R, zeta in zip(nuclei_coords, exponent):
        dist = np.linalg.norm(r - R)
        psi = np.exp( - zeta * dist)
        laplacian = (zeta**2 - 2*zeta / dist) * psi
        total_psi += psi
        total_laplacian += laplacian
        T += -0.5 * (total_laplacian / total_psi)
    return T


# Computing total energy as accumulation of kinetic and potential energy
def local_energy(electron_positions, nuclei_coords, charge, exponent):
    V = potential_energy(electron_positions, nuclei_coords, charge)
    T = kinetic_energy(electron_positions, nuclei_coords, exponent)

    return T + V




