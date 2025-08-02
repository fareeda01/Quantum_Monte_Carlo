import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python vmc_simulation.py <input_file>") # Accept Input File
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file) as f:
    n_nuclei = int(f.readline().strip())
    charges = []
    nuclei_coords = []
    for _ in range(n_nuclei):
        line = f.readline().split()
        charge = float(line[0])
        coord  = np.array([float(x) for x in line[1:4]])
        charges.append(charge)
        nuclei_coords.append(coord)
    n_electrons = int(f.readline().strip())
    a = float(f.readline().strip())
    dt = float(f.readline().strip())
    nsteps = int(f.readline().strip())
    nruns = int(f.readline().strip())

def E_local(positions, a, charges, nuclei_coords):
    """Calculate the local energy for electron configuration."""
    kinetic_energy = 0.0
    potential_energy = 0.0

    for i in range(len(positions)):  # Loop over all electrons
        r_i = positions[i]
        r_norm = np.linalg.norm(r_i)

        # Kinetic Energy calculations
        kinetic_energy += -0.5 * (a**2 - 2 * a / r_norm)  

        # Potential Energy (Coulombic Attraction)
        for j in range(len(nuclei_coords)):
            r_n = nuclei_coords[j]
            Z = charges[j]
            r_e_n = np.linalg.norm(r_i - r_n)
            potential_energy += -Z / r_e_n 

        # Electron-electron repulsion (Coulombic Repulsion)
        for j in range(i+1, len(positions)):
            r_j = positions[j]
            r_e_e = np.linalg.norm(r_i - r_j)
            if r_e_e > 1e-10:  # Avoid division by zero
                potential_energy += 1 / r_e_e 

    return kinetic_energy + potential_energy  # Return total local energy


# Initialize electron positions near their nuclei
positions = np.zeros((n_electrons, 3))
for i in range(n_electrons):
    nucleus_index = i % n_nuclei  # Assign electron to a nucleus
    positions[i] = nuclei_coords[nucleus_index] + np.random.normal(scale=0.5, size=3)

def proposal_step(dt):
    """ Generate a random move for the electron using normal distribution """
    return np.random.normal(loc=0, scale=np.sqrt(dt), size=3)

def wavefunction_molecule(a, r, nuclei_coords):
    psi = 0.0
    for R in nuclei_coords:
        psi += np.exp(-a * np.linalg.norm(r - R))
    return psi

def drift(a, r, nuclei_coords):
    total = np.zeros(3)
    psi = 0.0
    for R in nuclei_coords:
        r_R = r - R
        norm = np.linalg.norm(r_R)
        if norm > 1e-8:
            contrib = np.exp(-a * norm)
            psi += contrib
            total += -a * contrib * (r_R / norm)
    return total / max(psi, 1e-10)

energy_values = [] 
energy_sum = 0.0  # Initialize energy accumulator
accept_count = 0  # Initialize acceptance counter

for step in range(nsteps):
    for i in range(n_electrons):
        r_i_old = positions[i]  
        d_old = drift(a, r_i_old, nuclei_coords)  # Compute drift at old position 

        # Find new position using drift-diffusion
        r_i_new = r_i_old + dt * d_old + np.random.normal(loc=0, scale=np.sqrt(dt), size=3)

        d_new = drift(a, r_i_new, nuclei_coords)  # Drift at new position

        # Compute only the changed wavefunction factor for electron i
        psi_old = max(wavefunction_molecule(a, r_i_old, nuclei_coords), 1e-10)
        psi_new = max(wavefunction_molecule(a, r_i_new, nuclei_coords), 1e-10)

        # Compute acceptance probability__q
        diff = r_i_new - r_i_old
        prod = np.dot(d_new + d_old, diff)
        argexpo = 0.5 * (np.linalg.norm(d_new)**2 - np.linalg.norm(d_old)**2) * dt + prod
        q = np.exp(-argexpo) * (psi_new / psi_old)**2
        q = min(q, 1) 

        # Generate a uniform random number
        u = np.random.rand()

        # Accept or reject the move
        if u <= q:
            positions[i] = r_i_new  # Accept: update position
            accept_count += 1       # Increment acceptance counter
        else:
            positions[i] = r_i_old  #Keep old position if rejected

    # Accumulate local energy
    energy = E_local(positions, a, charges, nuclei_coords)
    energy_values.append(energy)  
    energy_sum += energy 

# Compute final statistics
energy_avg = np.mean(energy_values)  # Compute mean energy
energy_std = np.std(energy_values) / np.sqrt(len(energy_values))  # Corrected standard deviation

# Print results
print("=" * 43)
print(" Quantum Monte Carlo Simulation Summary ")
print("=" * 43)
print(f"Number of Electrons: {n_electrons}")
print(f"Number of Nuclei: {n_nuclei}")
print(f"Slater Orbital Exponent (a): {a:.5f}")
print(f"Final Energy Estimate: {energy_avg:.5f} ± {energy_std:.5f} Ha")
print(f"Acceptance Ratio: {accept_count / (nsteps * n_electrons):.4f}")
print("=" * 43)
