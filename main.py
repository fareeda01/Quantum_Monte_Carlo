import numpy as np
from input_parser import parse_vmc_input
from vmc_algorithm import metropolis
from local_energy import local_energy
from wavefunction import wavefunction  # if separate
from quantum_force import quantum_force
import sys

# Entry point
def run_simulation(input_file):
    # Parse input
    params = parse_vmc_input(input_file)
    nuclei_coords = params['nuclei_coords']
    charge = [params['charge']] * params['n_nuclei']
    exponent = params['exponent'] * params['n_nuclei']
    n_electrons = params['n_electrons']
    n_steps = params['no_of_steps']
    delta_t = 0.01  # or from input

    # Initialize electron positions randomly
    electron_positions = [np.random.normal(0.0, 1.0, size=3) for _ in range(n_electrons)]

    accepted = 0
    energy_accumulator = 0.0

    for step in range(n_steps):
        for i in range(n_electrons):
            electron_positions[i], accepted_move = metropolis(electron_positions[i], nuclei_coords, exponent, delta_t)
            if accepted_move:
                accepted += 1

        # Compute energy
        E_loc = local_energy(electron_positions, nuclei_coords, charge, exponent)
        energy_accumulator += E_loc

    avg_energy = energy_accumulator / n_steps
    acceptance_ratio = accepted / (n_steps * n_electrons)

    print(f"\nInput: {input_file}")
    print(f"Average Energy: {avg_energy:.6f} Hartree")
    print(f"Acceptance Ratio: {acceptance_ratio:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    run_simulation(input_file)

