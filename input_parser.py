# Import numercial python library 
import numpy as np
# to access command‑line arguments (module to interact strongly with interpreter)
import sys
from typing import Dict, List

class InvalidInputError(Exception):
    """
    raise error if the input file format is incorrect
    """
    pass

def clean_line(line: str):
    """function to remove text from comment (#) or any 
    leading/trailing space strip() while reading input """
    return line.split("#", 1)[0].strip()

def parse_vmc_input(input_file):
    with open(input_file, 'r') as f:
        def clean_read(field: str):
            while True:
                raw = f.readline()
                if not raw:
                    raise InvalidInputError(f'Error while reading {field}')
                parameters = clean_line(raw) 
                if parameters:
                    return parameters
        n_nuclei = int(clean_read('n_nuclei'))
        charge = float(clean_read("nuclear_charge"))

        # initialise coordinates list 
        coords: List[np.ndarray] = []

        # loop over coordinates lines for n_nuclei
        for i in range (n_nuclei):
            xyz_coord = clean_read(f"nucleus {i+1} coordinates")
            #xyz_coord = clean_read(f.readline())
            parts = xyz_coord.split()  # split on any whitespace
            if len(parts) < 3: 
                raise InvalidInputError(f"Line {i+3}: expected 3 floats for nucleus {i+1}.")
            coords.append(np.array([float(p) for p in parts[:3]]))

        n_elec = int(clean_read("n_electrons"))
        zeta = float(clean_read("orbital_exponent"))
        delta_t = float(clean_read("time_step"))
        n_steps = int(clean_read("num_of_steps"))
    return{
        "n_nuclei": n_nuclei,
        "charge": charge,
        "nuclei_coords": coords,
        "n_electrons": n_elec,
        "exponent": zeta,
        "no_of_steps": n_steps
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        params = parse_vmc_input(input_file)
    except InvalidInputError as e:
        print(f"Error parsing input file: {e}", file=sys.stderr)
        sys.exit(1)

    print("== parsed parameters ==")
    for k, v in params.items():
        print(f"{k:15} = {v}")

