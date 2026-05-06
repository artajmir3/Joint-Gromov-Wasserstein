from Bio import PDB
from Bio.PDB.vectors import Vector, rotmat
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO
import os


def split_pdb(filename):
	parser = PDB.MMCIFParser()
	#parser = PDB.PDBParser()
	print(filename)
	struct = parser.get_structure('6n52', filename)
	points = []
	t = 0
	for model in struct:
		for chain in model:
			print(chain)
			print(chain.id)
			for residue in chain:
				for atom in residue:
					coord = atom.get_coord()
					print(help(atom))
					t += 1
					if t == k:
						t = 0
						points.append([coord[0], coord[1], coord[2]])
	points = np.array(points)
	return points




def split_cif_by_chain(input_cif, output_dir="chains"):
    """
    Reads an mmCIF file and saves each chain as a separate PDB file.

    Parameters
    ----------
    input_cif : str
        Path to the input .cif file (mmCIF format)
    output_dir : str, optional
        Directory where chain files will be saved. Default = "chains"
    """


    # Initialize the mmCIF parser
    parser = MMCIFParser(QUIET=True)
    structure_id = os.path.basename(input_cif).split('.')[0]
    structure = parser.get_structure(structure_id, input_cif)

    output_dir = structure_id + output_dir
    # Create output directory if it doesn’t exist
    os.makedirs(output_dir, exist_ok=True)


    # Initialize a PDBIO object to save chains
    io = PDBIO()

    # Iterate over all chains in the structure
    num = 0
    for model in structure:
        for chain in model:
            num += 1
            chain_id = chain.id
            output_file = os.path.join(output_dir, f"{structure_id}_chain_{chain_id}.pdb")

            # Save this chain only
            io.set_structure(chain)
            io.save(output_file)
            print(f"Saved: {output_file}, num_chain: {num}")

# Example usage:
# split_cif_by_chain("example_structure.cif")

#split_pdb("6GZV.cif")
#split_cif_by_chain("6GZV.cif")
file = open("names.txt", "r")
lines = file.readlines()
for line in lines:
	split_cif_by_chain(line.strip() + ".cif")
