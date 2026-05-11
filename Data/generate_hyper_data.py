from Bio import PDB
from Bio.PDB.vectors import Vector, rotmat
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO
import os
import json


def generate_hyper_data_pdb(filename):
	structure_id = filename.split('.')[0]

	parser = PDB.MMCIFParser()
	#parser = PDB.PDBParser()
	print(filename)
	struct = parser.get_structure('6n52', filename)
	points = []
	t = 0
	num_chains = 0
	num_atoms = []
	chain_files = []
	for model in struct:
		for chain in model:
			print(chain)
			print(chain.id)
			num_chains += 1
			num_atoms.append(0)
			chain_files.append(structure_id + "_chain_" + chain.id + ".pdb")
			for residue in chain:
				for atom in residue:
					coord = atom.get_coord()
					t += 1
					num_atoms[-1] += 1

	f = open(structure_id+"chains/hyper.json", 'w')
	json.dump({"num_chains": num_chains, "num_atoms": num_atoms, "chain_files": chain_files}, f)
	f.close()




file = open("names.txt", "r")
lines = file.readlines()
for line in lines:
	generate_hyper_data_pdb(line.strip() + ".cif")
