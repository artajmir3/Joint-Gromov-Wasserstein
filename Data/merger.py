import os
from Bio.PDB import PDBParser, PDBIO, Structure

def idx_from_name(fname):
	return fname.split('.')[0].split('_')[-1]

def get_pdb_files(folder):
	files = {}
	for f in os.listdir(folder):
		#print(f, idx_from_name(f))
		if idx_from_name(f) not in files:
			files[idx_from_name(f)] = []
		files[idx_from_name(f)].append(os.path.join(folder, f))
	print(files)
	return files
	#return sorted(
	#	[os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdb")]
	#)

def merge_pdbs(pdb_files):
	parser = PDBParser(QUIET=True)

	# Create a new empty structure
	merged_structure = Structure.Structure("merged")

	for i, pdb_file in enumerate(pdb_files):
		structure = parser.get_structure(f"struct_{i}", pdb_file)
		print(i, pdb_file, structure)
		# Take first model only (common for most PDBs)
		model = structure[0]
		model.id = i  # ensure unique model IDs

		merged_structure.add(model)

	return merged_structure

def save_structure(structure, output_file):
	io = PDBIO()
	io.set_structure(structure)
	io.save(output_file)

#if __name__ == "__main__":
file = open("data/embuild/names.txt", "r")
lines = file.readlines()
for line in lines:
	name = line.strip()
	print(name)
	#folder = "data/output/%schains"%(name, )
	folder = "data/output_empot/%schains"%(name, )
	print(folder)

	pdb_files = get_pdb_files(folder)

	if not pdb_files:
		print("No PDB files found.")
		exit()

	#os.makedirs("data/merged_jgw/" + name + "/", exist_ok=True)
	os.makedirs("data/merged_empot/" + name + "/", exist_ok=True)

	for i in pdb_files.keys():
		output_file = "data/merged_empot/%s/merged_%s.pdb"%(name, i)
		#output_file = "data/merged_jgw/%s/merged_%s.pdb"%(name, i)
		merged_structure = merge_pdbs(pdb_files[i])
		save_structure(merged_structure, output_file)
		print(f"Merged {len(pdb_files[i])} PDB files into '{output_file}'")
