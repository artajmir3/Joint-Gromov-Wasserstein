import os
from chimerax.core.commands import run

def pdbs_to_density_maps(session, folder_path, resolution=1, output_dir="maps"):
	"""
	Convert all PDB files in a folder to density maps using ChimeraX molmap.

	Parameters
	----------
	session : chimerax.core.session.Session
		The current ChimeraX session (automatically provided in ChimeraX shell).
	folder_path : str
		Path to folder containing .pdb files.
	resolution : float, optional
		Resolution (Å) for molmap command. Default = 5.0 Å.
	output_dir : str, optional
		Folder to save generated maps (created if not present).
	"""

	folder_path = os.path.abspath(folder_path)
	output_dir = os.path.join(folder_path, output_dir)
	os.makedirs(output_dir, exist_ok=True)

	pdb_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdb") or f.lower().endswith(".cif")]

	if not pdb_files:
		print(f"No PDB files found in {folder_path}")
		return

	for pdb in pdb_files:
		pdb_path = os.path.join(folder_path, pdb)
		base_name = os.path.splitext(pdb)[0]
		print(f"\nProcessing {pdb_path} ...")

		# Open structure
		run(session, f"open {pdb_path}")

		# Create density map
		run(session, f"molmap #1 {resolution}")

		# Save the map
		output_path = os.path.join(output_dir, f"{base_name}_map.mrc")
		run(session, f"save {output_path} format mrc")

		# Close all models before next iteration
		run(session, "close all")

		print(f"Saved map: {output_path}")

	print("\n✅ All maps generated successfully.")

def generate_all_maps(session, parent_folder = "C:\\Users\\artaj\\Data\\embuild-server\\embuild\\"):
	file = open(parent_folder + 'names.txt', 'r')
	lines = file.readlines()
	for line in lines:
		pdbs_to_density_maps(session, parent_folder + line.strip() +"chains" )
	pdbs_to_density_maps(session, parent_folder)