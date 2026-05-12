import numpy as np
import ot

import json
import os
import argparse
import time

from code.jgw import jgw_wrapper
from code.cryo_utils import sample, generate_pdb_no_mrc
from code.alignment import build_coup, find_optimal_alignment


		
def generate_random_initial_method3(points_x, points_y):
	points_x_adjusted = []
	for k in range(len(points_x)):
		t = np.random.uniform(5 * float(points_y.min()), 5 * float(points_y.max()), 3)
		for i in range(points_x[k].shape[0]):
			points_x_adjusted.append(points_x[k][i] + t)
	points_x_adjusted = np.array(points_x_adjusted)
	C = ot.dist(points_x_adjusted, points_y)
	
	mu_y = [1/points_y.shape[0]] * points_y.shape[0]
	mu_y = np.array(mu_y)
	mu_x = [1/points_x_adjusted.shape[0]] * points_x_adjusted.shape[0]
	mu_x = np.array(mu_x)

	#print(points_x_adjusted.shape)
	#print(mu_x.shape)
	#print(points_y.shape)
	#print(mu_y.shape)
	#print(C.shape)
	mu = ot.emd(mu_x, mu_y, C)
	return mu


def apply_jgw(name, base_path = 'data/embuild/', output_bpath="data/output/"):
	f = open(base_path + name + "chains/hyper.json", 'r')
	hyper = json.load(f)
	f.close()

	file_path_y = base_path + "maps/" + name + "_map.mrc"

	chain_ids = []
	file_paths_x = []
	for c in hyper["chain_files"]:
		chain_ids.append(c.split('.')[0].split('_')[-1])
		file_paths_x.append(base_path + name + "chains/maps/" + name + "_chain_" + chain_ids[-1] + "_map.mrc")

	volumes = np.array(hyper["num_atoms"])

	n_points = 1000

	points_x = []
	index = []
	for i in range(len(file_paths_x)):
		file_path = file_paths_x[i]
		num = int(n_points / volumes.sum() * volumes[i])
		x, y, z = sample(file_path, 0.03, num, random_seed=1)
		print(file_path, num)
		points_x.append([])
		for j in range(num):
			index.append(i)
			points_x[-1].append([x[j],y[j],z[j]])
		points_x[-1] = np.array(points_x[-1])

		# print(points_x[-1].shape)
		print("cluster %d is sampled."%(i,))


	points_y = []
	x, y, z = sample(file_path_y, 0.03, n_points, random_seed=1)
	print(file_path_y, n_points)
	for j in range(n_points):
		points_y.append([x[j],y[j],z[j]])

	points_y = np.array(points_y)
	print(points_y.shape)
	print("points_y is sampled.")

	mus = []
	num_repitition = 5
	optimal_cost = 1e10
	optimal_idx = 0

	t = time.time()
	for i in range(num_repitition):
		init = None
		if i > 0:
			init = generate_random_initial_method3(points_x, points_y)
		mu, dif, costs = jgw_wrapper(points_x, points_y, epsilon=1e7, max_iter=500, ot_solver='emd', diff_thrsh=1e-3)
		mus.append(mu)
	print(time.time() - t)
	print("JGW finished")

	x = []
	y = []
	z = []
	for i in range(points_y.shape[0]):
		x.append(points_y[i, 0])
		y.append(points_y[i, 1])
		z.append(points_y[i, 2])

	ss = 0

	for idx in range(num_repitition):
		ss = 0
		for k in range(len(file_paths_x)):
			x1 = []
			y1 = []
			z1 = []
			for i in range(points_x[k].shape[0]):
				x1.append(points_x[k][i, 0])
				y1.append(points_x[k][i, 1])
				z1.append(points_x[k][i, 2])

			all_coup = build_coup(x, y, z, x1, y1, z1, mu[ss: , ])

			Abar, Bbar, r = find_optimal_alignment(x, y, z, x1, y1, z1, all_coup)
			print("Alignment of complex %s chain %d idx %d"%(name, k, idx))
			print("Abar: ", Abar)
			print("Bbar: ", Bbar)
			print("r: ", r.as_rotvec(degrees=True))
			print("Translational diff: ", np.linalg.norm(Abar - Bbar))
			print("Rotational diff: ", np.linalg.norm(r.as_rotvec(degrees=True)))
			alignment = {'Abar':Abar, 'Bbar':Bbar, 'r':r}

			os.makedirs(output_bpath + name + "chains/", exist_ok=True)
			out_path = (output_bpath + name + "chains/JGW_output_%s_%d.pdb")%(chain_ids[k],idx)
			# if idx == optimal_idx:
			# 	out_path = (output_bpath + name + "chains/CHOSEN_JGW_output_%s_%d.pdb")%(chain_ids[k],idx)
			generate_pdb_no_mrc((base_path + name + "chains/" + name + "_chain_%s.pdb")%(chain_ids[k],), out_path, alignment)

			ss += len(x1)


parser = argparse.ArgumentParser(description="A script that applies jgw on a dataset.")
parser.add_argument("-b", "--basepath", help="The base path of data folder", required=False, default='data/embuild/')
parser.add_argument("-d", "--dataset", help="The dataset for the experiment, put full for the full dataset of PDB ID for one", required=False, default='full')
parser.add_argument("-o", "--output", help="The base output path", required=False, default="data/output/")

args = parser.parse_args()
print(args.dataset)

if args.dataset != 'full':
	apply_jgw(args.dataset, base_path=args.basepath, output_bpath=args.output)
else:
	file = open(args.basepath + "names.txt", "r")
	lines = file.readlines()
	for line in lines:
		print("\n\n\n\n\n" + line.strip())
		apply_jgw(line.strip(), base_path=args.basepath, output_bpath=args.output)

# apply_jgw('3JCL')
