import os

import math
import numpy as np
import ot


import random
import time

import json

import mrcfile
import pandas as pd
import code.trn as trn, code.coords as coords
import importlib
importlib.reload(trn)

import torch
from unbalancedgw.vanilla_ugw_solver import exp_ugw_sinkhorn
from unbalancedgw.vanilla_ugw_solver import log_ugw_sinkhorn
from unbalancedgw._vanilla_utils import ugw_cost
from unbalancedgw.utils import generate_measure
from unbalancedgw._vanilla_utils import l2_distortion

from scipy.spatial.transform import Rotation

from Bio import PDB
from Bio.PDB.vectors import Vector, rotmat

class Term:
	def __init__(self, coef=1, exps=[0,0,0,0]):
		self.coef = coef
		self.exps = exps
		while len(self.exps) < 8:
			self.exps.append(0)
	
	def derivative(self, index):
		new_exps = [0,0,0,0,0,0,0,0]
		for i in range(8):
			if i != index:
				new_exps[i] = self.exps[i]
			else:
				new_exps[i] = self.exps[i] - 1
		return Term(coef=self.coef*self.exps[index], exps=new_exps)
	
	def evaluate(self, vals):
		s = self.coef
		if s == 0:
			return 0
		for i in range(min(len(vals), len(self.exps))):
			if self.exps[i] == 0:
				continue
			elif self.exps[i] == 1:
				s *= vals[i]
			elif self.exps[i] == 2:
				s *= vals[i] * vals[i]
			elif self.exps[i] == 3:
				s *= vals[i] * vals[i] * vals[i]
			else:
				print(self.exps[i])
				s *= vals[i] ** self.exps[i]
		return s
	
	def bunch_evaluate(self, bunch_vals):
		s = self.coef
		for i in range(min(len(bunch_vals), len(self.exps))):
			for j in range(self.exps[i]):
				s = np.multiply(s, bunch_vals[i])
		return s
	
	def __mul__(self, other):
		coef = self.coef * other.coef
		exps = []
		for i in range(8):
			exps.append(self.exps[i] + other.exps[i])
		return Term(coef=coef, exps=exps)
	
	def __neg__(self):
		return Term(coef=-self.coef, exps=self.exps)
	
	def __str__(self):
		s = ""
		for i in range(8):
			if self.exps[i] != 0:
				s += "q" + str(i) + "^" + str(self.exps[i])
		return str(self.coef) + "*" + s
	
	def is_similar(self, other):
		for i in range(8):
			if self.exps[i] != other.exps[i]:
				return False
		return True
	
	def simplify(self, vals):
		new_exps = [0,0,0,0,0,0,0,0]
		coef = self.coef
		for i in range(8):
			if vals[i] != 'x':
				coef *= vals[i]**self.exps[i]
			else:
				new_exps[i] = self.exps[i]
		return Term(coef=self.coef*self.exps[index], exps=new_exps)
		
	
class Polynomial:
	def __init__(self, terms=None):
		if terms is None:
			terms = []
		self.terms = {}
		for term in terms:
			self.terms[tuple(term.exps)] = term
		
	def add_term(self, term):
		if term.coef == 0:
			return
		if tuple(term.exps) not in self.terms:
			self.terms[tuple(term.exps)] = term
		else:
			self.terms[tuple(term.exps)].coef += term.coef

	def get_terms(self):
		terms = []
		for exps in self.terms:
			terms.append(self.terms[exps])
		return terms
		
	def derivative(self, index):
		res = Polynomial()
		for term in self.get_terms():
			res.add_term(term.derivative(index))
		return res
	
	def simplify(self, vals):
		res = ()
		for term in self.get_terms():
			res.add_term(term.simplify(vals))
		return res
	
	def evaluate(self, vals):
		t = 0
		for term in self.get_terms():
			t += term.evaluate(vals)
		return t
	
	def bunch_evaluate(self, bunch_vals):
		t = 0
		for term in self.get_terms():
			t = np.add(t, term.bunch_evaluate(bunch_vals))
		return t
	
	def __add__(self, other):
		res = Polynomial()
		for term in self.get_terms():
			res.add_term(term)
		for term in other.get_terms():
			res.add_term(term)
		return res
	
	def __sub__(self, other):
		res = Polynomial()
		for term in self.get_terms():
			res.add_term(term)
		for term in other.get_terms():
			res.add_term(-term)
		return res
	
	def __neg__(self):
		res = Polynomial()
		for term in self.get_terms():
			res.add_term(-term)
		return res
	
	def __mul__(self, other):
		res = Polynomial()
		for term1 in self.get_terms():
			for term2 in other.get_terms():
				res.add_term(term1*term2)
		return res
	
	def __str__(self):
		s = ""
		terms = self.get_terms()
		for i in range(len(terms)):
			if i > 0:
				s += " + "
			s += str(terms[i])
		return s

class Quaternion:
	def __init__(self, real_pol, i_pol, j_pol, k_pol):
		self.real_pol = real_pol
		self.i_pol = i_pol
		self.j_pol = j_pol
		self.k_pol = k_pol
		
	def conjugate(self):
		return Quaternion(self.real_pol, -self.i_pol, -self.j_pol, -self.k_pol)
	
	def __mul__(self, other):
		return Quaternion(self.real_pol*other.real_pol - self.i_pol*other.i_pol - self.j_pol*other.j_pol - self.k_pol*other.k_pol,
						  self.real_pol*other.i_pol + self.i_pol*other.real_pol + self.j_pol*other.k_pol - self.k_pol*other.j_pol,
						  self.real_pol*other.j_pol + self.j_pol*other.real_pol + self.k_pol*other.i_pol - self.i_pol*other.k_pol,
						  self.real_pol*other.k_pol + self.k_pol*other.real_pol + self.i_pol*other.j_pol - self.j_pol*other.i_pol)
	
	def __str__(self):
		return str(self.real_pol) + " + (" + str(self.i_pol) + ")i + (" + str(self.j_pol) + ")j + (" + str(self.k_pol) + ")k"
	


def get_quaternion_vals(theta, ax, ay, az):
	"""
	Compute the quaternion representation for a given rotation in angle-axis representation
	params:
		theta: the angle of the rotation in radians
		ax, ay, az: three floats in a way that *ax, ay, az) shows the 3d axis of the rotation

	retrun:
		q: is a list of length 4 that has values of the corresponding quaternion
	"""
	
	n = math.sqrt(ax**2 + ay**2 + az**2)
	return [math.cos(theta/2), math.sin(theta/2)*ax/n, math.sin(theta/2)*ay/n, math.sin(theta/2)*az/n]

def convert_to_poly(vals):
	return Quaternion(Polynomial([Term(coef=vals[0], exps=[0,0,0,0])]), Polynomial([Term(coef=vals[1], exps=[0,0,0,0])]), 
					  Polynomial([Term(coef=vals[2], exps=[0,0,0,0])]), Polynomial([Term(coef=vals[3], exps=[0,0,0,0])]))
 

def perform(x, y, z, vals):
	"""
	Apply a given rotation on a given point cloud and generate a new point cloud
	params:
		x, y, z: three lists with len(x)=len(y)=len(z) in a way that (x[i], y[i], z[i]) is the 3d coordinates of the i-th point
		vals: a list of length 4 that contains the values of the quaternion correponding the the ritation

	return:
		xr, yr, zr: three lists with len(xr)=len(yr)=len(zr) in a way that (xr[i], yr[i], zr[i]) is the 3d coordinates of the i-th point after the rotation
	"""

	xr = []
	yr = []
	zr = []
	q = convert_to_poly(vals)
	qs = q.conjugate()
	p = Quaternion(Polynomial([Term(coef=1, exps=[1,0,0,0])]), Polynomial([Term(coef=1, exps=[0,1,0,0])]), 
				   Polynomial([Term(coef=1, exps=[0,0,1,0])]), Polynomial([Term(coef=1, exps=[0,0,0,1])]))
	t = time.time()
	b = q*p*qs
	t = time.time()

	bunch_vals = [np.zeros(len(x)), np.array(x), np.array(y), np.array(z)]
	xr = list(b.i_pol.bunch_evaluate(bunch_vals))
	yr = list(b.j_pol.bunch_evaluate(bunch_vals))
	zr = list(b.k_pol.bunch_evaluate(bunch_vals))

	return xr, yr, zr

def change_quat_format(vals):
	return [vals[3], vals[0], vals[1], vals[2]]

def sample(fname, thresh, M, invalid=False, random_seed=None, verbose=False):
	"""
	Sample a given file using a topology representing network and return sampled points
	params:
		fname: the name and address of the mrc file for the input map
		thresh: the thresholding parameter, to be more robust to noise the values in the map with intensity < thresh
					will be changed to 0
		M: number of point you want to sample

	return:
		x, y, z: the coordinated of the sampled points
		x, y, z are lists so we have len(x)=len(y)=len(z)=M and
		(x[i], y[i], z[i]) shows the 3d coordinates of the i-th point
	"""
	
	if invalid:
		with mrcfile.open(fname, mode='r+', permissive=True) as mrc:
			mrc.header.map = mrcfile.constants.MAP_ID
			mrc.update_header_from_data()
	map_mrc = mrcfile.open(fname)
	map_original = map_mrc.data
	N = map_original.shape[0]
	psize_original = map_mrc.voxel_size.item(0)[0]
	psize = psize_original

	map_th = map_original.copy()
	if verbose:
		print(np.where(np.isnan(map_th)==True))
		print(map_th.sum())
	map_th[map_th < thresh] = 0

	rm0,arr_flat,arr_idx,xyz,coords_1d = trn.trn_rm0(map_th,M,random_seed=random_seed)

	l0 = 0.005*M # larger tightens things up (far apart areas too much to much, pulls together). smaller spreads things out
	lf = 0.5
	tf = M*10
	e0 = 0.3
	ef = 0.05

	rms,rs,ts_save = trn.trn_iterate(rm0,arr_flat,arr_idx,xyz,n_save=10,e0=e0,ef=ef,l0=l0,lf=lf,tf=tf,do_log=True,log_n=10)

	
	N_cube = max(map_mrc.header.tolist()[0],map_mrc.header.tolist()[1],map_mrc.header.tolist()[2])
	N_cube += N_cube%2
	
	x_res = []
	y_res = []
	z_res = []
	for p in rms[10]:
#         x_res.append(p[0])
#         y_res.append(p[1])
#         z_res.append(p[2])
		
		z_res.append(map_mrc.header.tolist()[24][2] + map_mrc.header.tolist()[10][2] * ((p[0] + N_cube//2)) / map_mrc.header.tolist()[2])
		y_res.append(map_mrc.header.tolist()[24][1] + map_mrc.header.tolist()[10][1] * ((p[1] + N_cube//2)) / map_mrc.header.tolist()[1])
		x_res.append(map_mrc.header.tolist()[24][0] + map_mrc.header.tolist()[10][0] * ((p[2] + N_cube//2)) / map_mrc.header.tolist()[0])

	return x_res,y_res,z_res

def find_optimal_alignment(x, y, z, x1, y1, z1, all_coup, verbose=False):
	A = []
	B = []
	for i in range(len(all_coup)):
		X = [x[all_coup[i][1]], y[all_coup[i][1]], z[all_coup[i][1]]]
		X = np.array(X)
		A.append(X)
	#     print(all_coup[i][1])
		X = [x1[all_coup[i][0]], y1[all_coup[i][0]], z1[all_coup[i][0]]]
		X = np.array(X)
		B.append(X)
	Abar = np.array([0., 0., 0.])
	Bbar = np.array([0., 0., 0.])
	for i in range(len(all_coup)):
		Abar += 1/len(all_coup) * A[i]
		Bbar += 1/len(all_coup) * B[i]
	if verbose:
		print(Abar)
		print(Bbar)
	H = np.zeros((3,3))
	for i in range(len(all_coup)):
		H += np.outer(A[i] - Abar, (B[i] - Bbar))
	# print(H)
	U, S, V = np.linalg.svd(H)
	# print(U,S,V)
	# R = np.matmul(np.transpose(V), np.transpose(U))
	R = np.matmul(U,V)
	d = np.linalg.det(R)
	R = np.matmul(np.matmul(U,np.diag([1,1,d])),V)
	if verbose:
		print(H)
		print(np.diag(S))
		print(np.matmul(np.matmul(U,np.diag(S)),V))
		print(V)
		print(R)
		print(np.linalg.det(R))
	r = Rotation.from_matrix(R)
	return Abar, Bbar, r

def generate_pdb_no_mrc(pdbname_in, pdbname_out, alignment):
	Abar = alignment['Abar']
	Bbar = alignment['Bbar']
	r = alignment['r']
	
	parser = PDB.PDBParser()
	io = PDB.PDBIO()
	print(pdbname_in)
	struct = parser.get_structure('6n52', pdbname_in)

	x_ins = []
	y_ins = []
	z_ins = []

	x_outs = []
	y_outs = []
	z_outs = []

	for model in struct:
		for chain in model:
			for residue in chain:
				for atom in residue:
					coord = atom.get_coord()

#                     x_in = (coord[2] - map_mrc_in.header.tolist()[24][2]) * map_mrc_in.header.tolist()[2] / map_mrc_in.header.tolist()[10][2] - N_cube_in//2
#                     y_in = (coord[1] - map_mrc_in.header.tolist()[24][1]) * map_mrc_in.header.tolist()[1] / map_mrc_in.header.tolist()[10][1] - N_cube_in//2
#                     z_in = (coord[0] - map_mrc_in.header.tolist()[24][0]) * map_mrc_in.header.tolist()[0] / map_mrc_in.header.tolist()[10][0] - N_cube_in//2
					
					x_in = coord[0]
					y_in = coord[1]
					z_in = coord[2]
				
					x_ins.append(x_in)
					y_ins.append(y_in)
					z_ins.append(z_in)

					x_in -= Bbar[0]
					y_in -= Bbar[1]
					z_in -= Bbar[2]


					x_temp, y_temp, z_temp = perform([x_in], [y_in], [z_in], change_quat_format(r.as_quat()))

					x_out = x_temp[0] + Abar[0]
					y_out = y_temp[0] + Abar[1]
					z_out = z_temp[0] + Abar[2]

					x_outs.append(x_out)
					y_outs.append(y_out)
					z_outs.append(z_out)

#                     z_temp = map_mrc_out.header.tolist()[24][2] + map_mrc_out.header.tolist()[10][2] * ((x_out + N_cube_out//2)) / map_mrc_out.header.tolist()[2]
#                     y_temp = map_mrc_out.header.tolist()[24][1] + map_mrc_out.header.tolist()[10][1] * ((y_out + N_cube_out//2)) / map_mrc_out.header.tolist()[1]
#                     x_temp = map_mrc_out.header.tolist()[24][0] + map_mrc_out.header.tolist()[10][0] * ((z_out + N_cube_out//2)) / map_mrc_out.header.tolist()[0]

					x_temp = x_out
					y_temp = y_out
					z_temp = z_out
					
					atom.set_coord([x_temp, y_temp, z_temp])


	#                 print(coord)
	#                 break

	io.set_structure(struct)
	io.save(pdbname_out)


def sample_pdb(filename, k):
#     parser = PDB.MMCIFParser()
	parser = PDB.PDBParser()
	print(filename)
	struct = parser.get_structure('6n52', filename)
	points = []
	t = 0
	for model in struct:
		for chain in model:
			for residue in chain:
				for atom in residue:
					coord = atom.get_coord()
					
					t += 1
					if t == k:
						t = 0
						points.append([coord[0], coord[1], coord[2]])
	points = np.array(points)
	return points

def generate_mu(clusters):
	s = 0
	for cluster in clusters:
		s += cluster.shape[0]
	return [1/s]*s

def dist(point1, point2):
	if point1.shape[0] == 2:
		return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
	if point1.shape[0] == 3:
		return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2
	
def generate_d_graph_1(points, adj):
	n = points.shape[0]
	max_dist = 10.0
	d = np.zeros((n, n)) + max_dist
	for src in range(n):
		not_visited = np.array([True] * n)
		shortest_path = np.array([max_dist] * n)
		shortest_path[src] = 0
		while not_visited.sum() > 0:
#             print(shortest_path)
#             print(not_visited)
			current_min = np.argmin((shortest_path - max_dist)*(not_visited))
			if shortest_path[current_min] >= max_dist:
				break
			if not_visited[current_min] == False:
				break
			for neigh in adj[current_min]:
				dis = math.sqrt(dist(points[current_min], points[neigh]))
#                 print(current_min, neigh, dis)
				if shortest_path[current_min] + dis < shortest_path[neigh]:
					shortest_path[neigh] = shortest_path[current_min] + dis
			not_visited[current_min] = False
		for i in range(n):
			d[src, i] = shortest_path[i]
	return d

def generate_d_graph(clusters, adjs):
	s = 0
	for cluster in clusters:
		s += cluster.shape[0]
	n = s
	ix = np.zeros((n,n))
	d = np.zeros((n,n))
		
	s = 0
	for k in range(len(clusters)):
		dc = generate_d_graph_1(clusters[k], adjs[k])
		for i in range(clusters[k].shape[0]):
			for j in range(clusters[k].shape[0]):
				ix[i+s, j+s] = 1
				d[i+s, j+s] = dc[i, j]**2
		s += clusters[k].shape[0]
	return d, ix
			
				
	

def generate_d(clusters):
	s = 0
	for cluster in clusters:
		s += cluster.shape[0]
	n = s
	ix = np.zeros((n,n))
	d = np.zeros((n,n))
		
	s = 0
	for cluster in clusters:
		for i in range(cluster.shape[0]):
			for j in range(cluster.shape[0]):
				ix[i+s, j+s] = 1
				d[i+s, j+s] = dist(cluster[i], cluster[j])
		s += cluster.shape[0]
	return d, ix


def gamma(mu, d_x, i_x, d_y):
#     print(mu.shape)
#     print(d_x.shape)
#     print(i_x.shape)
#     print(d_y.shape)
	return np.matmul(np.matmul(i_x, mu), d_y*d_y) - 2*np.matmul(np.matmul(d_x, mu), d_y)

def cost(gamma, mu):
	if mu.shape[0] != gamma.shape[0]:
		print(mu.shape)
		print(gamme.shape)
	s = 0
	for i in range(mu.shape[0]):
		for j in range(mu.shape[1]):
			s += mu[i,j] * gamma[i,j]
	return s

def generate_random_initial_method3(points_x, points_y, mu_x, mu_y):
	points_x_adjusted = []
	for k in range(len(points_x)):
		t = np.random.uniform(float(points_y.min()), float(points_y.max()), 3)
		for i in range(points_x[k].shape[0]):
			points_x_adjusted.append(points_x[k][i] + t)
	points_x_adjusted = np.array(points_x_adjusted)
	C = ot.dist(points_x_adjusted, points_y)
	#print(points_x_adjusted.shape)
	#print(mu_x.shape)
	#print(points_y.shape)
	#print(mu_y.shape)
	#print(C.shape)
	mu = ot.emd(mu_x, mu_y, C)
	return mu

def ugw(x, y, z, x1, y1, z1, init=None, eps=4000,
					rho=100000, rho2=100000,
					nits_plan=100, tol_plan=1e-10,
					nits_sinkhorn=100, tol_sinkhorn=1e-10):
	a = [1/len(x)] * len(x)
	b = [1/len(x1)] * len(x1)
	n = len(x)
	dx = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			dx[i][j] = (x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2

	n = len(x1)
	dy = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			dy[i][j] = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2 + (z1[i] - z1[j])**2

	a = torch.from_numpy(np.array(a))
	b = torch.from_numpy(np.array(b))
	dx = torch.from_numpy(np.array(dx))
	dy = torch.from_numpy(np.array(dy))

	pi, gamma = log_ugw_sinkhorn(a, dx, b, dy, init=init, eps=eps,
                                 	rho=rho, rho2=rho2,
                                 	nits_plan=nits_plan, tol_plan=tol_plan,
                                 	nits_sinkhorn=nits_sinkhorn, tol_sinkhorn=tol_sinkhorn,
                                 	two_outputs=True)

	return pi, gamma, float(l2_distortion(pi, gamma, dx, dy))

def build_coup(x, y, z, x1, y1, z1, pi, verbose=False):
	res = np.array(pi)
	all_coup = []
	for i in range(len(x1)):
		maxi = None
		maxx = 0
		s = 0
		for j in range(len(x)):
			s += res[j,i]
			if res[j,i] > maxx:
	#             print(maxx, res[i,j])
	#             print(i,j)
				maxx = res[j,i]
				maxi = j
	#     print('!')
		if verbose:
			print(i,maxi, maxx, s)
	#     if maxi/maxx >0.8:
		if maxi is not None:
	#     if i == 0 and maxi >1e-4:
			all_coup.append((i, maxi))
	return all_coup

def empot(name, base_path = 'data/embuild/'):
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


	optimal_cost = 1e10
	optimal_idx = 0
	#points_x = []
	#index = []
	#for i in range(len(file_paths_x)):
	#	file_path = file_paths_x[i]
	#	num = int(n_points / volumes.sum() * volumes[i])
	#	x, y, z = sample(file_path, 0.03, num, random_seed=1)
	#	print(file_path, num)
	#	points_x.append([])
	#	for j in range(num):
	#		index.append(i)
	#		points_x[-1].append([x[j],y[j],z[j]])
	#	points_x[-1] = np.array(points_x[-1])

	#	# print(points_x[-1].shape)
	#	print("cluster %d is sampled."%(i,))


	#points_y = []
	x, y, z = sample(file_path_y, 0.03, n_points, random_seed=1)
	#print(file_path_y, n_points)
	#for j in range(n_points):
	#	points_y.append([x[j],y[j],z[j]])

	#points_y = np.array(points_y)

	#print(points_y.shape)

	t = time.time()
	print("points_y is sampled.")

	num_repitition = 5

	for k in range(len(file_paths_x)):
		file_path = file_paths_x[k]
		num = int(n_points / volumes.sum() * volumes[k])
		x1, y1, z1 = sample(file_path, 0.03, num, random_seed=1)

		for i in range(num_repitition):
			t_perm  = None
			if num_repitition > 1:
				perm = np.random.permutation(len(x))
				t_perm = np.zeros((len(x), len(x1)))
				for i in range(len(x)):
					if i < len(x1) and perm[i] < len(x1):
						t_perm[i, perm[i]] = 1/len(x)
				t_perm = torch.from_numpy(t_perm)

			pi, gamma, dist = ugw(x, y, z, x1, y1, z1, init=t_perm)

			all_coup = build_coup(x, y, z, x1, y1, z1, pi)


			Abar, Bbar, r = find_optimal_alignment(x, y, z, x1, y1, z1, all_coup)
			print("Alignment of complex %s chain %d iteration %d"%(name, k, i))
			print("Abar: ", Abar)
			print("Bbar: ", Bbar)
			print("r: ", r.as_rotvec(degrees=True))
			print("Translational diff: ", np.linalg.norm(Abar - Bbar))
			print("Rotational diff: ", np.linalg.norm(r.as_rotvec(degrees=True)))
			alignment = {'Abar':Abar, 'Bbar':Bbar, 'r':r}

			os.makedirs("data/output_empot/" + name + "chains/", exist_ok=True)
			out_path = ("data/output_empot/" + name + "chains/empot_output_%s_%d.pdb")%(chain_ids[k],i)
			generate_pdb_no_mrc((base_path + name + "chains/" + name + "_chain_%s.pdb")%(chain_ids[k],), out_path, alignment)


	print("time: " + str(time.time() - t))
file = open("data/embuild/names.txt", "r")
lines = file.readlines()
for line in lines:
	print("\n\n\n\n\n" + line.strip())
	empot(line.strip())
#empot('3JCL')
