import mrcfile
import numpy as np
import trn

from quaternions import perform
from alignment import change_quat_format

from Bio import PDB
from Bio.PDB.vectors import Vector, rotmat


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
    tf = M*8
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

def generate_pdb(mrcname_in, mrcname_out, pdbname_in, pdbname_out, alignment):
    Abar = alignment['Abar']
    Bbar = alignment['Bbar']
    r = alignment['r']
    
    map_mrc_in = mrcfile.open(mrcname_in)
    N_cube_in = max(map_mrc_in.header.tolist()[0],map_mrc_in.header.tolist()[1],map_mrc_in.header.tolist()[2])
    N_cube_in += N_cube_in%2
    
    map_mrc_out = mrcfile.open(mrcname_out)
    N_cube_out = max(map_mrc_out.header.tolist()[0],map_mrc_out.header.tolist()[1],map_mrc_out.header.tolist()[2])
    N_cube_out += N_cube_out%2
    
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

def generate_pdbs(mrcname, alignments, outputname):
    i = 0
    for alignment in alignments:
        generate_pdb(alignment['struct'][0], mrcname, alignment['struct'][2], outputname + str(i) + '.pdb', alignment)
        i += 1

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

def mrc_volume(fname):
    map_mrc = mrcfile.open(fname)
    return map_mrc.data.sum()
