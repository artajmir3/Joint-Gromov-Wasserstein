from alignment import build_coup, find_optimal_alignment, plot_alignment
from ot_utils import ugw
import matplotlib.pyplot as plt

def align_pointcloud(x, y, z, x1, y1, z1, num=1, verbose=True):
    alignments = []
    for i in range(num):
        if verbose:
            print(i)
            
        t_perm  = None
#         if num > 1:
#             perm = np.random.permutation(len(x))
#             t_perm = np.zeros((len(x), len(x)))
#             for i in range(len(x)):
#                 t_perm[i, perm[i]] = 1/len(x)
#             t_perm = torch.from_numpy(t_perm)
        
        pi, gamma, dist = ugw(x, y, z, x1, y1, z1, init=t_perm)
        
        all_coup = build_coup(x, y, z, x1, y1, z1, pi)
        
        Abar, Bbar, r = find_optimal_alignment(x, y, z, x1, y1, z1, all_coup)
        
        if verbose:
            plot_alignment(x, y, z, x1, y1, z1, Abar, Bbar, r, all_coup)
        
        covering = [0] * len(x)
        for i in range(len(all_coup)):
            covering[all_coup[i][1]] += 1
        
        alignments.append({'Abar':Abar, 'Bbar':Bbar, 'r':r, 'score':dist, 'covering':covering})
    return alignments

def align_substruct(x, y, z, structs, num=1, num_points=500):
    # structs is the list of possible structures came from prediction of alphafold for a chain
    # the format is struct = [('mrc address', threshold value, 'pdb address'), ...]
    alignments = []
    for struct in structs:
#         x1, y1, z1 = sample(struct[0], struct[1], num_points)
        x1 = []
        y1 = []
        z1 = []
        for i in range(struct[0].shape[0]):
            x1.append(struct[0][i, 0])
            y1.append(struct[0][i, 1])
            z1.append(struct[0][i, 2])
        struct_alignments = align_pointcloud(x, y, z, x1, y1, z1, num=num)
        for struct_alignment in struct_alignments:
            struct_alignment['struct'] = struct
        alignments += struct_alignments
    return alignments

def choose_alignments(x, y, z, possible_alignments, verbose=False):
    num_structs = len(possible_alignments)
    num_comb = 1
    for i in range(num_structs):
        num_comb *= len(possible_alignments[i])
    best_overlaps = float('inf')
    best_score = float('inf')
    best_alignments = None
    for i in range(num_comb):
        alignments = []
        comb_number = i
        if verbose:
            print('!')
        for j in range(num_structs):
            if verbose:
                print(comb_number%len(possible_alignments[j]))
            alignments.append(possible_alignments[j][comb_number%len(possible_alignments[j])])
            comb_number = comb_number//len(possible_alignments[j])
        if verbose:
            print('!')
        overlaps = 0
        score = 0
        covering = [0] * len(alignments[0]['covering'])
        x_ov = []
        y_ov = []
        z_ov = []
        for alignment in alignments:
            score += alignment['score']
            for j in range(len(alignment['covering'])):
                if covering[j] > 0  and alignment['covering'][j] > 0:
                    overlaps += 1
                    x_ov.append(x[j])
                    y_ov.append(y[j])
                    z_ov.append(z[j])
                covering[j] += alignment['covering'][j]
        if verbose:
            print('###')
            print(overlaps)
            print(score)
            print(best_overlaps)
            print(best_score)
        if verbose:
            fig = plt.figure()
            # ax = fig.gca(projection='3d', adjustable='box')
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x_ov, y_ov, z_ov,  marker='o')
            ax.scatter(x, y, z,  marker='o', alpha=0.5)
            plt.show()
        if (overlaps < best_overlaps) or ((overlaps == best_overlaps) and (score <= best_score)):
            best_alignments = alignments
            best_overlaps = overlaps
            best_score = score
            if verbose:
                print('BEST!!!!!!!!!!!!!!!!!!')
    return best_alignments