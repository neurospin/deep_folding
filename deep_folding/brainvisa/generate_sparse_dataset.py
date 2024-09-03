import numpy as np
import sparse
import os
import pandas as pd
from tqdm import tqdm

sides = ['L']
sulcus_list=['S.C.-sylv.']
"""
sulcus_list = ['F.Coll.-S.Rh.', 'S.F.median-S.F.pol.tr.-S.F.sup.', 'S.F.inf.-BROCA-S.Pe.C.inf.', \
               'S.Po.C.', 'fronto-parietal_medial_face.', 'F.I.P.', 'S.T.s.-S.GSM.', 'CINGULATE.', \
               'F.C.L.p.-S.GSM.', 'S.C.-S.Po.C.', 'S.F.inter.-S.F.sup.', 'F.C.M.post.-S.p.C.', \
               'S.s.P.-S.Pa.int.', 'S.Or.-S.Olf.', 'F.P.O.-S.Cu.-Sc.Cal.', 'S.F.marginal-S.F.inf.ant.', \
               'S.F.int.-F.C.M.ant.', 'S.T.i.-S.T.s.-S.T.pol.', 'S.F.int.-S.R.', 'Lobule_parietal_sup.', \
               'S.T.i.-S.O.T.lat.', 'S.Pe.C.', 'S.T.s.br.', 'Sc.Cal.-S.Li.', 'S.T.s.', 'F.C.L.p.-subsc.-F.C.L.a.-INSULA.', \
               'S.C.-sylv.', 'S.C.-S.Pe.C.', 'OCCIPITAL', 'S.Or.']
"""
root_save_dir = '/volatile/jl277509/data/UkBioBank/crops/2mm/'

for sulcus in tqdm(sulcus_list):
    for side in sides:

        data_dir = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/{sulcus}/mask'
        subjects = pd.read_csv(os.path.join(data_dir, f'{side}skeleton_subject.csv'))
        subjects = subjects['Subject'].tolist()
        skels = np.load(os.path.join(data_dir, f'{side}skeleton.npy'))

        # need to make sure that skels, foldlabels, and distbottoms have the same coordinates
        foldlabels = np.load(os.path.join(data_dir, f'{side}label.npy'))
        foldlabels[skels==0]=0

        distbottoms = np.load(os.path.join(data_dir, f'{side}distbottom.npy'))
        distbottoms[distbottoms==0]=-1
        distbottoms[skels==0]=0

        save_dir = f'{root_save_dir}{sulcus}/mask/{side}skeleton_sparse'
        if not os.path.isdir(os.path.join(save_dir, 'coords')):
            os.makedirs(os.path.join(save_dir, 'coords'))
        if not os.path.isdir(os.path.join(save_dir, 'skeleton')):
            os.makedirs(os.path.join(save_dir, 'skeleton'))
        if not os.path.isdir(os.path.join(save_dir, 'foldlabel')):
            os.makedirs(os.path.join(save_dir, 'foldlabel'))
        if not os.path.isdir(os.path.join(save_dir, 'distbottom')):
            os.makedirs(os.path.join(save_dir, 'distbottom'))

        nb_subs = len(skels)
        for k, subject in enumerate(subjects):
            skel = skels[k,:,:,:,0]
            s = sparse.COO.from_numpy(skel)
            np.save(os.path.join(save_dir, f'coords/{side}{subject}_coords.npy'), s.coords)
            np.save(os.path.join(save_dir, f'skeleton/{side}{subject}_skeleton_values.npy'), s.data)
            fold = foldlabels[k,:,:,:,0]
            s = sparse.COO.from_numpy(fold)
            np.save(os.path.join(save_dir, f'foldlabel/{side}{subject}_foldlabel_values.npy'), s.data)
            distb = distbottoms[k,:,:,:,0]
            s = sparse.COO.from_numpy(distb)
            np.save(os.path.join(save_dir, f'distbottom/{side}{subject}_distbottom_values.npy'), s.data)
