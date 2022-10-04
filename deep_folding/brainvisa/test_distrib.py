import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
from soma import aims
import random

# p = os.path.abspath('../')
# if p not in sys.path:
#     sys.path.append(p)

# q = os.path.abspath('../../')
# if q not in sys.path:
#     sys.path.append(q)

# from vae import *
# from preprocess import SkeletonDataset
import matplotlib.pyplot as plt

from deep_folding.brainvisa.utils.bbox import compute_max_box
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.brainvisa.utils.mask import compute_simple_mask

_DEFAULT_DATA_DIR = '/neurospin/dico/data/bv_databases/human/hcp/hcp'
_DEFAULT_MASK_DIR = '/neurospin/dico/data/deep_folding/current/mask/1mm/'
_DEFAULT_SAVING_DIR = '/neurospin/dico/lguillon/distmap/benchmark/deletion_2/200/skeletons/raw/'
_DEFAULT_BBOX_DIR = '/neurospin/dico/data/deep_folding/current/bbox/'

subjects_list='/neurospin/dico/lguillon/distmap/data/test_list.csv'

sulci_list = complete_sulci_name(['S.C.'], 'R')
mask, bbmin, bbmax = compute_simple_mask(sulci_list, 'R', _DEFAULT_MASK_DIR)
