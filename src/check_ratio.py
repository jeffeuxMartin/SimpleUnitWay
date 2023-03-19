import numpy as np, pandas as pd, torch

split = 'train'
split = 'dev'
original = f'/storage/LabJob/Projects/SimpleUnitWay/DataProcessed/original/{split}.unit'
reduced = f'/storage/LabJob/Projects/SimpleUnitWay/DataProcessed/dedup/{split}.unit'
with open(original) as f: originaldata = f.read().strip().split('\n')
with open(reduced) as f: reduceddata = f.read().strip().split('\n')
len_original = [len(l) for l in originaldata]
len_reduced = [len(l) for l in reduceddata]
len_original = np.array(len_original)
len_reduced = np.array(len_reduced)
ratio = len_original / len_reduced
ratio.mean()

# train = 1.8141369005341414
# dev   = 1.7633476183925756
