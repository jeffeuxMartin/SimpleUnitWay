# %%
import os
from tqdm import tqdm
import numpy as np, pandas as pd, torch
from itertools import groupby

from src.UNITS import UNITS


def downsampler(s, skip=6, start=0):
    return s[start::skip]

def deduplicate(s):
    res = [item for item, grouper in groupby(s)]
    return (''.join(res) if isinstance(s, str) 
            else type(s)(res))

# processor = lambda s: downsampler(s)

def split_processor(splitname, processor=downsampler, expname='dwn_6_0'):
    split_unit_df = pd.read_csv(f'Data/{splitname}.unit.tsv', sep='\t')
    split_eng_df = pd.read_csv(f'Data/{splitname}.eng.tsv', sep='\t')

    assert all(split_unit_df['utt_id'] == split_eng_df['utt_id'])

    def unit_parser(txt):
        unit_str = txt.strip().split()
        return ''.join(UNITS[int(u)] for u in unit_str)

    split_unit = [unit_parser(l) for l in (
        tqdm(split_unit_df['units'], desc='Transform units...'))]
    split_eng = list(split_eng_df['transcription'])
    assert len(split_unit) == len(split_eng)
    
    if not os.path.isdir(f'DataProcessed/{expname}'):
        os.mkdir(f'DataProcessed/{expname}')

    with open(f'DataProcessed/{expname}/{splitname}.unit', 'w') as fout:
        for line in (
            tqdm(split_unit, desc='Save units...')):
            print(processor(line), file=fout)

    with open(f'DataProcessed/{expname}/{splitname}.eng', 'w') as fout:
        for line in (
            tqdm(split_eng, desc='Save texts...')):
            print(line, file=fout)

for proc, expname in [
    [lambda s: s, 'original'],
    [lambda s: deduplicate(s), 'dedup'],
    [downsampler, 'dwn_6_0'],
    [lambda s: deduplicate(downsampler(s)), 'dwn_6_0__dedup'],
    [lambda s: (downsampler(s, 6, 3)), 'dwn_6_3'],
    [lambda s: deduplicate(downsampler(s, 6, 3)), 'dwn_6_3__dedup'],
]:
    split_processor('train', proc, expname)
    split_processor('dev',   proc, expname)
