from common.paths import MARIUS_CLEAN
# from common.data_aggregation import [GATHERING FUNCS]
from pathlib import Path
from itertools import product
import re

import numpy as np
import pandas as pd

"""
Takes the data files which have been cleaned from Marius' data and projects the co-ordinates onto the direction vector
to isolate the direction of movement. We should therefore not mix the trials for calculating the basis vector bit, I think.
There is likely enough data in 8 trials to get an accurate representation.

So here I want to:
    Go through each condition (2-7 inc.), take xyz for each participant, create the rail coordinate, and save the file
    again.

"""

def get_basis_vector(xyz):
    centre = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - centre, full_matrices=False)
    direction = vt[0] # unit vector along the track
    # s = (xyz - centre) @ direction
    return direction, centre

if __name__ == "__main__":
    conditions = range(2, 8)
    pairs = range(3,11)
    combinations = list(product(conditions, pairs)) #reduce number of indentations

    chunks = [] # the csvs that are to be grouped together for the co-ordinate projection
    for combo in combinations:
        (cond, pair) = combo #unpack the sublist
        chunks.append([list(MARIUS_CLEAN.glob(f'condition_{cond}/pair_{pair:02d}/*A*.csv')), list(MARIUS_CLEAN.glob(f'condition_{cond}/pair_{pair:02d}/*B*.csv'))])

    for lst in chunks:
        pathsA = lst[0]
        pathsB = lst[1]
        m = re.search(r"condition_(\d+)/pair_(\d+)/p_", str(pathsA[0]))
        cond, pair = int(m[1]), int(m[2])
        dfsA = []
        for path in pathsA:
            df = pd.read_csv(path)
            dfsA.append(df)
        dfsB = []
        for path in pathsB:
            df = pd.read_csv(path)
            dfsB.append(df)

        dfA = pd.concat(dfsA)
        dfB = pd.concat(dfsB)
        
        xyzA = dfA[['x', 'y', 'z']].to_numpy()
        vA, cA = get_basis_vector(xyzA)
        
        xyzB = dfB[['x', 'y', 'z']].to_numpy()
        vB, cB = get_basis_vector(xyzB)
        
        if vA @ vB < 0:
            vB *= -1
        if vA[2] < 0:
            vA, vB = -vA, -vB
        
        sA = (xyzA - cA) @ vA
        sB = (xyzB - cB) @ vB

        dfA['v'] = sA
        dfB['v'] = sB

        trial_values = dfA['new_trial_no'].unique().tolist()
        for trial in trial_values:
            dfA_trial = dfA[dfA['new_trial_no'] == trial]
            dfB_trial = dfB[dfB['new_trial_no'] == trial]
            outdir = f"{MARIUS_CLEAN}/projected/condition_{cond}/pair_{pair:02d}"
            Path(outdir).mkdir(parents=True, exist_ok=True)
            dfA_trial.to_csv(f'{outdir}/p_{pair:02d}A_trial_{trial:02d}.csv', index=False)
            dfB_trial.to_csv(f'{outdir}/p_{pair:02d}B_trial_{trial:02d}.csv', index=False)
