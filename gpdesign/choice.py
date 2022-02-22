#!/usr/bin/python3

import numpy as np

# 加载自定义包
from base_classes_funcs import *

def choice_individual_simple(candidates:list(), num:int) ->list():
    res, si_candi = [], np.argsort([-individual.fitness for individual in candidates])
    for i_candi in si_candi:
        res.append(candidates[i_candi])
    return res[:num]


def choice_individual_reg(candidates:list(), num:int) ->list():
    res, si_candi = [], np.argsort([-individual.fitness for individual in candidates])
    #import pdb; pdb.set_trace()
    for i_candi in si_candi:
        indi = candidates[i_candi]
        if np.isinf(indi.fitness) or np.isnan(indi.fitness): continue
        x = indi.stats['pnl'][-500:] * (-1, 1)[indi.fitness_raw>0]
        if len(res) > 0:
            y = np.stack([I.stats['pnl'][-500:]*(-1,1)[I.fitness_raw>0] for I in res], axis=1)
            if maxcorr_with_pool(x, y) < 0.7:
                res.append(indi)
        else:
            res.append(indi)
    return res[:num]


