#!/usr/bin/python3
"""
一些基础的类和函数
注意, import关系需要能够拓扑排序(不要相互调用).
"""

# 加载不应该被COPY的包
import io2 as io
import deap
from deap import algorithms, base, creator, gp, tools
from prettytable import PrettyTable

# COPY #
import copy
import random
import warnings
import sys
import pdb
import inspect
import shutil
import os
import time
import argparse
import datetime
import collections
import traceback
import math
import subprocess
import yaml
import multiprocessing
from itertools import repeat
from functools import partial
from copy import deepcopy

# 禁用NumPy自带的多线程, 这样一个进程最多100% CPU占用. 这段代码必须保证在import numpy前执行.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# 加载外部包
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class EvalInfo:
    """currently contains shape and unit information to evaluate."""
    def __init__(self, shape, unit, ret_type):
        """shape should be tuple, while unit should be np.ndarray."""
        self.shape = shape
        self.unit = unit
        self.ret_type = ret_type


class Individual:
    """an individual in genetic programming"""
    def __init__(self, expr=None, fitness=None, fitness_raw=None, pnl=None, turnover=None):
        self.expr = expr
        self.fitness = fitness
        self.fitness_raw = fitness_raw
        self.pnl = pnl
        self.turnover = turnover
        self.stats = dict()


class IllegalSyntex(Exception):
    """illegal syntax when checking."""
    pass


class Array2d:
    """a symbolic class, only used for STGP."""
    pass


class Array2dNeutralise:
    """由于中性化参数也是需要根据输入的数据调整的, 因此必须把X_NEUTRALISE作为一个参数传入. 它由这个类代表."""
    pass


class Array2dValid:
    """同上. 表示例如UnivTOP4000.valid"""
    pass


class Array3d:
    """a symbolic class, only used for STGP."""
    pass


class Ephemeral:
    """a class representing ephemeral constants."""
    pass


class FinalResult:
    """a class representing the final result, usual generated from ewma."""
    pass


# 修改 ###################################################################

# 可以在这里添加自定义STGP类, 和需要被COPY的自定义函数.

# 修改 ###################################################################


def check_same_unit(unit_1, unit_2):
    """check whether two units are numerically similar, by calculating the chebyshev distance."""
    epsilon = 0.001
    if np.max(np.abs(unit_1 - unit_2)) <= epsilon:
        return True
    else:
        return False


def replace_inf(arr):
    ret = arr.copy()
    ret[np.isinf(ret)] = np.nan
    return ret


def mask(arr):
    """returns a boolean mask of an arr"""
    return ~np.isnan(arr)


def imposter(arr):
    """returns an imposter of an arr"""
    return np.full_like(arr, np.nan)


def ts_delay(arr, window=1, axis=0):
    """delay by window along an axis. the first/last window rows are filled with nan. """
    ret = arr.copy()
    if window >= 0:  # if window == 0, returns exactly the input
        slc1 = [slice(None)] * len(arr.shape)
        slc1[axis] = slice(window, arr.shape[axis])
        slc2 = [slice(None)] * len(arr.shape)
        slc2[axis] = slice(0, arr.shape[axis] - window)
        slc3 = [slice(None)] * len(arr.shape)
        slc3[axis] = slice(0, window)
        ret[tuple(slc1)] = ret[tuple(slc2)]
        ret[tuple(slc3)] = np.nan
    else:  # delay by negative, fetching future data
        slc1 = [slice(None)] * len(arr.shape)
        slc1[axis] = slice(-window, arr.shape[axis])
        slc2 = [slice(None)] * len(arr.shape)
        slc2[axis] = slice(0, window)
        slc3 = [slice(None)] * len(arr.shape)
        slc3[axis] = slice(window, arr.shape[axis])
        ret[tuple(slc2)] = ret[tuple(slc1)]
        ret[tuple(slc3)] = np.nan
    return ret


def rolling(arr, window, f, axis=0):
    """
    rolling with NumPy and for loop. Note: np.nanxxx is much slower than np.xxx
    :param f: a function which accepts array and axis as the first two arguments. e.g. np.nanstd
    """
    ret = []
    slc = [slice(None)] * len(arr.shape)
    for ti in range(arr.shape[axis]):
        slc[axis] = slice(max(ti - window + 1, 0), ti + 1)
        rolling_data = arr[tuple(slc)]
        ret.append(f(rolling_data, axis))
    ret = np.stack(ret, axis=axis)
    return ret


def rolling_cross(x, y, window, f, axis):
    """
    rolling fucn of two arrays
    :param f: a function which accepts two arrays and axis as the first three arguments. e.g. cal_pearson_r
    """
    ret = []
    slc = [slice(None)] * len(x.shape)
    for ti in range(x.shape[axis]):
        slc[axis] = slice(max(ti - window + 1, 0), ti + 1)
        rolling_x = x[tuple(slc)]
        rolling_y = y[tuple(slc)]
        ret.append(f(rolling_x, rolling_y, axis=axis))
    ret = np.stack(ret, axis=axis)
    return ret


def ts_quantile_aux(arr, axis, standardize):
    """用于ts_quantile的辅助函数, 会作为f传入rolling中. axis参数不管设成什么都按照0来算. """
    arr_rank = (arr[-1, ...][np.newaxis, ...] > arr).sum(0).astype('float')
    arr_rank[np.isnan(arr[-1, ...])] = np.nan
    if standardize:
        arr_rank = arr_rank / mask(arr).sum(0)
    return arr_rank


def rank(arr, axis=1, method='average'):
    """rank along an axis, starting at zero. deals with nan. """
    ranks = stats.rankdata(arr, method=method, axis=axis).astype('float')  # nans are given largest rank
    ranks[np.isnan(arr)] = np.nan  # mstats.rankdata assign 0 to masked values
    return ranks - 1


#def cal_pearson_r(x, y, axis=0):
#    """calculate Pearson correlation coefficient along an axis."""
#    x = x.copy()  # 关键的步骤, 如果不进行会导致对数据进行inplace的修改, 最终nan充满整个数组.
#    y = y.copy()
#    nanmask = (np.isnan(x) | np.isnan(y))  # make x and y have the same nan values
#    x[nanmask] = np.nan
#    y[nanmask] = np.nan
#    x = x - np.nanmean(x, axis=axis, keepdims=True)
#    y = y - np.nanmean(y, axis=axis, keepdims=True)
#    result = np.nansum(x * y, axis) / np.sqrt(np.nansum(x ** 2, axis) * np.nansum(y ** 2, axis))
#    return result


def cal_pearson_r(x, y, axis=0):
    #import pdb; pdb.set_trace()
    #raise Exception('bug')
    xy = np.hstack((x, y))
    isnan = np.isnan(xy).any(axis=1)
    _x = x[np.ix_(~isnan)]
    _y = y[np.ix_(~isnan)]
    if _x.shape[0] < 2:
        return np.nan
    else:
        _x = _x - np.mean(_x, axis=axis, keepdims=True)
        _y = _y - np.mean(_y, axis=axis, keepdims=True)
        if np.allclose(_x, 0) or np.allclose(_y, 0):
            return 0.0
        else:
            res = np.sum(_x*_y) / np.sqrt(np.sum(_x**2, axis)) / np.sqrt(np.sum(_y**2, axis))
            return res[0]


def cal_cov(x, y, axis=0):
    """calculate covariance along an axis."""
    x = x.copy()  # 关键的步骤, 如果不进行会导致对数据进行inplace的修改, 最终nan充满整个数组.
    y = y.copy()
    nanmask = (np.isnan(x) | np.isnan(y))  # make x and y have the same nan values
    x[nanmask] = np.nan
    y[nanmask] = np.nan
    x = x - np.nanmean(x, axis=axis, keepdims=True)
    y = y - np.nanmean(y, axis=axis, keepdims=True)
    result = np.nansum(x * y, axis) / (~nanmask).sum(axis, keepdims=True)
    return result


#def load_data_2d(fields, f_load_data):
#    """读取数据. f_load_data中已经包含了start_date的信息. """
#    data = dict()
#    for field in fields:
#        field_ = field.split('.')[-1]
#        data[field_] = f_load_data(field)
#    return data

def load_data_2d(fields, f_load_data):
    """读取数据. f_load_data中已经包含了start_date的信息. """
    data = dict()
    for field in fields:
        data[field.replace('.', '_')] = f_load_data(field).to_numpy()
    return data

def load_tradedate(field, f_load_data):
    return f_load_data(field).index

def alpha_to_weights(alpha):
    """归一化. 最终截面绝对值和为2. """
    alpha = alpha - np.nanmean(alpha, axis=1, keepdims=True)
    mask_pos = (alpha > 0)
    mask_neg = (alpha < 0)
    alpha_pos = imposter(alpha)
    alpha_pos[mask_pos] = alpha[mask_pos]
    alpha_pos = alpha_pos / np.nansum(alpha_pos, 1, keepdims=True)
    alpha_neg = imposter(alpha)
    alpha_neg[mask_neg] = alpha[mask_neg]
    alpha_neg = -alpha_neg / np.nansum(alpha_neg, 1, keepdims=True)
    alpha[mask_pos] = alpha_pos[mask_pos]
    alpha[mask_neg] = alpha_neg[mask_neg]
    return alpha


# ENDCOPY #  以下为提交时不需要的函数


# 修改 ###################################################################

# 可以在这里添加不应该或不需要被COPY的自定义函数(如可能导致cython编译出现问题)

# 修改 ###################################################################


def get_eval_info(expr, dict_operators, dict_data_eval_info):
    """
    tries to get EvalInfo of expr's root node.
    if legal, returns EvalInfo of root node; otherwise, raises IllegalSyntax exception.
    原来写的check_syntax函数代码习惯比较糟糕, 导致出现了各种问题. 现在写一个更规范的版本.
    我们在这里尽量规避使用eval()函数. 事实上, 任何情况下都应避免使用该函数. 此后计算阿尔法值的代码可能也要改.
    TODO: 可能需要更改计算阿尔法值的代码
    取而代之, 利用该list深度优先遍历的特性, 采用递归的方法检查是否存在句法错误.
    首先从根节点开始, 计算它的所有子节点处的EvalInfo, 随后检查该节点处的输入是否符合算子的句法.
    而子节点处的EvalInfo用类似方法计算, 直到某个子节点不再有任何入度(arity, 每一个元素都有这个属性),
    此时返回的是该terminal的EvalInfo(如果是数据则有显示定义, 如果是ephemeral则可以返回任何东西, 因为不参与任何运算).
    至于如何检查句法, 我们通过传入的operators找到节点名字对应的算子, 直接调用该算子.
    调用算子过程中可能raise IllegalSyntax错误, 则会向外不断抛出, 需要接住(如check_syntax中的处理)
    TODO: 这个写法涉及一些重复计算, 可能效率较低
    :param expr: list, holds tree expression from DFS
    """
    if expr[0].arity > 0:  # primitive
        eval_info_of_subtrees = []
        begin = 1
        while True:
            # 寻找子树. 这部分代码来自gp.PrimitiveTree的searchSubtree方法
            end = begin + 1
            total = expr[begin].arity
            while total > 0:
                total += expr[end].arity - 1
                end += 1
            # 上述循环结束. 此时[begin, end)是子树的区间. end刚好停在下一个子树的开始, 或者在数组末尾.
            eval_info_of_subtrees.append(get_eval_info(expr[begin: end], dict_operators, dict_data_eval_info))
            begin = end
            if end == len(expr):  # 已经进行到列表的末尾
                break
        f = dict_operators[expr[0].name][0]
        return f(*eval_info_of_subtrees)

    else:  # terminal, could be data or ephemeral
        if expr[0].ret == Ephemeral:
            return expr[0].value  # Ephemeral则返回它自己的值, 因为有些算子(例如SIGNED_POWER)需要用到它计算unit
        else:  # data
            return dict_data_eval_info[expr[0].value]  # .value returns e.g. 'OPEN', while .name returns 'ARG0'


def check_syntax(expr, dict_operators, dict_data_eval_info):
    """检查一个输入列表expr的句法是否正确."""
    try:
        eval_info = get_eval_info(expr, dict_operators, dict_data_eval_info)
        return True
    except IllegalSyntex:
        return False


def compare_subtree(expr1, expr2, dict_operators, dict_data_eval_info):
    """
    We check whether the return type, shape, and units of two subtrees are the same, before performing crossover or mutation.
    """
    if expr1[0].ret != expr2[0].ret: # check return type
        return False
    eval_info_1 = get_eval_info(expr1, dict_operators, dict_data_eval_info)
    eval_info_2 = get_eval_info(expr2, dict_operators, dict_data_eval_info)
    # check shape and unit
    try:
        if eval_info_1.shape != eval_info_2.shape or check_same_unit(eval_info_1.unit, eval_info_2.unit) == False:
            return False
        return True
    except AttributeError:
        return False


def find_primitive(pset, return_type, name):
    """find a primitive in pset by name"""
    for op in pset.primitives[return_type]:
        if op.name == name:
            return op
    print("Primitive not found!")
    return None


def find_terminal(pset, return_type, value=None):
    """find a terminal in pset by name"""
    for op in pset.terminals[return_type]:
        # 这里用了or的short-circuiting. Ephemeral类型没有name或value属性, 故先判断是否为Ephemeral; 若不是, 再比较value.
        if return_type == Ephemeral or op.value == value:
            if inspect.isclass(op):  # Ephemeral类需要实例化. 其他算子直接就是函数了.
                return op()
            else:
                return op
    print("Terminal not found!")
    return None


def cal_pool_corr(pnl, pnl_pool):
    """
    计算一个阿尔法与pool的pearson最大相关系数. 下面n_days必须相同, 且为与asim一致, 必须都为500天, 且时间戳需对齐.
    :param pnl: shape = (n_days,)的ndarray
    :param pnl_pool: shape = (n_days, n_alphas)的ndarray
    :param axis: 沿哪个轴计算
    """
    # 用np.broadcast_to比用repeat快一点
    maxcorr = np.max(cal_pearson_r(np.broadcast_to(pnl[:, np.newaxis], pnl_pool.shape), pnl_pool, axis=0))
    return maxcorr

def maxcorr_with_pool(pnl, pnl_pool):
    res = []
    x = pnl.reshape(len(pnl), 1)
    if len(pnl_pool.shape) > 1:
        for i in range(pnl_pool.shape[1]):
            y = pnl_pool[:, i].reshape(len(pnl), 1)
            res.append(cal_pearson_r(x, y))
        return max(res)
    else:
        return cal_pearson_r(x, pnl_pool.reshape(len(pnl), 1))


def expr_to_str(expr):
    return str(gp.PrimitiveTree(expr))


def cal_ic_series(alpha, future_returns, ranked=True):
    """
    calculate time series of information coefficient
    """
    if ranked:  # calculate rank IC
        alpha = rank(alpha, axis=1)
        future_returns = rank(future_returns, axis=1)
    ic_series = cal_pearson_r(alpha, future_returns, axis=1)
    return ic_series


def cal_pnl_series(alpha, today_returns):
    """
    计算pnl序列, 其实只是收益率序列, 因为只差本金(常数).
    :param alpha: 一个2维ndarray. 其必须满足值可以解释为权重, 例如每天的正数部分和负数部分和都为1.
    :param today_returns: 也是2d数组, 其与alpha对齐的时间代表当日的收益率(alpha本身是用delay的数据计算的)
    :return: 返回的是当日的策略收益率
    """
    return np.nansum(ts_delay(alpha, 1) * today_returns, 1) / 2  # 有多空, 收益率要除以2. 注意若全nan则当日为0.


# basic functions used in mutation and crossover
def exception_printer(k, name):
    pass  # 测试时打印太多了, 先关掉
    # if k == 0:
    #     print(f"=====Catch exception in function {name}=====")


def compare(ind1, ind2, cxpoint1, cxpoint2, dict_operators, dict_data_eval_info):
    tree1, tree2 = gp.PrimitiveTree(ind1), gp.PrimitiveTree(ind2)
    root1, root2 = ind1[cxpoint1], ind2[cxpoint2]
    # Eliminate the situation while leaf(terminal) is selected as "subtree".
    if(type(root1) == 'deap.gp.Terminal' and type(root2) == 'deap.gp.Terminal'):
        return (False, 0, 0)
    slice1, slice2 = tree1.searchSubtree(cxpoint1), tree2.searchSubtree(cxpoint2)
    sublst1, sublst2 = ind1[slice1], ind2[slice2]
    # Only consider crossover of subtree when its height is greater than 1.
    if compare_subtree(sublst1, sublst2, dict_operators, dict_data_eval_info) and \
            gp.PrimitiveTree(sublst1).height >= 1 and gp.PrimitiveTree(sublst2).height >= 1:
        return (True, len(sublst1), len(sublst2))
    else:
        return (False, 0, 0)


def pw(text, log):
    """print and write. note that write() does not append newline automatically, and print() is adjusted accordingly."""
    print(text, end='')
    log.write(text)


def get_population_statistics(population):
    """获取一个种群的统计量. 注意所有个体必须都有fitness."""
    fitness_list = np.sort(np.array([individual.fitness for individual in population if not (np.isinf(individual.fitness) or np.isnan(individual.fitness))]))
    text = f'population size: {len(population)}, valid individual size: {len(fitness_list)}\n'
    if len(fitness_list)==0: return text
    statistics = [
        np.mean(fitness_list),
        np.std(fitness_list),
        stats.skew(fitness_list),
        stats.kurtosis(fitness_list),
        fitness_list[0],
        fitness_list[int(len(fitness_list) * 0.25)],
        fitness_list[int(len(fitness_list) * 0.5)],
        fitness_list[int(len(fitness_list) * 0.75)],
        fitness_list[-1]
    ]
    text += 'MEAN:{:4.2f}    STD :{:4.2f}    SKEW:{:4.2f}    KURT:{:4.2f}\n' \
            'MIN :{:4.2f}    QT25:{:4.2f}    QT50:{:4.2f}    QT75:{:4.2f}    MAX :{:4.2f}\n'.format(*statistics)
    return text

def table_population_statistics(population, title):
    """获取一个种群的统计量. 注意所有个体必须都有fitness."""
    fitness_list = [(x.fitness, abs(x.fitness_raw)) for x in population if not (np.isinf(x.fitness) or np.isnan(x.fitness))]
    fitness_list.sort(key=lambda x:x[0])
    fitness_list = np.array(fitness_list)
    ttc, vac = len(population), len(fitness_list)
    if vac > 0:
        q25, q50, q75 = fitness_list[int(vac*0.25), :], fitness_list[int(vac*0.5), :], fitness_list[int(vac*0.75), :]
        mm, ss, mi, ma = np.mean(fitness_list, axis=0), np.std(fitness_list, axis=0), fitness_list[0, :], fitness_list[-1, :]
    else:
        q25, q50, q75 = ([np.nan]*2, ) * 3
        mm, ss, ma, mi= ([np.nan]*2, ) * 4
    table = PrettyTable()
    table.title = title
    table.field_names = ['No.', 'Stats', 'Value1', 'Value2']
    table.add_row(['0', 'mean', f'{mm[0]:.2f}', f'{mm[1]:.2f}'])
    table.add_row(['1', 'std', f'{ss[0]:.2f}', f'{ss[1]:.2f}'])
    table.add_row(['2', 'max', f'{ma[0]:.2f}', f'{ma[1]:.2f}'])
    table.add_row(['3', 'Q75', f'{q75[0]:.2f}', f'{q75[1]:.2f}'])
    table.add_row(['4', 'Q50', f'{q50[0]:.2f}', f'{q50[1]:.2f}'])
    table.add_row(['5', 'Q25', f'{q25[0]:.2f}', f'{q25[1]:.2f}'])
    table.add_row(['6', 'min', f'{mi[0]:.2f}', f'{mi[1]:.2f}'])
    table.add_row(['7', 'ttCount', f'{ttc:.0f}', f'{ttc:.0f}'])
    table.add_row(['8', 'vaCount', f'{vac:.0f}', f'{vac:.0f}'])
    return table


def cal_frequent_subtrees(population, hof_num=10):
    '''
    Calculate frequently appeared subtrees (with certain operations and data).
    The output will be used in subtreeMT function. Computationally inexpensive.
    '''
    all_count = []
    for individual in population:
        ind1 = individual.expr
        tree1 = gp.PrimitiveTree(ind1)
        size = len(ind1)
        for cxpoint1 in range(size):
            t1 = ind1[tree1.searchSubtree(cxpoint1)]
            subtree1 = gp.PrimitiveTree(t1)
            if subtree1.height > 1:
                all_count.append(t1)
    result = pd.value_counts(all_count)
    return result.index[0:hof_num]


def cal_frequent_structure(population, hof_num=10):
    '''
    Calculate frequently appeared structures (with consecutive certain primitives, but the auxiliary data could be arbitrary.)

    '''
    all_count = []
    for individual in population:
        this_count = []
        ind1 = individual.expr
        size = len(ind1)
        for cxpoint1 in range(size):
            if isinstance(ind1[cxpoint1], deap.gp.Primitive):
                this_count.append(ind1[cxpoint1])
            else:
                if (len(this_count) > 2):
                    all_count.append(this_count)
                    this_count = []
    result = pd.value_counts(all_count)
    return result.index[0:hof_num]


def f_load_data_io(field, data_folder, start_date, end_date):
    """用io的函数读取数据. 这个函数仅用于load_data_2d的f_load_data参数. 另一种f_load_data仅在submit时才定义, 使用self.get_data"""
    # 尝试读为2d, 若失败则读为1d数据
    data_field = io.read2d_from_asimcache(os.path.join(data_folder, field))[1].to_dataframe()
    if data_field is None:
        data_field = io.read1d_from_asimcache(os.path.join(data_folder, field))[1].to_dataframe()
    if data_field is None:
        raise AttributeError(f'{field} not found')
    data_field = data_field.loc[start_date: end_date]
    return data_field


def safe_regression(X:np.ndarray, Y:np.ndarray):
    y, x = Y.reshape(len(Y), 1), X.reshape(len(X), 1)
    yx = np.hstack((y, x))
    nanspl = np.isnan(yx).any(axis=1)
    _y, _x = y[~nanspl, :], x[~nanspl, :]
    if _y.shape[0]==0: return (-1, None, None)
    allzero = (_x==0).all(axis=0)
    _x = _x[:, ~allzero]
    if _x.shape[1]==0: return (-1, None, None)
    _y = _y.reshape(len(_y), 1)
    coef = np.linalg.lstsq(_x, _y, rcond=None)[0]
    eps = np.full(fill_value=np.nan, shape=(Y.shape[0], ), dtype=float)
    eps[~nanspl] = (_y - np.dot(_x, coef))
    return (0, eps, coef)