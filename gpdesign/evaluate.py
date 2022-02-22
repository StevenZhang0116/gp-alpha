#!/usr/bin/python3

"""
计算适应度的各种指标
"""

from abc import ABC, abstractmethod

# 加载自定义包
from base_classes_funcs import *

def parser_alpha_value(expr, data_list, pset):
    #import pdb; pdb.set_trace()
    f_alpha = gp.compile(expr=gp.PrimitiveTree(expr), pset=pset)
    return f_alpha(*data_list)


class AlphaEvalBase(ABC):
    days_in_a_year = 250
    days_in_a_year2= 15.81
    def __init__(self, ret_stock:np.ndarray, ret_index:np.ndarray, alpha:np.ndarray=None):
        self.__retstock = ret_stock
        self.__retindex = ret_index
        self.__alpha = alpha
    @property
    def ret_stock(self):
        return self.__retstock
    @property
    def ret_index(self):
        return self.__retindex
    @property
    def alphavalue(self):
        return self.__alpha

    def parser_alpha_value(self, expr, data_list, pset):
        self.__alpha = parser_alpha_value(expr, data_list, pset)

    @alphavalue.setter
    def alphavalue(self, *args):
        raise AttributeError('alphavalue setting must call parser_alpha_value method')

    @abstractmethod
    def run(self):
        raise NotImplemented

class AlphaEvalSimpleSharpe(AlphaEvalBase):
    def __init__(self, ret_stock:np.ndarray, ret_index:np.ndarray, alpha:np.ndarray=None):
        super(AlphaEvalSimpleSharpe, self).__init__(ret_stock, ret_index, alpha)
        self.__net_pnl = None
    def __calc_net_pnl(self, alpha=None):
        if super().alphavalue is not None:
            return np.nansum(super().alphavalue * super().ret_stock, axis=1)
        elif alpha is not None:
            return np.nansum(alpha * super().ret_stock, axis=1)
        else:
            return None
    def run(self, expr, data_list, pset):
        # alpha
        if super().alphavalue is None:
            super().parser_alpha_value(expr, data_list, pset)
        self.__net_pnl = self.__calc_net_pnl()
        if self.__net_pnl is not None:
            return np.nanmean(self.__net_pnl) / np.nanstd(self.__net_pnl) * super().days_in_a_year2
        else:
            return np.nan
    @property
    def net_pnl(self):
        return self.__net_pnl


class AlphaEvalSimpleTurnover(AlphaEvalBase):
    def __init__(self, ret_stock:np.ndarray, ret_index:np.ndarray, alpha:np.ndarray=None):
        super(AlphaEvalSimpleTurnover, self).__init__(ret_stock, ret_index, alpha)
    def run(self, expr, data_list, pset):
        if super().alphavalue is None:
            super().parser_alpha_value(expr, data_list, pset)
        alpha = np.where(mask(super().alphavalue), super().alphavalue, 0.0)
        return np.mean(np.sum(abs(alpha[1:, :] - alpha[:-1, :]), axis=1))

        
class AlphaEvalSimpeAnnret(AlphaEvalBase):
    def __init__(self, ret_stock, ret_index, alpha):
        super(AlphaEvalSimpeAnnret, self).__init__(ret_stock, ret_index, alpha)
        self.__net_pnl = None
    def __calc_net_pnl(self, alpha=None):
        if super().alphavalue is not None:
            return np.nansum(super().alphavalue * super().ret_stock, axis=1)
        elif alpha is not None:
            return np.nansum(alpha * super().ret_stock, axis=1)
        else:
            return None
    def run(self, expr, data_list, pset):
        if super().alphavalue is None:
            super().parser_alpha_value(expr, data_list, pset)
        self.__net_pnl = self.__calc_net_pnl()
        if self.__net_pnl is not None:
            return np.nanmean(self.__net_pnl) * super().days_in_a_year
        else:
            return np.nan
    @property
    def net_pnl(self):
        return self.__net_pnl


def eval_fitness_sharpe(expr, data_list, today_returns, pset):
    alpha = parser_alpha_value(expr, data_list, pset)
    simple_shp = AlphaEvalSimpleSharpe(today_returns, None, alpha)
    sharpe = simple_shp.run(None, None, None)
    netpnl = simple_shp.net_pnl * (-1, 1)[sharpe>0]
    fitness= np.nan
    if sharpe and (~np.isnan(sharpe)):
        fitness = abs(sharpe)
    outstats = {
        'pnl': netpnl,
        'sharpe': sharpe
    }
    return fitness, sharpe, outstats



def eval_fitness_stats(expr, data_list, today_returns, pset, pnl_pool):
    #import pdb; pdb.set_trace()
    alpha = parser_alpha_value(expr, data_list, pset)
    simple_shp = AlphaEvalSimpleSharpe(today_returns, None, alpha)
    simple_tvr = AlphaEvalSimpleTurnover(today_returns, None, alpha)
    sharpe   = simple_shp.run(None, None, None)
    turnover = simple_tvr.run(None, None, None)
    netpnl   = simple_shp.net_pnl * (-1, 1)[sharpe>0]
    datelen  = min(pnl_pool.shape[0], netpnl.shape[0])
    maxcorr  = maxcorr_with_pool(netpnl[-datelen:].reshape(datelen,1), pnl_pool[-datelen:, :]) # 通过数据保证 netpnl 和 pnl_pool对齐. Pnl_pool是对齐的
    avgcorr  = cal_pearson_r(netpnl[-datelen:].reshape(datelen,1), np.nanmean(pnl_pool[-datelen:, :], axis=1).reshape(datelen,1))
    fitness  = np.nan
    if sharpe and turnover and maxcorr and avgcorr:
        if not any(np.isnan([sharpe, turnover, maxcorr, avgcorr])):
            fitness = abs(sharpe)
            fitness*= (2.0 - min(2.0, turnover))**5
            fitness*= max(0.0, (1.0 - avgcorr))
            maxcorr = max(-1.0, min(1.0, maxcorr))
            if maxcorr > 0.5:
                fitness /= (maxcorr*2.0)**2
    # reg_complexity = 100 / len(expr)
    outstats = {
        'pnl': netpnl,
        'turnover': turnover,
        'sharpe': sharpe,
        'maxcorr': maxcorr,
        'avgcorr': avgcorr
    }
    return fitness, sharpe, outstats


def eval_fitness_minfo(expr, minibatch_data_list, minibatch_future_returns, toolbox):
    """
    用互信息作为fitness.
    """
    pass

