#!/usr/bin/python3

from abc import ABC, abstractmethod
import numpy as np

# 加载自定义包
from base_classes_funcs import *

class EarlyStopBase(ABC):
    def __init__(self, patience:int):
        self.__patience = patience
    @property
    def patience(self):
        return self.__patience
    @abstractmethod
    def is_early_stop(self):
        raise NotImplemented
    @abstractmethod
    def rollback(self):
        raise NotImplemented


class EarlyStopCont(EarlyStopBase):
    def __init__(self, patience:int, backto:int):
        super(EarlyStopCont, self).__init__(patience)
        self.__curr_patience = patience
        self.__save_population = []
        self.__backto = backto

    @property
    def backto(self):
        return self.__backto

    def is_early_stop(self, population:list(), data_valid, today_returns, dict_data_eval_info, pset, f_evaluate, decay) -> bool:
        if super().patience < 0: 
            return False
        self.__save_population.append(population)
        data_list = [data_valid[k] for k in dict_data_eval_info.keys()]
        thres = np.sort([abs(x.fitness_raw) for x in population])[-int(len(population)*0.2)]
        topN  = [x for x in population if abs(x.fitness_raw) >= thres]
        fitness1, fitness2 = [], []
        for indi in topN:
            fitness1.append(abs(indi.fitness_raw))
            fitness2.append(abs(f_evaluate(indi.expr, data_list, today_returns, pset)[1]))
        # 连续patience次的valid夏普不如测试集，早停
        if np.nanmean(fitness2) < decay * np.nanmean(fitness1):
            self.__curr_patience -= 1
        else:
            self.__curr_patience = super().patience
            self.__save_population = []
        if self.__curr_patience == 0:
            return True
        else:
            return False

    def rollback(self, population:list()=[], t:int=0):
        if t==0:
            if super().patience >= 0:
                idx = max(0, self.__backto)+1
                idx = min(super().patience+1, idx)
                return self.__save_population[-idx]
            else:
                return population
        else:
            return self.__backto

