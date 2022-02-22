#!/usr/bin/python3
"""
这个文件定义了gpAlpha使用的数据, 和相关的预处理. 数据可以是daily或xmin, 不支持offset.
对于每个数据字段, 需要定义其量纲.
需要修改的部分已经标注.
"""
from abc import ABC, abstractmethod
import numpy as np, deap
from deap import gp
from base_classes_funcs import Array2d, Array2dValid, Array2dNeutralise, Ephemeral, EvalInfo, FinalResult, ts_delay, load_data_2d, load_tradedate

# COPY #

class DataConstrBase(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def load_data(self):
        raise NotImplemented
    @abstractmethod
    def register_data(self):
        raise NotImplemented
    @abstractmethod
    def set_future_return(self):
        raise NotImplemented
    @abstractmethod
    def set_valid_universe(self):
        raise NotImplemented


class DataConstr(DataConstrBase):
    def __init__(self, subUniv):
        super(DataConstr, self).__init__()
        self.__subUniv = subUniv+'.valid'
        self.__data = dict()
        self.__datatemp = dict()
        self.__dataname = set()
        self.__dates = []
    @property
    def dates(self):
        return self.__dates
    def load_data(self, delay, train_size, valid_size, f_load_data) ->dict():
        data = load_data_2d(
                ('BaseData.open', 'BaseData.close', 'BaseData.high', 'BaseData.low', 'BaseData.avg_price',
                 'BaseData.tvrvolume', 'BaseData.tvrvalue', 'BaseData.mkt_cap', 'BaseData.return',
                 'SWIndustry.L1', self.__subUniv, 'UnivST.valid'),
                f_load_data)
        self.__dates = load_tradedate('BaseData.open', f_load_data)
        keys_lower = list(data.keys())  # 注意keys在循环中会有变化
        for key_lower in keys_lower:  # 将数据字段全部改为大写, 防止与关键字重名, 如return
            K = key_lower.upper()
            data[K] = data.pop(key_lower)
            self.__dataname.add(K)
        for key in data.keys():
            if key.endswith(('VALID', 'L1', 'L2', 'L3', 'L4')): continue  # valid不应该取nan, 如果取了就破坏了np.bool类型
            data[key] = data[key].astype(float)
            if key.endswith(('OPEN', 'CLOSE', 'HIGH', 'LOW', 'AVG_PRICE', 'TVRVOLUME', 'TVRVALUE', 'MKT_CAP')):
                data[key] = np.where(data[key]>0, data[key], np.nan)
        
        self.__datatemp = data.copy()
        self.__data = {k: ts_delay(v, (delay, 0)[k.endswith(('VALID', ))], axis=0) for k, v in data.items()}

        self.set_future_return()
        self.set_valid_universe()
        self.set_dataneutral()

        if valid_size > 0 and train_size + valid_size <= self.__data['VALID'].shape[0]:
            slc_train = (slice(None, -valid_size),)
            slc_valid = (slice(-valid_size, None),)
        else:
            slc_train = (slice(None, None), )
            slc_valid = (slice(None, None), )
        return self.__data, slc_train, slc_valid


    def set_future_return(self):
        self.__data['TODAY_RETURNS'] = ts_delay(self.__datatemp['BASEDATA_RETURN'], -1, axis=0)
        self.__dataname.add('TODAY_RETURNS')

    def set_valid_universe(self):
        subuniv = self.__datatemp[self.__subUniv.upper().replace('.', '_')]
        fil = (~self.__datatemp['UNIVST_VALID']) & (self.__datatemp['BASEDATA_OPEN']>0) & (self.__datatemp['BASEDATA_HIGH']-self.__datatemp['BASEDATA_LOW'] > 0.01)
        subuniv = subuniv * fil # fil 用了 delay0 数据，但只用来参与 pnl 的计算，可以认为没有使用未来数据
        self.__data['VALID'] = np.where(np.isnan(subuniv), 0, subuniv).astype(bool)
        self.__dataname.add('VALID')

    def set_dataneutral(self):
        # 为了兼容旧代码, wtf
        self.__data['X_NEUTRALISE'] = self.__data['BASEDATA_RETURN'][:, :, np.newaxis]
        self.__dataname.add('X_NEUTRALISE')
        
# ENDCOPY #


    def register_data(self, dict_operators, data_type):
        """这个函数仅在挖掘时会用到, 所以不进行COPY. """
        # 注意，只有被用于 alpha 计算的字段才能注册到 pset 中
        dict_data_eval_info = dict()
        for K in self.__dataname:
            if K in ('BASEDATA_OPEN', 'BASEDATA_CLOSE', 'BASEDATA_HIGH', 'BASEDATA_LOW', 'BASEDATA_AVG_PRICE'):
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([1, 0]), Array2d)
            elif K in ('BASEDATA_TVRVOLUME', ):
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([0, 1]), Array2d)
            elif K in ('BASEDATA_TVRVALUE', ):
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([1, 1]), Array2d)
            elif K in ('BASEDATA_RETURN', ):
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([0, 0]), Array2d)
            elif K.startswith(('UNIV', )) and K != 'VALID':
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([0, 0]), Array2dValid)
            elif K in ('X_NEUTRALISE', ):
                dict_data_eval_info[K] = EvalInfo(self.__data[K].shape, np.array([0, 0]), Array2dNeutralise)
        # register pset
        n_units = len(dict_data_eval_info['BASEDATA_OPEN'].unit)
        data_types = [x.ret_type for x in dict_data_eval_info.values()] # terminal 的 STGP 类型
        pset = gp.PrimitiveSetTyped("main", data_types, FinalResult)
        for v in dict_operators.values():  # add all operators
            pset.addPrimitive(*v)
        pset.addEphemeralConstant("eph", lambda: np.random.rand(), Ephemeral)  # this adds a func, not a single number
        kwargs_rename = {f'ARG{i}': x for i, x in enumerate(dict_data_eval_info.keys())}
        pset.renameArguments(**kwargs_rename)
        # 按量纲对数据分类.
        # 将ndarray转换为tuple
        units = dict()  # 一个由量纲元组作为键的字典
        for K, V in dict_data_eval_info.items():
            du = tuple([u for u in V.unit])
            units[du] = units.get(du, []) +[K]
        names_data_classified_by_units = list(units.values()) # e.g [['OPEN', 'CLOSE', 'HIGH', 'LOW', 'VWAP'], ['VOLUME'], ['RETURN', 'X_NEUTRALISE', 'VALID']]
        data_classified_by_units = []
        for x in names_data_classified_by_units: # 从names_data_classified_by_units中筛选出返回类型为data_type的, 然后取对应的对象
            # e.g  [['OPEN', 'CLOSE', 'HIGH', 'LOW', 'VWAP'], ['VOLUME'], ['RETURN']]
            data_classified_by_units.append([term for term in pset.terminals[data_type] if term.value in x])
        return dict_data_eval_info, n_units, pset, data_classified_by_units


