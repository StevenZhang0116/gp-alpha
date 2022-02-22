#!/usr/bin/python3
#vim: set tabstop=4:softtabstop=4:shiftwidth=4:noexpandtab
"""
Created By Jiayi Xie
March 2020

io库，依赖于 asim_arena_lib 库函数, 和 asim python 封装, 和数据版本

"""


import os, sys, platform
sys.path.append('/work/asim/release/4.3.3/bin/release')

import numpy, pandas

class CubeData:
    """ 封装 asim cubedata 的结构体"""
    def __init__(self):
        self.datatag = ''
        self.data = None
        self.startidx = None
        self.endidx = None
        self.startti = None
        self.endti = None
        self.dates = None
        self.tickers = None
    
    def to_dataframe(self):
        if self.data is not None:
            res = pandas.DataFrame(self.data.reshape(-1, self.data.shape[2]))
            if self.tickers is not None :
                res.columns = self.tickers.tolist()
            if self.dates is not None:
                res.index = [(d, t) for d in self.dates.tolist()[self.startidx:self.endidx] for t in range(self.startti, self.endti)]
                res.index.name = ('TrdDate', 'BarTime')
            return res
        else:
            return None

class IndexXminData:
    """ 封装 asim index xmindata 的结构体 """
    def __init__(self):
        self.datatag = ''
        self.data = None
        self.startidx = None
        self.endidx = None
        self.startti = None
        self.endti = None
        self.dates = None
        self.tickers = None
    
    def to_dataframe(self):
        if self.data is not None:
            res = pandas.DataFrame(self.data[self.startidx:self.endidx])
            res.columns = list(range(self.startti, self.endti))
            if self.dates is not None:
                res.index = self.dates.tolist()[self.startidx:self.endidx]
                res.index.name = 'TrdDate'
            return res
        else:
            return None

class VectorData:
    """ 封装 asim vectordata 的结构体，
    只支持维数为日期的数据 """
    def __init__(self):
        self.datatag = ''
        self.data = None
        self.startidx = None
        self.endidx = None
        self.dates = None
    
    def to_dataframe(self):
        if self.data is not None:
            res = pandas.Series(self.data[self.startidx:self.endidx])
            if self.datatag != '':
                res.name = self.datatag
            if self.dates is not None:
                res.index = self.dates.tolist()[self.startidx:self.endidx]
                res.index.name = 'TrdDate'
            return res
        else:
            return None

class MatrixData:
    """ 封装 asim matrixdata 的结构体"""
    def __init__(self):
        self.datatag = ''
        self.data = None
        self.startidx = None
        self.endidx = None
        self.dates = None
        self.tickers = None
    
    def to_dataframe(self):
        if self.data is not None:
            res = pandas.DataFrame(self.data[self.startidx:self.endidx])
            if self.tickers is not None :
                res.columns = self.tickers.tolist()
            if self.dates is not None:
                res.index = self.dates.tolist()[self.startidx:self.endidx]
                res.index.name = 'TrdDate'
            return res
        else:
            return None

def read_indexXmin_from_asimcache(f:str) ->(int, IndexXminData):
    fn = os.path.basename(f)
    ix = IndexXminData()
    err = 'failed to read cachedata %s'%os.path.basename(f)
    try:
        import libDataCache as dlib
        cache_n = dlib.DataCache(os.path.dirname(f))
        a_n = cache_n.get_data(os.path.basename(f))
        if 'Matrix' not in a_n[1].datatype:
            return 'no index xmin data found', ix
        err, ix.startidx, ix.endidx = 0, 0, cache_n.diMax
        ix.startti, ix.endti = 0, a_n[1].datadim[1]
        ix.data, ix.datatag = a_n[0], os.path.basename(f)
        ix.dates, ix.tickers = cache_n.Dates, ''
    except Exception as e:
        err = ':'.join([err, str(e)])
    return err, ix


def read3d_from_asimcache(f:str) ->(int, CubeData):
    fn  = os.path.basename(f)
    cb = CubeData()
    err = 'failed to read cachedata %s'%os.path.basename(f)
    try:
        import libDataCache as dlib
        cache_n = dlib.DataCache(os.path.dirname(f))
        a_n = cache_n.get_data(os.path.basename(f))
        if 'Cube' not in a_n[1].datatype:
            return 'no cube data found', cubedate
        err, cb.startidx, cb.endidx = 0, a_n[0].startidx, cache_n.diMax
        cb.startti, cb.endti = 0, a_n[0].cubedata.shape[1]
        cb.data, cb.datatag = a_n[0].cubedata, os.path.basename(f)
        cb.dates, cb.tickers = cache_n.Dates, cache_n.Tickers
    except Exception as e:
        err = ':'.join([err, str(e)])
    return err, cb

def read2d_from_asimcache(f:str) ->(int, MatrixData):
    fn  = os.path.basename(f)
    ma = MatrixData()
    err = 'failed to read cachedata %s'%os.path.basename(f)
    try:
        import libDataCache as dlib
        cache_n = dlib.DataCache(os.path.dirname(f))
        a_n = cache_n.get_data(os.path.basename(f))
        if 'Matrix' not in a_n[1].datatype:
            return 'no matrix data found', ma
        err, ma.startidx, ma.endidx = 0, 0, cache_n.diMax
        ma.data, ma.datatag = a_n[0], os.path.basename(f)
        ma.dates, ma.tickers = cache_n.Dates, cache_n.Tickers
    except Exception as e:
        err = ':'.join([err, str(e)])
    return err, ma

def read1d_from_asimcache(f:str) ->(int, VectorData):
    fn = os.path.basename(f)
    vt = VectorData()
    err = 'failed to read cachedata %s'%os.path.basename(f)
    try:
        import libDataCache as dlib
        cache_n = dlib.DataCache(os.path.dirname(f))
        a_n = cache_n.get_data(os.path.basename(f))
        if 'Vector' not in a_n[1].datatype:
            return 'no vector data found', vt
        err, vt.startidx, vt.endidx = 0, 0, cache_n.diMax
        vt.data, vt.datatag = a_n[0], os.path.basename(f)
        vt.dates = cache_n.Dates
    except Exception as e:
        err = ':'.join([err, str(e)])
    return err, vt



def read_csv(filename:str, has_header:bool, has_index:bool, **kwargs) -> (int, pandas.DataFrame):
    _header = 0 if has_header else None
    _index = 0 if has_index else None
    try:
        data = pandas.read_csv(filename, header=_header, index_col=_index, **kwargs)
    except Exception as e:
        print(str(e))
        return -1, None
    return 0, data

def read_excel(filename:str, has_header:bool, has_index:bool, **kwargs) -> (int, pandas.DataFrame):
    _header = 0 if has_header else None
    _index = 0 if has_index else None
    try:
        data = pandas.read_excel(filename, header=_header, index_col=_index, **kwargs)
    except Exception as e:
        print(str(e))
        return -1, None
    return 0, data

def read_txt(filename:str, delimiter:str, newlinedel:str) -> (int, pandas.DataFrame):
    try:
        inf = open(filename, 'r')
        content = inf.readlines()
        inf.close()
        data = []
        for l in content:
            data.append(l.strip(newlinedel).split(delimiter))
        data = pandas.DataFrame(data)
    except Exception as e:
        print(str(e))
        return -1, None
    return 0, data

def read_txt2(filename:str, delimiter:str, newlinedel:str) -> (int, list):
    try:
        inf = open(filename, 'r')
        content = inf.readlines()
        inf.close()
        data = []
        for l in content:
            data.append(l.strip(newlinedel).split(delimiter))
    except Exception as e:
        print(str(e))
        return -1, None
    return 0, data

def write_csv(data:pandas.DataFrame, filename:str, has_header:bool, has_index:bool, **kwargs) -> int:
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        data.to_csv(filename, header=has_header, index=has_index, **kwargs)
    except Exception as e:
        print(str(e))
        return -1
    return 0

def write_txt(data:list, filename:str, newlinedel='\n') -> int:
    try:
        outf = open(filename, 'w')
        for l in data:
            outf.write(l)
            outf.write(newlinedel)
        outf.close()
    except Exception as e:
        print(str(e))
        return -1
    return 0

def write_excel(data:pandas.DataFrame, filename:str, has_header:bool, has_index:bool, **kwargs) -> int:
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        data.to_excel(filename, header=has_header, index=has_index, **kwargs)
    except Exception as e:
        print(str(e))
        return -1
    return 0
