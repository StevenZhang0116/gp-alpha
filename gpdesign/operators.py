"""
阅读时建议折叠所有代码块, 例如PyCharm中操作为Ctrl+Shift+减号.

构建算子库和参数的遍历范围
添加自定义算子时, 请添加在COPY与ENDCOPY间, 否则生成的submit代码中会没有该算子.
若不想使用某个算子, 将添加dict_operators和dict_params键(若有)的代码注释即可.

20210925    init ver    copy from Zehua Yu / Zihan Zhang
20211002    OOD         jyxie
"""

from abc import ABC, abstractmethod
from functools import wraps
# 加载自定义包
from base_classes_funcs import *

# COPY #


class OPBase(ABC):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        self.__prob = prob
        self.__checksyntax = checksyntax
    @property
    def prob(self):
        return self.__prob
    @property
    def checksyntax(self):
        return self.__checksyntax
    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplemented
    @abstractmethod
    def checksyntax(self):
        raise NotImplemented


class ADD(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(ADD, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, b):
        return a + b
    def checksyntax(f):
        def decorated(self, a, b):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, b)
            else:
                if (a.shape==b.shape) and (check_same_unit(a.unit, b.unit)):
                    return EvalInfo(a.shape, a.unit, Array2d)
                else: raise IllegalSyntex
        return decorated

            
class SUB(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(SUB, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, b):
        return a - b
    def checksyntax(f):
        def decorated(self, a, b):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, b)
            else:
                if (a.shape==b.shape) and (check_same_unit(a.unit, b.unit)):
                    return EvalInfo(a.shape, a.unit, Array2d)
                else: raise IllegalSyntex
        return decorated


class MUL(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(MUL, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, b):
        return a * b
    def checksyntax(f):
        def decorated(self, a, b):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, b)
            else:
                if (a.shape==b.shape) :
                    return EvalInfo(a.shape, a.unit+b.unit, Array2d)
                else: raise IllegalSyntex
        return decorated


class DIV(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(DIV, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, b):
        return replace_inf(a / b)
    def checksyntax(f):
        def decorated(self, a, b):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, b)
            else:
                if (a.shape==b.shape) :
                    return EvalInfo(a.shape, a.unit-b.unit, Array2d)
                else: raise IllegalSyntex
        return decorated


class ABS(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(ABS, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a):
        return np.abs(a)
    def checksyntax(f):
        def decorated(self, a):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class RANK(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(RANK, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a):
        return rank(a, axis=1) / mask(a).sum(1, keepdims=True)
    def checksyntax(f):
        def decorated(self, a):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a)
            else:
                return EvalInfo(a.shape, np.zeros_like(a.unit), Array2d)
        return decorated


class SIGNED_POWER(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(SIGNED_POWER, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, p, rankfirst:bool=True):
        rk = RANK(checksyntax=False)
        if rankfirst:
            a = rk(a)
        if p==0: return np.full(fill_value=1.0, dtype=float, shape=a.shape)
        elif p==1: return a
        elif p>0 : np.sign(a) * (np.abs(a) ** p)
        else: replace_inf(1.0 / (np.sign(a) * (np.abs(a) ** (-p))))
    def checksyntax(f):
        def decorated(self, a, p):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a)
            else:
                return EvalInfo(a.shape, a.unit*p, Array2d)
        return decorated


class SIGNED_LOG(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(SIGNED_LOG, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a):
        return np.sign(a) * np.log(np.abs(a))
    def checksyntax(f):
        def decorated(self, a):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class TS_DELTA(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(TS_DELTA, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, w, type:int=0):
        if type==0:
            if w > 0 : return a - ts_delay(a, window=w, axis=0)
            elif w==0: return np.zeros_like(a)
            else: return imposter(a)
        elif type==1:
            if w > 0 : return replace_inf(a / ts_delay(a, window=w, axis=0))
            elif w==0: return np.zeros_like(a)
            else: return imposter(a)
    def checksyntax(f):
        def decorated(self, a, w):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, w)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class TS_MEAN(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(TS_MEAN, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, w):
        if w > 1:
            c = np.cumsum(mask(a), dtype=float)
            return np.append(a[:w], (c[w:] - c[:-w])/w)
        else:
            return a
    def checksyntax(f):
        def decorated(self, a, w):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, w)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class TS_STD(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(TS_STD, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, w):
        if w > 2:
            tsmean = TS_MEAN(checksyntax=False)
            res = tsmean(a**2, w) - tsmean(a, w)**2
            if res>0: return np.sqrt(res)
            else: return imposter(a)
        else:
            return imposter(a)
    def checksyntax(f):
        def decorated(self, a, w):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, w)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class FILTER_INVALID(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(FILTER_INVALID, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, v):
        return np.where(v, a, np.nan)
    def checksyntax(f):
        def decorated(self, a, v):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, v)
            else:
                if (a.shape==v.shape) :
                    return EvalInfo(a.shape, a.unit, Array2d)
                else: raise IllegalSyntex
        return decorated


class OP(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(OP, self).__init__(prob, checksyntax)    
    @checksyntax
    def __call__(self, a, v, *args, **kwargs):
        fl = FILTER_INVALID(checksyntax=False)
        a  = fl(a, v)
        a  = a - np.nanmean(a, axix=1)[:, np.newaxis]
        a  = a / abs(np.nansum(a, axis=1))[:, np.newaxis]
        return a
    def checksyntax(f):
        def decorated(self, a, v, *args, **kwargs):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, v, *args, **kwargs)
            else:
                if (a.shape==v.shape) :
                    return EvalInfo(a.shape, a.unit, FinalResult)
                else: raise IllegalSyntex
        return decorated


class TS_ROBUST(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(TS_ROBUST, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, w, t):
        tsmean = TS_MEAN(checksyntax=False)
        tsstd  = TS_STD(checksyntax=False)
        if t==0: return tsmean(a, w) * tsstd(a, w)
        elif t==1: return tsmean(a, w) / tsstd(a, w)
        elif t==2: return tsstd(a, w) / tsmean(a, w)
        elif t==3: return (a - tsmean(a, w)) / tsstd(a, w)
        elif t==4: return (a - tsmean(a, w)) * tsstd(a, w)
    def checksyntax(f):
        def decorated(self, a, w, t):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, w, t)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class NEU_LS(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(NEU_LS, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a):
        l = np.where(a>0, a, 0).sum(axis=1)[:, np.newaxis]
        s = np.where(a<0, a, 0).sum(axis=1)[:, np.newaxis]*(-1)
        fl = np.logical_and(l>0, s>0)
        res = imposter(a)
        res[a==0] = 0
        res = np.where(np.logical_and(fl, a>0), a/l)
        res = np.where(np.logical_and(fl, a<0), a/s)
        res = np.where(~fl, a-np.nanmean(a, axis=1))
        res = res / abs(np.nansum(res, axis=1))[:, np.newaxis]
        return res        
    def checksyntax(f):
        def decorated(self, a):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a)
            else:
                return EvalInfo(a.shape, a.unit, Array2d)
        return decorated


class NEU_IND(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(NEU_IND, self).__init__(prob, checksyntax)
    @checksyntax
    def __call__(self, a, ind):
        g = np.where(ind>0, ind, 0)
        m = a.shape[0]
        res = np.full(fill_value=np.nan, dtype=float, shape=a.shape)
        for i in range(m):
            _x, _g = a[i, :], np.unique(g[i, :])
            for ig in _g:
                if len(_x[g==ig]) ==0: continue
                res[i, g==ig] = a[i, g==ig] - np.ma.masked_where(_g!=ig, _x).mean()
        return res
    def checksyntax(f):
        def decorated(self, a, ind):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, ind)
            else:
                if (a.shape==ind.shape) :
                    return EvalInfo(a.shape, a.unit, FinalResult)
                else: raise IllegalSyntex
        return decorated


class NEU_DATA_C(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(NEU_DATA_C, self).__init__(prob, checksyntax)
    def splneutral(ar):
        return ar - np.nanmean(ar)
    @checksyntax
    def __call__(self, a, r, w:int=0, type:int=0):
        m = a.shape[0]
        res = imposter(a)
        if type==0:
            X = r.copy()
        elif type==1:
            tsd = TS_DELTA(checksyntax=False)
            X = tsd(r, type=1, w=w)
        elif type==2:
            tsd = TS_DELTA(checksyntax=False)
            X = tsd(r, type=0, w=w)
        elif type==3:
            tsd = TS_STD(checksyntax=False)
            X = tsd(r, w=w)
        elif type==4:
            tsd = SIGNED_LOG(checksyntax=False)
            X = tsd(a)
        elif type==5:
            tsd = SIGNED_POWER(checksyntax=False)
            X = tsd(a, p=1, rankfirst=True)
        else:
            X = r.copy()
        for i in range(m):
            x = self.splneutral(X[i, :])
            y = self.splneutral(a[i, :])
            out = safe_regression(X=x, Y=y)
            if out[0]==0: 
                res[i, :] = out[1]                
        return res
    def checksyntax(f):
        def decorated(self, a, r):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, r)
            else:
                if (a.shape==r.shape) :
                    return EvalInfo(a.shape, a.unit, FinalResult)
                else: raise IllegalSyntex
        return decorated


class NEU_DATA(OPBase):
    def __init__(self, prob:float=1, checksyntax:bool=True):
        super(NEU_DATA, self).__init__(prob, checksyntax)
    def splneutral(ar):
        return ar - np.nanmean(ar)
    @checksyntax
    def __call__(self, a, r, w:int=0, type:int=0):
        m = a.shape[0]
        res = imposter(a)
        if type==0:
            X = r.copy()
        elif type==1:
            tsd = TS_DELTA(checksyntax=False)
            X = tsd(r, type=1, w=w)
        elif type==2:
            tsd = TS_DELTA(checksyntax=False)
            X = tsd(r, type=0, w=w)
        elif type==3:
            tsd = TS_STD(checksyntax=False)
            X = tsd(r, w=w)
        elif type==4:
            tsd = SIGNED_LOG(checksyntax=False)
            X = tsd(a)
        elif type==5:
            tsd = SIGNED_POWER(checksyntax=False)
            X = tsd(a, p=1, rankfirst=True)
        else:
            X = r.copy()
        for i in range(m):
            x = self.splneutral(X[i, :])
            y = self.splneutral(a[i, :])
            out = safe_regression(X=x, Y=y)
            if out[0]==0: 
                res[i, :] = out[1]                
        return res
    def checksyntax(f):
        def decorated(self, a, r):
            if (not super().checksyntax) or type(a) != EvalInfo:
                return f(self, a, r)
            else:
                if (a.shape==r.shape) :
                    return EvalInfo(a.shape, a.unit, FinalResult)
                else: raise IllegalSyntex
        return decorated



dict_operators = dict()
dict_params = dict()  # 参数遍历范围. 其值为双层元组, 考虑到一个算子可能有多个Ephemeral输入.

dict_operators = {
    'ADD': (ADD(), (Array2d, Array2d), Array2d),
    'OP': (OP(), (Array2d, Array2dValid), FinalResult)
}


dict_params = {
    #'SIGNED_POWER': ((0, 0.1, 1, 3), ),
    #'TS_DELTA':     ((1, 5, 20, 60, 120, 250), ),
    #'TS_MEAN':      ((5, 20, 60, 120, 250), ),
    #'TS_STD':       ((5, 20, 60, 120, 250), ),
    #'TS_ROBUST':    ((5, 20, 60, 120, 250), (0, 1, 2, 3, 4), ),
    # 'OP':           ((1, 5, 10), )
}


# ENDCOPY #
