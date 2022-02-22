"""
代码截取自showpnl
"""

import os, sys
if '/inc' not in sys.path:
    sys.path.append('/inc')
import add_py_path
import os.path, sys, optparse, numpy, shutil, pathlib
from py3lib import asim_arena_lib as alib

from base_classes_funcs import *


def generate_csv_from_pnl(pnl_file_name):
    """在.pnl文件的源路径下新生成一个.csv文件. 拷贝自export_to_csv函数. pnl_file_name需包含路径. """
    pnlc = alib.read_pnl_from_file(pnl_file_name)
    pnl = pnlc[1]
    if pnl is None:
        print('pnl文件{}不存在!'.format(pnl_file_name))
        pdb.set_trace()
    csv_file_name = pnl_file_name[:-4] + '.csv'
    outf = open(csv_file_name, 'w')
    outf.write(alib.pnl_columns + '\n')
    f = ','.join(['%g'] * 14) + ',%d,%d,%d'
    for d in pnl:
        outf.write((f + '\n') % d)
    outf.close()
    return csv_file_name


def generate_ndarray_from_pnl(pnl_file_name, end_date, delete_csv=True):
    """首先用generate_csv_from_pnl生成csv文件, 然后读取csv文件的一部分为ndarray文件. pnl_file_name需包含路径. """
    csv_file_name = generate_csv_from_pnl(pnl_file_name)
    df = pd.read_csv(csv_file_name)
    ret = df.loc[:, ['pnl_net', 'date']]

    index_end_date = np.searchsorted(ret.date, end_date)

    ret = ret.pnl_net.to_numpy()[:index_end_date]

    # 取最后500日数据. 若不足500日, 补足.
    if ret.shape[0] >= 500:
        ret_500 = ret[-500:]
    else:
        ret_500 = np.full((500, ), np.nan)
        if ret.shape[0] > 0:  # ret可能为空
            ret_500[-ret.shape[0]:] = ret

    if mask(ret_500).sum(0) == 0:  # 如果没有任何一天有pnl
        print('剔除{}'.format(pnl_file_name))
        return

    if delete_csv:
        os.remove(csv_file_name)

    return ret_500


def load_pnl_from_pickle(srcfile:pathlib.Path) -> (int, numpy.ndarray):
    if srcfile.exists():
        return 0, np.load(srcfile, allow_pickle=True)
    else:
        return -1, None

def dump_pnl_to_pickle(src:pathlib.Path, dst:pathlib.Path, pickfile, dates:list()):
    #shutil.rmtree(dst)
    dst.mkdir(parents=True, mode=0o755, exist_ok=True)
    cnt = 0
    for f in src.iterdir():
        if f.is_dir(): continue
        if f.name[-4:] == '.pnl' and 'ZZ500' not in f.name and 'HS300' not in f.name:
            #shutil.copyfile(f, pathlib.Path(dst).joinpath(f.name))
            cnt += 1
    if cnt == 0:
        raise Exception('no pnl found')
    res = pd.DataFrame(data=0.0, dtype=float, index=dates, columns=range(cnt))
    cnt = 0
    for f in dst.iterdir():
        if f.is_dir(): continue
        if f.name[-4:] == '.pnl' and 'ZZ500' not in f.name and 'HS300' not in f.name:
            _, pnl = alib.read_pnl_from_file(str(f))
            if pnl:
                x = np.array([(x.date, x.pnl_net/x.booksize) for x in pnl if (x.date>=dates[0] and x.date<=dates[-1] and x.booksize>0)]) # 这里是 [] 区间,dates[-1] 是最后一个数据点，非enddate. 
                res.loc[x[:, 0].astype(int), cnt] = x[:, 1]
                cnt += 1
    res = res.to_numpy()
    np.save(pickfile, res, allow_pickle=True)
    return res
