#!/usr/bin/python3
"""
daily的遗传规划, 使用daily的数据.

注意, run_daily.py中目前没有需要修改的内容, 或COPY到submit中的内容.
依赖 deap 库

2021092    init ver    Zehua Yu / Zihan Zhang
"""

import os
# 禁用NumPy自带的多线程, 这样一个进程最多100% CPU占用. 这段代码必须保证在import numpy前执行.
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ['NUMEXPR_NUM_THREADS'] = '1'
#os.environ['OMP_NUM_THREADS'] = '1'

import pathlib, numpy as np
np.set_printoptions(linewidth=300)

# 加载自定义包
import dlog
from base_classes_funcs import *
from data import DataConstr
from evaluate import eval_fitness_stats, eval_fitness_sharpe
from gp_algorithm import GPLearn
from grow_tree import gen_alpha_tree
from crossover import basicCO, selfCO
from mutation import displacementMT, hoistMT, subtreeMT, pointMT
from choice import choice_individual_reg, choice_individual_simple
from earlystop import EarlyStopCont
from operators import dict_operators, dict_params  # 注意这里import了所有的算子和dict_operators, dict_params
#from operator_daily import dict_operators, dict_params
from optimizer import optimizer
from pnl import load_pnl_from_pickle, dump_pnl_to_pickle

if __name__ == "__main__":
    data_type = Array2d  # 数据的类型
    # 读取配置文件
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # crossover和mutation的名字, 保证顺序一致.
    f_crossover_names = ('basicCO',)
    f_mutation_names = ('hoistMT', 'subtreeMT', 'pointMT', 'selfCO', 'displacementMT')

    args = {
        'quiet': config['GLOBAL']['QUIET'],
        'logfolder': config['GLOBAL']['LOGFOLDER'],
        'datacache': config['GLOBAL']['DATACACHE'],
        'pnlprod': config['GLOBAL']['PNLCACHE'],
        'pnllocal': config['GLOBAL']['PNLLOCAL'],
        'gotosubmit': config['GLOBAL']['FINALSUBMIT'],
        'gotodump': config['GLOBAL']['FINALDUMP'],
        'epochs': config['MODEL']['EPOCHS'],
        'patience': config['MODEL']['PATIENCE'],
        'rollback': config['MODEL']['ROLLBACK'],
        'pa_decay': config['MODEL']['PADECAY'],
        'pop_size': config['MODEL']['POP_SIZE'],
        'min_height': config['MODEL']['MIN_HEIGHT'],
        'max_height': config['MODEL']['MAX_HEIGHT'],
        'delay': config['INPUT']['DELAY'],
        'p_crossovers': [config['MODEL']['STRUCTURE_OPTIMIZER']['CROSSOVER'][f_name]['PROB'] for f_name in f_crossover_names],
        'p_mutations': [config['MODEL']['STRUCTURE_OPTIMIZER']['MUTATION'][f_name]['PROB'] for f_name in f_mutation_names],
        'n_processes': config['MULTIPROCESSING']['N_PROCESSES'],
        'valid_size': config['INPUT']['VALID_SIZE'],
        'start_date': config['INPUT']['START_DATE'],
        'end_date': config['INPUT']['END_DATE'],
        'subUniv': config['INPUT']['SUBUNIV']
    }

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # 日志
    logfolder = pathlib.Path(args['logfolder'])
    logfolder.mkdir(mode=0o755, parents=True, exist_ok=True)
    dlog.init_log(logfolder, not args['quiet'])

    # 数据读取和处理.
    dlog.info('GPLearn Start')
    dlog.info('register data and op ...')
    dc = DataConstr(subUniv=args['subUniv'])
    f_load_data_io = partial(f_load_data_io, data_folder=args['datacache'], start_date=args['start_date'], end_date=args['end_date'])
    data, slc_train, slc_valid = dc.load_data(args['delay'], 0, args['valid_size'], f_load_data=f_load_data_io)
    dict_data_eval_info, n_units, pset, data_classified_by_units = dc.register_data(dict_operators, data_type)
    data_train = {k: v[slc_train] for k, v in data.items()}
    data_valid = {k: v[slc_valid] for k, v in data.items()}
    tradedates = dc.dates

    """
    pset.context打印如下:
    {'__builtins__': None, 
    'add_2d': <function add_2d at 0x7f91403427b8>, 
    'sub_2d': <function sub_2d at 0x7f9140342840>, 
    'mul_2d': <function mul_2d at 0x7f91403428c8>, 
    'div_2d': <function div_2d at 0x7f9140342950>, 
    'ewma': <function ewma at 0x7f9140342ae8>}

    also, pset.arguments prints:
    ['OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME', 'VWAP', 'RETURN']

    pset.primitives (a defaultdict that returns an empty list if key does not exist):
    defaultdict(<class 'list'>, 
    {<class 'base_classes_funcs.Array2d'>: [<deap.gp.Primitive object at 0x7f933bb10e58>, <deap.gp.Primitive object at 0x7f933bb10f48>, <deap.gp.Primitive object at 0x7f933bb10138>, <deap.gp.Primitive object at 0x7f933bb10868>], 
    <class 'base_classes_funcs.FinalResult'>: [<deap.gp.Primitive object at 0x7f933bb10d18>],
     <class 'base_classes_funcs.Ephemeral'>: []})

    pset.terminals (similarly, a defaultdict):
    defaultdict(<class 'list'>, 
    {<class 'base_classes_funcs.Array2d'>: [<deap.gp.Terminal object at 0x7f93336001f8>, <deap.gp.Terminal object at 0x7f933be1c3f0>, <deap.gp.Terminal object at 0x7f933bdb5f78>, <deap.gp.Terminal object at 0x7f933bb18fc0>, <deap.gp.Terminal object at 0x7f933bb18c18>, <deap.gp.Terminal object at 0x7f933bb18c60>, <deap.gp.Terminal object at 0x7f933bb188b8>], 
    <class 'base_classes_funcs.FinalResult'>: [], 
    <class 'base_classes_funcs.Ephemeral'>: [<class 'deap.gp.eph'>]})

    """

    """
    data_classified_by_units: (内存地址应该和上面一样)
    [[<deap.gp.Terminal at 0x18e006795c0>,
      <deap.gp.Terminal at 0x18e00679440>,
      <deap.gp.Terminal at 0x18e00679900>,
      <deap.gp.Terminal at 0x18e00679680>,
      <deap.gp.Terminal at 0x18e00679740>],
     [<deap.gp.Terminal at 0x18e006797c0>],
     [<deap.gp.Terminal at 0x18e00679040>]]
    """

    # register pnl
    dlog.info('register pnl ...')
    datastr = datetime.datetime.today().date().strftime('%Y%m%d')
    filepkl = '_'.join([datastr, str(args['start_date']), str(args['end_date'])])+'.npy'
    localfolder = pathlib.Path(args['pnllocal'])
    prodfolder  = pathlib.Path(args['pnlprod'])
    localpnl_pkl = localfolder.joinpath(filepkl)
    err, pnl_pool = load_pnl_from_pickle(localpnl_pkl)
    if err != 0:
        pnl_pool = dump_pnl_to_pickle(src=prodfolder, dst=localfolder, pickfile=localpnl_pkl,
                                      dates=tradedates)
    pnl_pool = ts_delay(pnl_pool, -1, axis=0) # 注意pnl.date和future_ret的日期对应. 这里是delay负1天
    pnl_pool_train = pnl_pool[slc_train]
    pnl_pool_valid = pnl_pool[slc_valid]

    # 生成初始种群
    dlog.info('initialize population ...')
    dlog.info('individual sample')
    population_init = []
    for i in range(args['pop_size']):
        expr = gen_alpha_tree(pset,
                              args['min_height'],
                              args['max_height'],
                              FinalResult,
                              data_classified_by_units,
                              n_units,
                              data_type,
                              dict_operators,
                              dict_data_eval_info)
        # print('初始化前: {}'.format(str(gp.PrimitiveTree(expr))))
        individual = Individual(expr)
        optimizer.init(individual, 
                       dict_operators=dict_operators,
                       dict_params=dict_params)
        # print('初始化后: {}'.format(str(gp.PrimitiveTree(expr))))
        population_init.append(individual)
        if i < 5:
            dlog.info(expr_to_str(expr))

    # 生成partial function作为参数传入
    dlog.info('register partial functions ...')
    functions = {
        # 这个字典里的键名要与YAML中的函数名对应, 而后者是与真实的函数名对应的.
        'basicCO': partial(basicCO,
                           dict_operators=dict_operators,
                           dict_data_eval_info=dict_data_eval_info),
        'hoistMT': partial(hoistMT,
                           dict_operators=dict_operators,
                           dict_data_eval_info=dict_data_eval_info),
        'subtreeMT': partial(subtreeMT,
                             minHeight=args['min_height'],
                             maxHeight=args['max_height'],
                             attemptTime=5,
                             pset=pset,
                             dict_operators=dict_operators,
                             dict_data_eval_info=dict_data_eval_info,
                             data_classified_by_units=data_classified_by_units,
                             n_units=n_units,
                             data_type=data_type,
                             dict_params=dict_params,
                             f_init_a_tree=optimizer.init,
                             frequent_subtrees=None),
        'pointMT': partial(pointMT,
                           pset=pset,
                           replace_num=5,
                           dict_operators=dict_operators,
                           dict_data_eval_info=dict_data_eval_info,
                           data_classified_by_units=data_classified_by_units,
                           dict_params=dict_params),
        'selfCO': partial(selfCO,
                          dict_operators=dict_operators,
                          dict_data_eval_info=dict_data_eval_info),
        'displacementMT': partial(displacementMT,
                                  dict_operators=dict_operators,
                                  dict_data_eval_info=dict_data_eval_info)
    }
    # 输入ind1, ind2
    f_crossovers = [functions[f_name] for f_name in config['MODEL']['STRUCTURE_OPTIMIZER']['CROSSOVER']]
    # 输入ind1
    f_mutations = [functions[f_name] for f_name in config['MODEL']['STRUCTURE_OPTIMIZER']['MUTATION']]
    f_evaluate = partial(eval_fitness_stats,
                         pset=pset,
                         pnl_pool=pnl_pool_train
                         )
    # 输入expr, data_list, future_returns
    f_optimize = partial(optimizer.run,
                         f_evaluate=f_evaluate,
                         dict_operators=dict_operators,
                         dict_params=dict_params,
                         quiet=True)
    f_choice = partial(choice_individual_reg,
                       num=args['pop_size'])
    # 早停的parital
    el_obj = EarlyStopCont(args['patience'], args['rollback'])
    f_earlystop = partial(el_obj.is_early_stop,
                          data_valid=data_valid,
                          today_returns=data_valid['TODAY_RETURNS'],
                          dict_data_eval_info=dict_data_eval_info,
                          pset=pset,
                          f_evaluate=eval_fitness_sharpe,
                          decay=args['pa_decay'])
    f_rollback = partial(el_obj.rollback)

    # 进行遗传规划
    try:
        dlog.info('start to run gplearn ...')
        gpa = GPLearn(f_crossovers=f_crossovers,
                      f_mutations=f_mutations,
                      f_choice=f_choice)
        gpa.run(epochs=args['epochs'],
                population_init=population_init,
                data_train=data_train,
                dict_data_eval_info=dict_data_eval_info,
                n_processes=args['n_processes'],
                p_crossovers=args['p_crossovers'],
                p_mutations=args['p_mutations'],
                f_optimize=f_optimize,
                f_earlystop=f_earlystop,
                f_rollback=f_rollback,
                dlog=dlog,
                timeout=60,
                waitCPU=False)
    except KeyboardInterrupt as e:
        dlog.warning('Ctrl-C catched')
        dlog.warning('gplearn interrupted')
        sys.exit(-127)
    except Exception as e:
        dlog.warning(str(e))
        dlog.warning('failed to run gplearn')
        sys.exit(-1)
    dlog.info('succ to run gplearn')
    # 保存
    if args['gotosubmit']:
        time.sleep(5)
        subprocess.call('python3 generate_code.py --log {}'.format(logfile), shell=True)
        time.sleep(5)
        subprocess.call('python3 submit.py', shell=True)
    if args['gotodump']:
        pass


