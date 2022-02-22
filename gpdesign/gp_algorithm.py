#!/usr/bin/python3
"""
对标deap包中的algorithms部分. 例如algorithms.eaSimple. todo: 继承和重写 deap.algorithms.eaSimple 等
"""

from abc import ABC, abstractmethod
import concurrent.futures
# 加载自定义包
from base_classes_funcs import *


class GPLearnBase(ABC):
    def __init__(self, f_crossovers:list(), f_mutations:list(), f_choice):
        self.__f_crossovers = f_crossovers
        self.__f_mutations  = f_mutations
        self.__f_choice = f_choice
    @property
    def f_crossovers(self):
        return self.__f_crossovers
    @property
    def f_mutations(self):
        return self.__f_mutations
    @property
    def f_choice(self):
        return self.__f_choice
    @abstractmethod
    def run(self):
        raise NotImplemented


class GPLearn(GPLearnBase):
    def __init__(self, f_crossovers, f_mutations, f_choice):
        super(GPLearn, self).__init__(f_crossovers, f_mutations, f_choice)
    def run(self, epochs, population_init, data_train, 
            dict_data_eval_info, n_processes,
            p_crossovers, p_mutations, f_optimize, f_earlystop, f_rollback, dlog,
            timeout, waitCPU):
        # 一些参数的初始化
        timeout_l = timeout if timeout > 0 else np.inf
        population = population_init.copy()
        # population_fittest = population
        initial_pop_size = len(population)
        current_pop_size = len(population)
        data_train_list = [data_train[k] for k in dict_data_eval_info.keys()] 
        # 统计量
        stats_succ_rate = dict()
        try:
            # 训练
            for epoch in range(epochs):
                st = time.time()
                dlog.info(f'Epoch {epoch+1} / {epochs}')
                dlog.info(f'current population size = {current_pop_size}') 
                # 检查当前时间. 若在AM 4:00 - 8:59期间, 暂停, 防止干扰生产.
                while waitCPU: time.sleep(5)
                # 所有可能进入下一代的候选个体
                candidates = population.copy()
                # 1. 结构寻优
                dlog.info('generating alpha tree ...')
                # 首先生成所有树的两两组合
                # crossover, 需要将parent代两两配对. 有概率不成功.
                dlog.info('running crossover ...')
                expr_from_crossover = []
                for i_crossover, f_crossover in enumerate(super().f_crossovers):
                    if p_crossovers[i_crossover] <= 0: continue
                    n_succ, n_tries = 0, 0
                    permutation_individuals = np.random.permutation(current_pop_size)
                    for i_individual in range(0, current_pop_size-2, 2):
                        if i_individual % 10 == 0: dlog.info(f'try {f_crossover.func.__name__} for No. {i_individual} individual ...')
                        pair = permutation_individuals[i_individual], permutation_individuals[i_individual+1]
                        p = random.random()
                        if p < p_crossovers[i_crossover]:
                            #import pdb; pdb.set_trace()
                            startTime, endTime = time.time(), time.time()
                            success, stamp1, stamp2 = False, None, None
                            while (not success) and (endTime-startTime<timeout_l):
                                n_tries += 1
                                #_st = time.time()
                                success, offspring1, offspring2, stamp1, stamp2 = f_crossover(population[pair[0]].expr, population[pair[1]].expr, stamp1, stamp2)
                                endTime = time.time()
                                #print(endTime-_st)
                            #dlog.info('time passed %.2f sec to try cross' % (endTime-startTime))
                            if success:
                                n_succ += 1  
                                expr_from_crossover.append([population[pair[0]].expr, population[pair[1]].expr, offspring1, offspring2])
                                candidates = candidates + [Individual(offspring1), Individual(offspring2)] # 这里不会更改 population，并且个体数会增加大于等于初始的种群数
                                #import pdb; pdb.set_trace()
                    stats_succ_rate[f_crossover.func.__name__] = n_succ, n_tries
                    dlog.info(f'f_crossover = {f_crossover.func.__name__} has {n_succ}/{n_tries} succs')

                # mutation
                dlog.info('running mutation ...')
                expr_from_mutation = []
                for i_mutation, f_mutation in enumerate(super().f_mutations):
                    if p_mutations[i_mutation] <= 0: continue
                    n_succ, n_tries = 0, 0
                    for i_individual in range(current_pop_size):
                        if i_individual % 10 == 0: dlog.info(f'try {f_mutation.func.__name__} for No. {i_individual} individual ...')
                        p = random.random()
                        if p < p_mutations[i_mutation]:
                            startTime, endTime = time.time(), time.time()
                            success, cxpoint = False, None
                            while (not success) and (endTime-startTime<timeout_l):
                                n_tries += 1
                                success, offspring, cxpoint = f_mutation(population[i_individual].expr, cxpoint)
                                endTime = time.time()
                            if success:
                                n_succ += 1
                                expr_from_mutation.append([population[i_individual].expr, offspring])
                                candidates = candidates + [Individual(offspring)]
                    stats_succ_rate[f_mutation.func.__name__] = n_succ, n_tries
                    dlog.info(f'f_mutation = {f_mutation.func.__name__} has {n_succ}/{n_tries} succs')


                # 2. 参数寻优. 这是对每个个体逐个执行的. 部分个体没有变化, 无需重新计算.
                # 由于gpalpha计算密集而IO开销较小, 且多线程无法将一个线程限制到一个核内、无法绕过GIL, 使用多进程而非多线程
                st2 = time.time()
                total_candi_size = len(candidates)
                if n_processes > 0:
                    tasks, runresults = [], []
                    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
                        for i_candi, candi in enumerate(candidates):
                            if candi.fitness is None: # 不重复计算
                                tasks.append(executor.submit(f_optimize, i_candi, candi, data_train_list, data_train['TODAY_RETURNS']))
                            else: runresults.append(candi)
                        for ret in concurrent.futures.as_completed(tasks): # submit 不会按顺序返回，但是没有关系，以下顺序无关
                            if not ret: dlog.warning('error when run a candidate')
                            else:
                                r = ret.result() # idx, individual, text
                                if r is None: dlog.warning('error when run a candidate. returned None')
                                else:
                                    runresults.append(r[1])
                                    dlog.info('succ to run %d candidate' % (r[0]))
                    candidates = runresults.copy()
                    #pool = multiprocessing.Pool(n_processes)
                    #for i_candi, candi in enumerate(candidates):
                    #    if candi.fitness is None:
                    #        tasks.append(pool.apply_async(f_optimize, args=(i_candi, candi, data_train_list, data_train['TODAY_RETURNS'])))
                    #pool.close()
                    #pool.join()
                    #candidates = [] # 原代码不重复计算的处理有误. 这里重新清空了candidates, fitness is not None 的个体没有参与这一轮计算，不会被append到新的candidates中
                    #for ret in tasks:
                    #    r = ret.get()
                    #    if r is None: dlog.warning('error when run a candidate. returned None')
                    #    else:
                    #        candidates.append(r[1])
                    #        dlog.info('succ to run %d candidate' % (r[0]))

                else: # 单进程，调试用
                    runresults = []
                    for i_candi, candi in enumerate(candidates):
                        if candi.fitness is None:
                            r = f_optimize(i_candi, candi, data_train_list, data_train['TODAY_RETURNS'])
                            if r is None: dlog.warning(f'error when run {i_candi} candidate. returned None')
                            else:
                                runresults.append(r[1])
                                dlog.info(f'succ to run {i_candi} candidate')
                        else: runresults.append(candi)
                    candidates = runresults.copy()
                et = time.time() - st
                et2= time.time() - st2
                dlog.info('time passed for generating tree candidates: %.2f sec totally, included %.2f sec in optimization, %s/%d' % (et, et2, len(candidates), total_candi_size))

                # 3. 根据个体适应度函数等筛选优良的子代
                #import pdb; pdb.set_trace()
                population = super().f_choice(candidates)
                current_pop_size = len(population)
                print(table_population_statistics(candidates, 'curr candidates stats.'))
                print(table_population_statistics(population, 'next population stats.'))
                if current_pop_size == 0:
                    raise Exception('no enough individual found')

                # 4. 早停
                if f_earlystop and f_earlystop(population):
                    dlog.info(f'early stop when epoch={epoch}')
                    if f_rollback:
                        population = f_rollback()
                        dlog.info(f'final population will rollback to {f_rollback(t=1)}')
                    break

        except BaseException as e:  # 注意不是Excepetion, KeyboardInterrupt和Exception都继承自BaseException
            print(e)
            print(traceback.format_exc())
            # pdb.set_trace()
        return population


