#!/usr/bin/python3
"""
这里存放局部优化器, 即在对结构进行一定程度的寻优后, 对数值参数寻优(这两者应该都算模型的参数), 交替进行.
之前用的是梯度下降法, 但带来了很多问题. 这里从简, 直接对每个算子给出范围遍历.

局部优化器函数的特殊之处在于, 它们会被在多个进程中执行. 这意味着Python会对传入的参数所在的内存进行拷贝(而非深拷贝).
因此, 这些函数无法对传入的对象进行原址修改. 任何想要修改的内容必须以返回值的形式传出. 不过, initialise=True的情况除外.
另外, text仍然会在运行时进行打印, 否则要等到所有进程结束再打印, 运行时看不到任何东西. 但是, log是全部运行完后再写入的,
因为Writer对象无法序列化, 因此无法作为参数传入一个异步执行的函数里.
"""
from abc import ABC, abstractmethod
import sys

# 加载自定义包
from base_classes_funcs import *


class OptimizerBase(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def run(self):
        raise NotImplemented


class OptimizerGreedyAlgoDFS(OptimizerBase):
    def __init__(self):
        super(OptimizerGreedyAlgoDFS, self).__init__()
    def init(self, indi, dict_operators, dict_params): 
        # 利用原址修改
        expr = indi.expr
        # 遍历所有算子. 由于是DFS, 算子的参数一定在算子之后被遍历.
        for idx_node in reversed(range(len(expr))): # 从树的接近叶结点处开始尝试参数, 这些数值参数影响更大
            node = expr[idx_node]
            if isinstance(node, gp.Primitive) and node.name in dict_params.keys():
                count_ephemerals_in_offsprings = 0
                offsprints_type = dict_operators[node.name][1] # (Array2d, Ephemeral)
                idx_curr = idx_node + 1 # 从第一个子树的根节点开始
                for offspring_type in offsprints_type: # 注意，可能有多个Ephemeral
                    # 注意, 若grow_tree过程进行语法检查，那么此时应该恰好成立offspring_type == expr[idx_curr].ret
                    if offspring_type != Ephemeral:
                        total = expr[idx_curr].arity
                        idx_curr += 1
                        while total > 0: # todo: 非Ephemeral也可能是数据，同样arity==0
                            total += expr[idx_curr].arity - 1
                            idx_curr += 1
                    else:
                        expr[idx_curr].value = random.choice(dict_params[node.name][count_ephemerals_in_offsprings])
                        expr[idx_curr].name = str(expr[idx_curr].value) # todo: name 的作用，可以相同吗
                        idx_curr += 1
                        count_ephemerals_in_offsprings += 1

    def run(self, idx, indi, 
            data_train_list, 
            data_train_today_returns, 
            dict_operators,
            dict_params,
            f_evaluate, 
            quiet=True): 
        expr = indi.expr
        text = '*'*10 + ' OPTIMIZE DETAIL: %s ' % (sys._getframe().f_code.co_name) + '*'*10
        text += '\n'
        text += expr_to_str(expr) + '\n'
        max_fitness_prev_node = -np.inf
        indi.fitness = -np.inf
        # 遍历所有算子. 由于是DFS, 算子的参数一定在算子之后被遍历.
        for idx_node in reversed(range(len(expr))): # 从树的接近叶结点处开始尝试参数, 这些数值参数影响更大
            node = expr[idx_node]
            if isinstance(node, gp.Primitive) and node.name in dict_params.keys():
                count_ephemerals_in_offsprings = 0
                offsprints_type = dict_operators[node.name][1] # (Array2d, Ephemeral)
                idx_curr = idx_node + 1 # 从第一个子树的根节点开始
                for offspring_type in offsprints_type: # 注意，可能有多个Ephemeral
                    # 注意, 若grow_tree过程进行语法检查，那么此时应该恰好成立offspring_type == expr[idx_curr].ret
                    if offspring_type != Ephemeral:
                        total = expr[idx_curr].arity
                        idx_curr += 1
                        while total > 0: # todo: 非Ephemeral也可能是数据，同样arity==0
                            total += expr[idx_curr].arity - 1
                            idx_curr += 1
                    else:
                        text += f'optimizing node {node.name} -> Ephemeral {count_ephemerals_in_offsprings} ...\n'
                        value_tried = expr[idx_curr].value
                        for value_candi in dict_params[node.name][count_ephemerals_in_offsprings]:
                            if max_fitness_prev_node > -np.inf and value_candi == value_tried:
                                # 如果之前就是按这个值算的, 无需重复计算. 不过所有第一个算子除外.
                                # 这里用 ==-np.inf 表示是否为第一个算子
                                # todo: why first OP again?
                                text += f'try {value_candi} again. previous node maximum fitness = {max_fitness_prev_node}\n'
                                continue
                            value_prev = expr[idx_curr].value
                            expr[idx_curr].value = value_candi
                            expr[idx_curr].name = str(expr[idx_curr].value)
                            fitness, fitness_raw, indi_stats = f_evaluate(expr,
                                                                          data_train_list,
                                                                          data_train_today_returns)
                            if (not (np.isnan(fitness) or np.isinf(fitness))) and fitness > indi.fitness:
                                expr[idx_curr].value = value_candi
                                indi.fitness = fitness
                                indi.fitness_raw = fitness_raw
                                indi.stats = indi_stats
                            else:
                                expr[idx_curr].value = value_prev
                                if np.isnan(fitness) or np.isinf(fitness):
                                    text += '**** invalid fitness found, exclude it. **** \n'
                            expr[idx_curr].name = str(expr[idx_curr].value)
                            text += f'try {value_candi}. curr fitness = {fitness}, maximum fitness = {indi.fitness}\n'
                            # todo: 原代码进入下面的分支则直接 reutrn 退出
                            # 如果fitness为nan或inf, 直接排除
                            # if np.isnan(fitness) or np.isinf(fitness):
                            #     text += 'invalid fitness found, exclude it.\n'
                            #     text += '出现nan或inf, 直接排除\n\n'
                            #     individual.fitness = -np.inf  # 将fitness设为-inf, 与未优化状态None相区分
                            #     if not quiet: print(text)
                            #     return idx, individual, text
                        max_fitness_prev_node = indi.fitness
                        idx_curr += 1
                        count_ephemerals_in_offsprings += 1
        text += '{}\n'.format(expr_to_str(expr))
        text += f'current alpha maximum fitness = {indi.fitness}, maximum raw fitness = {indi.fitness_raw}\n'
        text = '*'*10 + ' OPTIMIZE DETAIL: %s ' % (sys._getframe().f_code.co_name) + '*'*10
        if not quiet: print(text)
        return idx, indi, text
                                



def local_optimizer_sgd(expr, dict_operators, dict_params, initialise):
    """旧代码备份, 防止丢失. """

    # # extract all the ephemerals and create pytorch 1d-tensors from them
    # if mode == 'max':
    #     mode = 1
    # elif mode == 'min':  # α <- -α
    #     mode = -1
    # else:
    #     print('No optimization mode selected!')
    #
    # TENSOR_EPHS = []
    # expr_new = []
    # for node in expr:
    #     if node.ret == ephemeral:
    #         value = "TENSOR_EPHS[{}]".format(len(TENSOR_EPHS))
    #         expr_new.append(gp.Terminal(value, value, True))  # conv_fct=True makes the name be printed without ''
    #         TENSOR_EPHS.append(torch.tensor(node.value, requires_grad=True))
    #     else:
    #         expr_new.append(node)
    #
    # # transform the tree expression into a callable function. the code is modified from def compile() source code.
    # code = str(gp.PrimitiveTree(expr_new))
    # args = ",".join(arg for arg in pset.arguments)
    # code = "lambda {args}: {code}".format(args=args, code=code)
    # globs = pset.context.copy();
    # globs['__builtins__'] = {'TENSOR_EPHS': TENSOR_EPHS}
    #
    # # stochastic gradient descent
    # for epoch in range(epochs):
    #     t = time.time()
    #     # for each epoch, we note down the IC series for this epoch, and the IC of the final epoch is used to select individuals
    #     epoch_IC_series = np.array([])
    #     epoch_tvr_series = np.array([])
    #     # generate random permutation, and choose minibatches by that order
    #     permutation = np.random.permutation(int(np.ceil(standard_shape[0] / minibatch)))
    #     for i in permutation:  # train a minibatch
    #         func = eval(code, globs)  # (re)eval the func. This becomes a function that can accept parameters.
    #         minibatch_mask = (slice(i * minibatch, (i + 1) * minibatch), slice(None),
    #                           slice(None))  # for convenience, otherwise code too long
    #         minibatch_data_list = [data_tensor[field][minibatch_mask] for field in fields]
    #         alpha = mode * func(*minibatch_data_list)
    #
    #         # rescale alpha so that each row (day) has mean 0 and abssum of 1. it can now be interpreted as weights.
    #         temp_value = torch.zeros_like(alpha)
    #         temp_value[MASK(alpha)] = alpha[MASK(alpha)]
    #         temp_count = torch.zeros_like(alpha)
    #         temp_count[MASK(alpha)] = 1.
    #         temp_mean = torch.sum(temp_value, 1, keepdim=True) / torch.sum(temp_count, 1, keepdim=True)
    #         alpha -= temp_mean
    #
    #         temp_value = torch.zeros_like(alpha)
    #         temp_value[MASK(alpha)] = alpha[MASK(alpha)]
    #         temp_value = torch.sum(torch.abs(temp_value), 1, keepdim=True).expand(-1, alpha.shape[1])
    #         alpha[MASK(alpha)] = alpha[MASK(alpha)] / temp_value[MASK(alpha)]
    #
    #         # calculate minibatch IC
    #         minibatch_mask = (slice(i * minibatch, (i + 1) * minibatch), slice(None))
    #         minibatch_IC_series = calIC(alpha, return_window[minibatch_mask])  # this is a torch.tensor
    #         minibatch_IC_valid_mask = ~torch.isnan(minibatch_IC_series)
    #         epoch_IC_series = np.append(epoch_IC_series, minibatch_IC_series[
    #             minibatch_IC_valid_mask].detach().cpu().numpy())  # update. not in time order but fine.
    #         IC = torch.mean(minibatch_IC_series[minibatch_IC_valid_mask])
    #
    #         # penalize IC with turnover rate.
    #         temp_value = torch.zeros_like(alpha)
    #         mask_common = (~torch.isnan(alpha[1:, ...]) & ~torch.isnan(alpha[:-1, ...]))
    #         mask_1 = torch.cat([mask_common, torch.full((1, alpha.shape[1]), False)], dim=0)
    #         mask_2 = torch.cat([torch.full((1, alpha.shape[1]), False), mask_common], dim=0)
    #         temp_value[mask_1] = torch.abs(alpha[mask_2] - alpha[mask_1])
    #         alpha_tvr = temp_value.sum(1)
    #         epoch_tvr_series = np.append(epoch_tvr_series, alpha_tvr[alpha_tvr > 0].detach().cpu().numpy())
    #         alpha_tvr_avg = alpha_tvr[alpha_tvr > 0].mean()
    #         IC -= reg_tvr * alpha_tvr_avg  # if IC is 0.05 (very good), but tvr is 200%, then penalize to IC = 0.01
    #
    #         # calculated panalized ICIR
    #         ICIR = IC / torch.std(minibatch_IC_series[minibatch_IC_valid_mask])
    #
    #         # backpropagation and stochastic gradient ascent
    #         ICIR.backward()
    #         with torch.no_grad():
    #             for eph in TENSOR_EPHS:
    #                 eph += eph.grad * lr
    #                 print("gradient", eph.grad)
    #                 eph.grad.zero_()
    #     print("epoch {}, IC: {:6.4f}, IR: {:6.4f}, tvr: {:4.2f}%, time: {:4.2f}s".format(
    #         epoch + 1, epoch_IC_series.mean(), epoch_IC_series.mean() / epoch_IC_series.std(),
    #         epoch_tvr_series.mean() * 100, time.time() - t))
    #
    # # finally, update the values in the original expr
    # i = 0
    # for node in expr:
    #     if node.ret == ephemeral:
    #         value = TENSOR_EPHS[i].item()
    #         node.value = value
    #         node.name = str(value)
    #         i += 1
    # return epoch_IC_series
    pass

optimizer = OptimizerGreedyAlgoDFS()

