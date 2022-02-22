# 加载自定义包
import re
from base_classes_funcs import *

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

class Break(Exception):
    pass

def basicCO(ind1, ind2, stamp1, stamp2, dict_operators, dict_data_eval_info):
    '''
    Similar with cxTwoPoint() function of deep package.
    Crossover the whole subtree if the shape and unit of certain root nodes are consistent.
    Traversal all root points.
    '''
    original_ind1 = copy.deepcopy(ind1)
    original_ind2 = copy.deepcopy(ind2)
    # print(f"Original1: {expr_to_str(original_ind1)}")
    # print(f"Original2: {expr_to_str(original_ind2)}")

    random_arr1 = np.random.permutation(len(ind1))
    random_arr2 = np.random.permutation(len(ind2))

    for i in random_arr1:
        for j in random_arr2:
            replace_node1 = ind1[i]
            replace_node2 = ind2[j]
            # 声明不会出现换operator常数的情况, 同时保证换的子树的root node的种类要一样
            if (not replace_node1.ret == Ephemeral) and replace_node1.ret == replace_node2.ret:
                result = compare(ind1, ind2, i, j, dict_operators, dict_data_eval_info)
                # 0806 Update: 这里不需要判断换的点的量纲和shape是否相同，只要交换完的结果在后面check_syntax的部分符合要求即可
                # if result[0]:
                if result[0] == 0 or result[0] == 1:
                    # begins and ends of crossover subtree
                    tree1_stamp = [i, i + result[1]]
                    tree2_stamp = [j, j + result[2]]
                    if tree1_stamp == stamp1 and tree2_stamp == stamp2: continue
                    # print(tree1_stamp, tree2_stamp)

                    # 确保存储的内存地址不一样
                    period_ind1, period_ind2 = copy.deepcopy(ind1), copy.deepcopy(ind2)

                    newind1, newind2 = [], []  # create new objects and modify / store
                    newind1.extend(period_ind1[:tree1_stamp[0]])
                    newind1.extend(period_ind2[tree2_stamp[0]: tree2_stamp[1]])  # crossover subtree
                    newind1.extend(period_ind1[tree1_stamp[1]:])
                    newind2.extend(period_ind2[:tree2_stamp[0]])
                    newind2.extend(period_ind1[tree1_stamp[0]: tree1_stamp[1]])  # crossover subtree
                    newind2.extend(period_ind2[tree2_stamp[1]:])

                    # Evaluate whether the new trees are valid in syntax.
                    if check_syntax(newind1, dict_operators, dict_data_eval_info) \
                            and check_syntax(newind2, dict_operators, dict_data_eval_info):
                        # Evaluate whether the new generated trees are identical with previous.
                        if (expr_to_str(newind1) != expr_to_str(ind1) and expr_to_str(newind2) != expr_to_str(ind2)) \
                                and (expr_to_str(newind1) != expr_to_str(ind2) and expr_to_str(newind2) != expr_to_str(ind1)):
                            # print(f"Output1: {expr_to_str(newind1)}")
                            # print(f"Output2: {expr_to_str(newind2)}")
                            # return newind1, newind2, True
                            return True, newind1, newind2, tree1_stamp, tree2_stamp

    # go over all circumstances but do not find any eligible crossover.
    exception_printer(0, sys._getframe().f_code.co_name)
    # return original_ind1, original_ind2, False
    return False, original_ind1, original_ind2, [0, 0], [0, 0]

def selfCO(ind1, stamp1, stamp2, dict_operators, dict_data_eval_info):
    '''
    Self crossover. Strengthen the signal. Notice that only one input tree is required.
    Sometimes not very useful, ex. DIV_2D(subtree(a), subtree(b)) is altered to DIV_2D(subtree(a), subtree(a)).
    TODO:虽然是crossover但需要按照mutation的方式传参并且发生的概率应小于正常的mutation.
    '''
    result = basicCO(ind1, ind1, None, None, dict_operators, dict_data_eval_info)
    # return result[0], result[2]
    return result[0], result[1]

# TODO: 此alpha的逻辑性还有待衡量。目前看上去和Point Mutation没有本质区别，不符合基因交换的基本逻辑。
# def uniformCO(ind1, ind2, indpb, dict_operators, dict_data_eval_info):
#     '''
#     Executes a uniform crossover that modify the two term sequence individuals.
#     Only swap the (single) nodes if their units and shapes are consistent.
#     The attributes are swapped according to the *indpb* probability.
#     '''
#     print(f"original: {expr_to_str(ind1)}")
#     print(f"original: {expr_to_str(ind2)}")
#     newind1, newind2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
#     size = min(len(newind1), len(newind2))
#     push = random.randint(0, max(len(newind1), len(newind2)) - size)
#     print(f'push: {push}; size: {size}')
#     periodic_result = [[newind1, newind2]]
#
#     for i in range(1, size):
#         if len(newind1) < len(newind2):
#             (a, b) = (i, i + push)
#         else:
#             (a, b) = (i + push, i)
#         if random.random() < indpb and compare(newind1, newind2, a, b, dict_operators, dict_data_eval_info):
#             print(a, b)
#             print(len(periodic_result))
#             print(f"now: {expr_to_str(newind1)}")
#             print(f"now: {expr_to_str(newind2)}")
#             newind1[a], newind2[b] = newind2[b], newind1[a]
#             try:
#                 if check_syntax(newind1, dict_operators, dict_data_eval_info) \
#                         and check_syntax(newind2, dict_operators, dict_data_eval_info):
#                     periodic_result.append(copy.deepcopy([newind1, newind2]))
#                     print("success")
#             except:
#                 backrow = copy.deepcopy(periodic_result[-1])
#                 newind1, newind2 = backrow[0], backrow[1]
#             continue
#
#         except Break as e:
#             print(e)
#
#             print("=====")
#             # Check whether the lists are modified or not.
#             if newind1 != ind1 and newind2 != ind2:
#                 # For safety, though not necessary.
#                 try:
#                     if check_syntax(newind1, dict_operators, dict_data_eval_info) \
#                             and check_syntax(newind2, dict_operators, dict_data_eval_info):
#                         print(f"Final result: {expr_to_str(newind1)}")
#                         print(f"Final result: {expr_to_str(newind2)}")
#                         return newind1, newind2, True
#                 except (TypeError, IndexError):
#                     return ind1, ind2, False
#             else:
#                 exception_printer(0, sys._getframe().f_code.co_name)
#                 return ind1, ind2, False
#     except (IndexError, TypeError):
#         return ind1, ind2, False

