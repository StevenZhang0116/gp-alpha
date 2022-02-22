# 加载自定义包
from base_classes_funcs import *
from grow_tree import *
from optimizer import *

# 加载系统包
# from __future__ import division

class Break(Exception):
    pass

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

def inter(a,b):
    '''
    Check whether the two lists have intersection.
    '''
    return list(set(a) & set(b))

def exchangeMT(ind1, cxpoint, dict_operators, dict_data_eval_info):
    '''
    TODO: 还有一点小问题
    Select two positions at random. Swap the relative cities, like appropriate data or operations.
    Similar to pointMT, but guarantees that no information of chromosome will be lost.
    Also called swap mutation.
    '''
    # print(f"original: {expr_to_str(ind1)}")
    original_ind1 = copy.deepcopy(ind1)
    size = len(ind1)
    random_arr1 = np.random.permutation(size)
    for cxpoint1 in random_arr1:
        tree1 = gp.PrimitiveTree(ind1)
        t1 = ind1[tree1.searchSubtree(cxpoint1)]
        subtree1 = gp.PrimitiveTree(t1)
        try:
            for cxpoint2 in random_arr1:
                if [cxpoint1, cxpoint2] == cxpoint: continue
                if cxpoint2 != cxpoint1:
                    t2 = ind1[tree1.searchSubtree(cxpoint2)]
                    subtree2 = gp.PrimitiveTree(t2)
                    if compare_subtree(subtree1, subtree2, dict_operators, dict_data_eval_info):
                        # 不存在换常数的情况
                        if ind1[cxpoint1] != ind1[cxpoint2] and ind1[cxpoint1].ret != Ephemeral \
                                and ind1[cxpoint1].ret == ind1[cxpoint2].ret:
                            if cxpoint1 > cxpoint2:
                                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                            newind1 = []
                            newind1.extend(ind1[:cxpoint1])
                            newind1.append(ind1[cxpoint2])
                            newind1.extend(ind1[cxpoint1+1:cxpoint2])
                            newind1.append(ind1[cxpoint1])
                            newind1.extend(ind1[cxpoint2+1:])
                            # ind1[cxpoint1], ind1[cxpoint2] = ind1[cxpoint2], ind1[cxpoint1]
                            if check_syntax(newind1, dict_operators, dict_data_eval_info):
                                # 确保PARAM参数(必须对应OP)的位置不会乱掉 check_syntax无法捕获这个异常情况
                                if expr_to_str(newind1) != expr_to_str(original_ind1) \
                                        and type(newind1[1]) == deap.gp.Primitive:
                                    # 保证内存地址不一样
                                    result_ind1 = copy.deepcopy(newind1)
                                    # print(f"after exchange:{expr_to_str(result_ind1)}")
                                    return True, result_ind1, [cxpoint1, cxpoint2]
        except:
            continue
    exception_printer(0, sys._getframe().f_code.co_name)
    return False, original_ind1, cxpoint

def displacementMT(ind1, cxpoint, dict_operators, dict_data_eval_info):
    '''
    Displacement mutation selects a subtour at random and inserts it at a random position outside the subtour.
    Insertion can be viewed as a special case of displacement in which the substring contains only one city.
    '''
    # print(f"original: {expr_to_str(ind1)}")
    original_ind1 = copy.deepcopy(ind1)
    size = len(ind1)
    random_arr1 = np.random.permutation(size)
    for cxpoint1 in random_arr1:
        tree1 = gp.PrimitiveTree(ind1)
        t1 = ind1[tree1.searchSubtree(cxpoint1)]
        subtree1 = gp.PrimitiveTree(t1)
        stamp1 = [i for i in range(cxpoint1, cxpoint1 + len(t1) + 1)]
        # stamp1 = [cxpoint1, cxpoint1 + len(t1)]
        # TODO: 如果catch了attribute error, 如no shape attribute for int, 就说明mutation后的式子不合规范
        for cxpoint2 in random_arr1:
            if [cxpoint1, cxpoint2] == cxpoint: continue
            try:
                # 确保基础属性保持一致，同时保证换的不是常数
                if ind1[cxpoint1] != ind1[cxpoint2] and ind1[cxpoint1].ret != Ephemeral \
                        and ind1[cxpoint1].ret == ind1[cxpoint2].ret:
                    t2 = ind1[tree1.searchSubtree(cxpoint2)]
                    subtree2 = gp.PrimitiveTree(t2)
                    stamp2 = [i for i in range(cxpoint2, cxpoint2 + len(t2) + 1)]
                    # 判断两个list是否存在交集，即两个树是否存在subtree的关系
                    if not inter(stamp1, stamp2):
                        # 确保换的subtree的深度要大于1
                        if compare_subtree(subtree1, subtree2, dict_operators, dict_data_eval_info) \
                                and len(stamp1) > 2 and len(stamp2) > 2:
                            # 修改stamp1和stamp2的相对顺序，调整为原ind1=[blank, stamp1, blank, stamp2, blank]
                            # mutation完以后即为[blank, stamp2, blank, stamp1, blank]
                            if stamp1[0] > stamp2[0]:
                                stamp1, stamp2 = stamp2, stamp1
                            newind1 = [] # create new tree
                            newind1.extend(ind1[:stamp1[0]])
                            newind1.extend(ind1[stamp2[0]:stamp2[-1]])
                            newind1.extend(ind1[stamp1[-1]:stamp2[0]])
                            newind1.extend(ind1[stamp1[0]:stamp1[-1]])
                            newind1.extend(ind1[stamp2[-1]:])
                            if check_syntax(newind1, dict_operators, dict_data_eval_info):
                                if expr_to_str(newind1) != expr_to_str(original_ind1):
                                    result_ind1 = copy.deepcopy(newind1)
                                    return True, result_ind1, [cxpoint1, cxpoint2]
            except:
                continue


    exception_printer(0, sys._getframe().f_code.co_name)
    return True, original_ind1, cxpoint

def hoistMT(ind1, cxpoint, dict_operators, dict_data_eval_info):
    '''
    Perform Hoist permutation to reduce the depth / complexity of the whole tree by removing certain branches / leaves.
    Notice that the smaller subtree is always the subtree of the larger subtree.
    '''
    # print(f"original: {expr_to_str(ind1)}")
    original_ind1 = copy.deepcopy(ind1)
    size = len(ind1)
    random_arr1 = np.random.permutation(size)
    for cxpoint1 in random_arr1:
        tree1 = gp.PrimitiveTree(ind1)
        t1 = ind1[tree1.searchSubtree(cxpoint1)]
        subtree1 = gp.PrimitiveTree(t1) # larger subtree
        stamp1 = [cxpoint1, cxpoint1 + len(t1)]
        random_arr2 = np.random.permutation(stamp1[1] - stamp1[0])
        for cxpoint2 in random_arr2:
            if [cxpoint1, cxpoint2] == cxpoint: continue
            try:
                cxpoint2 += stamp1[0]
                t2 = ind1[tree1.searchSubtree(cxpoint2)]
                subtree2 = gp.PrimitiveTree(t2) # smaller subtree
                stamp2 = [cxpoint2, cxpoint2 + len(t2)]
                if compare_subtree(subtree1, subtree2, dict_operators, dict_data_eval_info):
                    newind1 = [] # create new tree
                    newind1.extend(ind1[:stamp1[0]])
                    newind1.extend(ind1[stamp2[0]: stamp2[1]])
                    newind1.extend(ind1[stamp1[1]:])
                    if check_syntax(newind1, dict_operators, dict_data_eval_info):
                        if expr_to_str(newind1) != expr_to_str(original_ind1):
                            # print(f"Final result: {expr_to_str(newind1)}")
                            result_ind1 = copy.deepcopy(newind1)
                            return True, result_ind1, [cxpoint1, cxpoint2]
                        else:
                            continue
            except:
                continue

    exception_printer(0, sys._getframe().f_code.co_name)
    return True, original_ind1, cxpoint

def manyHoist(ind1, cxpoint, num, dict_operators, dict_data_eval_info):
    '''
    Execute hoist mutation for several times since it is highly manageable.
    '''
    original_ind = copy.deepcopy(ind1)
    for i in range(num):
        newind1 = hoistMT(ind1, dict_operators, dict_data_eval_info, cxpoint)[1]
        ind1 = newind1
        # print(f"{i+1}th time of hoist: {expr_to_str(ind1)}")
    try:
        if check_syntax(ind1, dict_operators, dict_data_eval_info):
            if ind1 != original_ind:
                # print(f"final: {expr_to_str(ind1)}")
                result_ind1 = copy.deepcopy(ind1)
                return True, result_ind1, None
    except (TypeError, IndexError):
        return False, original_ind, None


def subtreeMT(ind1, cxpoint, minHeight, maxHeight, attemptTime, pset, dict_operators, dict_data_eval_info,
              data_classified_by_units, n_units, data_type, dict_params, f_init_a_tree, frequent_subtrees=None):
    '''
    Generate random subtree and replace it to the arbitrary subtree of ind1.
    The depth range is set as hyper-parameter. Its optimization could be accomplished later.
    By definition, the whole new subtree will be used in mutation, ow the function is similar to basicCO()
    '''
    # print(f"original: {expr_to_str(ind1)}")
    original_ind1 = copy.deepcopy(ind1)

    size = len(ind1)
    for i in range(attemptTime):
        # TODO: return type is array2D so that the ops operator will not be shown at the outermost shell
        if frequent_subtrees == None:
            ind2 = gen_alpha_tree(pset, minHeight, maxHeight, Array2d, data_classified_by_units, n_units,
                                  data_type, dict_operators, dict_data_eval_info)
            individual = Individual(ind2)
            f_init_a_tree(individual, dict_operators=dict_operators, dict_params=dict_params)
        else:
            ind2 = random.choice(frequent_subtrees)
        # new_subtree = gp.PrimitiveTree(ind2)
        random_arr1 = np.random.permutation(size) # cannot be the root node
        for cxpoint in random_arr1: # find the cutting point of ind1 subtree.
            try:
                result = compare(ind1, ind2, cxpoint, 0, dict_operators, dict_data_eval_info)
                if result[0]:
                    stamp1 = [cxpoint, cxpoint + result[1]]
                    newind1= [] # create new tree
                    newind1.extend(ind1[:stamp1[0]])
                    newind1.extend(ind2[0:])
                    newind1.extend(ind1[stamp1[1]:])
                    if check_syntax(newind1, dict_operators, dict_data_eval_info):
                        if expr_to_str(newind1) != expr_to_str(original_ind1):
                            result_ind1 = copy.deepcopy(newind1)
                            return True, result_ind1, None
            except:
                continue

    exception_printer(0, sys._getframe().f_code.co_name)
    return False, original_ind1, None

def pointMT(ind1, cxpoint, pset, replace_num, dict_operators, dict_data_eval_info, data_classified_by_units, dict_params):
    '''
    Aggressive point mutation. Guarantee the replaced nodes(terminals/primitives) have the same shape/unit as previous.
    Number of replace_num nodes will be randomly replaced.
    TODO: 注意如果expr_to_str()打印出的树的表达式不正确，即为树的结构出现问题，如换node时没有正确比对input/output的格式。
    TODO: 目前是存在换不了点的情况的，即选出某个候选node发现无格式相同的点可供替换，比如说Volume, Return的terminal。
    '''

    # 包装成dict格式
    data_classified_by_units_dict = {}
    for i in range(len(data_classified_by_units)):
        data_classified_by_units_dict[i] = data_classified_by_units[i]

    # print(f"original: {expr_to_str(ind1)}")
    original_ind1 = copy.deepcopy(ind1)
    size = len(ind1)
    random_arr1 = np.random.permutation(size)
    allTree = gp.PrimitiveTree(ind1)
    # 存储阶段性的结果。在每一次成功的点变异后更新。如果失败则回溯到上一阶段的结果。
    stack = [original_ind1]

    # 确保所有对于ind1的修改不是inplace的
    newind1 = copy.deepcopy(ind1)

    for i in range(min(replace_num, round(size / 2))):
        replace_node = newind1[random_arr1[i]]
        # print(f"replace_node: {replace_node.name, type(replace_node)}")
        selective = (pset.primitives, data_classified_by_units_dict)

        # # TODO: 换operator内常数
        # if replace_node.ret == Ephemeral:
        #     back_count = collections.deque([])
        #     back_count.append(replace_node)
        #     index = random_arr1[i]
        #     for prev_ind in range(index - 1, -1, -1):
        #         node_iter = newind1[prev_ind]
        #         back_count.appendleft(node_iter)
        #         if isinstance(node_iter, deap.gp.Primitive):
        #             prop = dict_operators[node_iter.name]
        #             input_size = len(prop[1])
        #             # 利用常数会是该operator subtree的列表中最后一个的特性
        #             if len(back_count) == input_size + 1:
        #                 match_operator = back_count[0]
        #                 break
        #             else:
        #                 for t in range(input_size + 1):
        #                     back_count.popleft()
        #                 back_count.appendleft('operator')
        #     param = list(dict_params[match_operator.name][0])
        #     param.remove(replace_node.value)
        #     newind1[random_arr1[i]].value = random.choice(param)

        # TODO: 换terminal或者operator
        if replace_node.ret != Ephemeral:
            try:
                for pick in selective:
                    for key in pick.keys():
                        if replace_node in pick[key]:
                            backup = copy.deepcopy(pick[key])
                            backup.remove(replace_node)
                            try:
                                # 20 consecutive attempts.
                                for rand in range(20):
                                    new_node = random.choice(backup)
                                    # print(f"new_node: {new_node.name}")
                                    # it is a terminal node or ephemeral. ARGx.
                                    if isinstance(replace_node, deap.gp.Terminal):
                                        newind1[random_arr1[i]] = new_node
                                        raise Break('Find a terminal')
                                    # otherwise check their input/output formats are consistent or not.
                                    else:
                                        (prop1, prop2) = (dict_operators[replace_node.name], dict_operators[new_node.name])
                                        # check whether the input/output format is consistent.
                                        if prop1[1:] == prop2[1:]:
                                            newind1[random_arr1[i]] = new_node
                                            #TODO: 这里同时涉及到替换operator后修改伴随常数
                                            try:
                                                index = random_arr1[i]
                                                #TODO: 目前这么写仅处理了一个伴随参数的情况。其位置处于以operator为根节点的subtree组成的list的最后一位。
                                                replaced_constant = newind1[allTree.searchSubtree(index)][-1]
                                                # print(f"replaced: {replaced_constant.value}")
                                                val_range = list(dict_params[new_node.name][0])
                                                # print(f"val_range: {val_range}")
                                                if replaced_constant.value in val_range:
                                                    # print('Already in it!')
                                                    raise Break('Find one')
                                                else:
                                                    replaced_constant.value = random.choice(val_range)
                                                    # print(f"new: {replaced_constant.value}")
                                                    raise Break('Find one')
                                            # 意味着替换的operator和原本的operator均没有额外参数设定，如abs。这样可以直接换。
                                            except KeyError:
                                                # print('Cannot change!')
                                                raise Break('Find one')
                                        else:
                                            # print('Try again')
                                            continue
                            except Break:
                                continue
            except IndexError:
                continue
        if check_syntax(newind1, dict_operators, dict_data_eval_info):
            # print(f"period:{expr_to_str(ind1)}")
            # store this period result
            copy1 = copy.deepcopy(newind1)
            stack.append(copy1)
        else:
            # go back to last periodic value.
            newind1 = copy.deepcopy(stack[-1])

    if expr_to_str(newind1) == expr_to_str(original_ind1):
        exception_printer(0, sys._getframe().f_code.co_name)
        return False, original_ind1, None
    if check_syntax(newind1, dict_operators, dict_data_eval_info):
        result_ind1 = copy.deepcopy(newind1)
        return True, result_ind1, None



