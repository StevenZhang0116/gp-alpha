# 加载自定义包
from base_classes_funcs import *
from operators import *


def gen_alpha_tree(pset, min_, max_, output_type, data_classified_by_units, n_units,
                   data_type, dict_operators, dict_data_eval_info, patience=1000) -> list:
    """generates a legal alpha syntax tree.
    :param patience: number of maximum failures before regenerate. -1 means try until exhaustion. """
    while True:
        if data_type == Array3d:
            expr = gen_alpha_tree_structure_from_3d(pset,
                                                    min_,
                                                    max_,
                                                    dict_operators,
                                                    output_type)
            if alternate_leaves(expr,
                                patience,
                                data_classified_by_units,
                                n_units,
                                data_type,
                                dict_operators,
                                dict_data_eval_info):
                break
        elif data_type == Array2d:
            expr = gen_alpha_tree_structure_from_2d(pset,
                                                    min_,
                                                    max_,
                                                    dict_operators,
                                                    output_type)
            if alternate_leaves(expr,
                                patience,
                                data_classified_by_units,
                                n_units,
                                data_type,
                                dict_operators,
                                dict_data_eval_info):
                break
    return expr


def gen_alpha_tree_structure_from_3d(pset, min_, max_, dict_operators, output_type) -> list:
    """
    Generates an alpha syntax tree structure. None of the existing gen funcs meets
    our requirements, thus we write our own. Specifically, all DEAP funcs randomly
    chooses a max depth. However, we cannot force end the growth of a tree since
    we only have terminals of type Array3d and Ephemeral. See def condition()
    for more details.

    :param pset: Primitive set from which primitives are selected.
    :param output_type: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(max_, depth):
        """
        Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.

        Thats what this function does in vanilla DEAP. Here we modify it,
        to indicate whether we should 'advance' to the next 'layer' - that is,
        to force the input be a func of target input array_3d instead of
        possible array_2d (if our target input is array_2d) or to force the
        input be one of the several (currently 7) raw data fields (if our
        target input is array_3d). The probability that we would like to do so
        increases with the depth of the current branch, and reaches 1.0 once it
        surpasses the max depth. Thus we are still able to control the max depth,
        though not as precisely. Also, if condition() returns False, that means
        we would strcitly wish to stay in the current 'layer'.

        Note: for implementation, we do not actually limit the input types to
        array_3d, instead we limit them to have no array_2d in them.
        """
        # raw_prob = 1 - np.exp(-depth / max_ * 2)
        raw_prob = 2 / max_  # uniform
        # transform raw_prob such that (1) depth >= max_ - 1 always retuns 1 (2) depth < min_ - 1 always returns 0
        return np.random.uniform() < min(max(raw_prob, depth >= max_ - 1),
                                         depth >= min_ - 1)  # increasing prob, lambda=1, x=depth/max*2

    # Generate a Tree as a list of expr, via DFS. The tree is built from the root to the leaves.
    if output_type is None:
        output_type = pset.ret
    expr = []
    stack = [(0, output_type)]
    while len(stack) != 0:
        depth, output_type = stack.pop()
        """
        output_type has several choices: FinalResult (for the root node), Arrray2d (for the next layer), 
        Array3d (final layer), Ephemeral.
        """
        prim = None  # shorthand for primitive
        term = None  # shorthand for terminal
        ############################
        try:
            if output_type == FinalResult:
                prim = random.choice(pset.primitives[output_type])
            elif output_type == Array2d:
                # 这里先简单处理: 如果当前算子的要求输入类型(output_type)是Array2d, 直接强制进入下一层, 即daily最多之有一层
                candidates = pset.primitives[output_type]
                # if output_type not in input types
                candidates = [c for c in candidates if output_type not in dict_operators[c.name][1]]
                prim = random.choice(candidates)  # choose one random terminal of the specified type
            elif output_type == Array3d:
                if condition(max_, depth):
                    # search in terminals. the only terminals with array_3d as output types are data.
                    term = random.choice(pset.terminals[output_type])
                else:
                    prim = random.choice(pset.primitives[output_type])
            else:  # Ephemeral, Array2dNeutralise, Array2dValid
                term = random.choice(pset.terminals[output_type])
        except IndexError:
            _, _, traceback = sys.exc_info()
            raise IndexError("gen_alpha_tree_structure_from_3d tries to add a {} of type '{}', but there is none"
                             " available.".format(['primitive', 'terminal'][prim is None], output_type))

        if (prim is None) and (term is not None):
            # inspect.isclass() returns True if the object is a class, whether built-in or created in Python code.
            if inspect.isclass(term):
                term = term()
            expr.append(term)
            # the tree doesn't grow
        elif (prim is not None) and (term is None):
            expr.append(prim)
            for arg in reversed(prim.args):  # add all input_types to the stack
                stack.append((depth + 1, arg))  # the tree grows!
        else: # this should not happen
            print("Error in gen_alpha_tree_structure_from_3d")

    return expr


def gen_alpha_tree_structure_from_2d(pset, min_, max_, dict_operators, output_type) -> list:
    """similar to gen_alpha_tree_structure_from_3d, except that terminals are now Array2d."""

    def condition(max_, depth):
        raw_prob = 2 / max_  # uniform
        return np.random.uniform() < min(max(raw_prob, depth >= max_ - 1), depth >= min_ - 1)

    if output_type is None:
        output_type = pset.ret
    expr = []
    stack = [(0, output_type)]
    while len(stack) != 0:
        depth, output_type = stack.pop()
        """
        output_type has several choices: FinalResult (for the root node), Arrray2d (for the next layer), 
        Array3d (final layer), Ephemeral.
        """
        prim = None  # shorthand for primitive
        term = None  # shorthand for terminal
        ############################
        try:
            if output_type == FinalResult:
                prim = random.choice(pset.primitives[output_type])
            elif output_type == Array2d:  # if we wish to proceed to the next layer
                if condition(max_, depth):
                    term = random.choice(pset.terminals[output_type])
                else:
                    candidates = pset.primitives[output_type]
                    candidates = [c for c in candidates if output_type in dict_operators[c.name][1]]
                    prim = random.choice(candidates)
            else:  # Ephemeral, Array2dNeutralise, Array2dValid
                term = random.choice(pset.terminals[output_type])
        except IndexError:
            _, _, traceback = sys.exc_info()
            raise IndexError("gen_alpha_tree_structure_from_2d tries to add a {} of type '{}', but there is none"
                             " available.".format(['primitive', 'terminal'][prim is None], output_type))

        if (prim is None) and (term is not None):
            if inspect.isclass(term):
                term = term()
            expr.append(term)
            # the tree doesn't grow
        elif (prim is not None) and (term is None):
            expr.append(prim)
            for arg in reversed(prim.args):  # add all input_types to the stack
                stack.append((depth + 1, arg))  # the tree grows!
        else: # this should not happen
            print("Error in gen_alpha_tree_structure_from_2d")

    return expr


def alternate_leaves(expr, patience, data_classified_by_units, n_units, data_type, dict_operators, dict_data_eval_info):
    """the final step of generating an Alpha tree. A randomly generated tree most likely has illegal syntax
    (checkSyntax(Alpha) == False), but by changing its leaf nodes, chances are we can get a legal syntax tree.
    :param expr: input alpha tree, a list of deap.gp.xx, likely from genAlphaTreeStructure.
    :param data_type: Array2d or Array3d
    returns: True if successful (expr is changed in-place, no need to return it)

    每个叶结点有len(data_classified_by_units)种量纲组(unit group)选择.
    我们先选择量纲组(每个量纲组被data_classified_by_units中第一个terminal代表).
    量纲组的排列是随机的, 但总共有len(data_classified_by_units) ** |slots|种组合.
    我们可以建立一个该范围内整数与组合的双射关系, 如当|slots| = 3, len(data_classified_by_units) = 3时
    0-000, 1-001, 2-002, 3-010, 4-011, 5-012, 6-021, ... 25-221, 26-222
    其中, 2-002表示整数2可以映射到三个槽位为(量纲组0, 量纲组0, 量纲组2)的组合.
    """

    slots = [i for i in range(len(expr)) if (expr[i].arity == 0 and expr[i].ret == data_type)]
    n_unit_groups = len(data_classified_by_units)
    N = n_unit_groups ** len(slots)
    if N >= 3 ** 10:  # too large. 注意对任意N, 理论上来说都可以用加密算法达到O(logN)空间复杂度, 但十分复杂.
        return False
    index_permutation = list(range(N))
    random.shuffle(index_permutation)
    for index in index_permutation:
        patience -= 1
        if patience == 0:
            return False
        # 将10进制转为n_unit_groups进制的表示, 共len(slots)位, 结果即为每个slot的单位.
        remainder = index
        for j in range(len(slots) - 1, -1, -1):
            denom = n_unit_groups ** j
            expr[slots[j]] = np.random.choice(data_classified_by_units[remainder // denom])  # change leaf nodes
            remainder = remainder % denom
        if check_syntax(expr, dict_operators, dict_data_eval_info):
            return True
    return False

