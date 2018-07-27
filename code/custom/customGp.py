import functools
import random
from deap import gp

import copy
import math
import random
import re
import sys
import warnings

from collections import defaultdict, deque
from functools import partial, wraps
from inspect import isclass
from operator import eq, lt

from global_definitions import AFS_HOME, DATA_DIR, GLOBAL_DEBUG_LEVEL, my_timer, debug_pr, tic, toc

def compile(expr, pset):
    """Compile the expression *expr*.
    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    print("GP.compile")
    tic()
    code = str(expr)
    toc(5)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        toc(5)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
        toc(5)
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError, ("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
"DEAP will now abort."), traceback

def genHalfAndHalfTyped(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrowTyped, genFullTyped))
    return method(pset, min_, max_, type_)

def genFullTyped(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth
    between *min* and *max*.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height
    return generateTyped(pset, min_, max_, condition, type_)


def genGrowTyped(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)
    return generateTyped(pset, min_, max_, condition, type_)

def generateTyped(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                # when the type_ specified is a list, choose one terminal from among all the types
                if type(type_) is list:
                    terminalList = []
                    for type_element in type_:
                        for terminal in pset.terminals[type_element]:
                            terminalList.append(terminal)
                    term = random.choice(terminalList)
                else:
                    term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError, "The gp.generate function tried to add "\
                                  "a terminal of type '%s', but there is "\
                                  "none available." % (type_,), traceback
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                # when the type_ specified is a list, choose one primitive from among all the types
                if type(type_) is list:
                    primitiveList = []
                    for type_element in type_:
                        for primitive in pset.primitives[type_element]:
                            primitiveList.append(primitive)
                    prim = random.choice(primitiveList)
                else:
                    prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError, "The gp.generate function tried to add "\
                                  "a primitive of type '%s', but there is "\
                                  "none available." % (type_,), traceback
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def full_uniform_mutation(individual, pset, min_, max_):
    expr = functools.partial(genFullTyped, min_=min_, max_=max_)
    return gp.mutUniform(individual, expr=expr, pset=pset)