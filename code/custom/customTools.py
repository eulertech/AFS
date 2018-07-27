import deap
import random
import sys
import numpy as np
import pandas as pd
from deap import gp

from global_definitions import AFS_HOME, DATA_DIR, GLOBAL_DEBUG_LEVEL, my_timer, debug_pr, tic, toc
from model_validation.cross_validation_output import CrossValidationOutput

# using this equivalence operator, we can determine whether a feature already belongs to a set
# passing this to the HallOfFame object ensures that features aren't added twice
def equals(individual1, individual2):
    return str(individual1) == str(individual2)


def compute_correlation(col1, col2):
    '''
    Method to return the correlation coefficient between two columns (with no checking for data type)
    @param col1 - The first column as a pd.Series or np columns array
    @param col2 - The second column as a pd.Series or np columns array

    '''
    corr_coef = fast_correlation(col1, col2)
    return corr_coef if np.isfinite(corr_coef) else 0

def fast_correlation(col1, col2):
    c1 = col1-np.mean(col1)
    c2 = col2-np.mean(col2)
    dotprod11 = np.dot(c1,c1.T)
    dotprod22 = np.dot(c2,c2.T)
    dotprod12 = np.dot(c1,c2.T)
    if(dotprod11*dotprod22 == 0):
        return 0
    corr_coeff = abs(dotprod12/np.sqrt(dotprod11*dotprod22))
    return corr_coeff

def get_label_encoded_col(categorical_feature, afs_tool):
    encoded_feature_str = 'encode_labels('+categorical_feature+')'
    return afs_tool.get_feature_col(encoded_feature_str)


def get_feature_correlation(individual1, individual2, afs_tool):
    str1 = str(individual1)
    str2 = str(individual2)
    if(str1 == str2):
        return 1.0
    elif(str1 < str2):
        key = 'corr<{},{}>'.format(str1, str2)
    else:
        key = 'corr<{},{}>'.format(str2, str1)
    if key in afs_tool.data_eval_dict:
        return afs_tool.data_eval_dict[key]

    if (str1 == afs_tool.target_name):
        col1 = afs_tool.target
    else:
        col1 = afs_tool.get_feature_col(str1)

    if (str2 == afs_tool.target_name):
        col2 = afs_tool.target
    else:
        col2 = afs_tool.get_feature_col(str2)

    if(type(col1) is np.ndarray):
        type_1 = type(col1[0])
    else:
        type_1 = type(col1.values[0])
    if(type(col2) is np.ndarray):
        type_2 = type(col2[0])
    else:
        type_2 = type(col2.values[0])
    if issubclass(type_1, bool) or issubclass(type_1, np.bool_):
        col1 = col1.astype(float)
    elif not issubclass(type_1, np.number):
        if(pd.Series(col1).nunique() > 200):
            return 0
        col1 = get_label_encoded_col(str1,afs_tool)
    if issubclass(type_2, bool) or issubclass(type_2, np.bool_):
        col2 = col2.astype(float)
    elif not issubclass(type_2, np.number):
        if(pd.Series(col1).nunique() > 200):
            return 0
        col2 = get_label_encoded_col(str2,afs_tool)
    
    corr_coef = fast_correlation(col1, col2)

    if not np.isfinite(corr_coef):
        corr_coef = 0
    afs_tool.data_eval_dict[key] = corr_coef
    return corr_coef

######################################
# GP Crossovers                      #
######################################

def cxJoinWithRandomOp(ind1, ind2, pset):
    """Combine two subtrees.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of one tree. If ind1 and ind2 can be combined
     by a valid operator, that result is returned. If they cannot be
     combined, ind1 is returned.
    """

    #TODO if commutative, swap (if necessary) to make ind1 come first lexographically

    index = 0 # the root index
    for idx, node in enumerate(ind1[index:(index+1)], 1):
        node1 = node
    for idx, node in enumerate(ind2[index:(index+1)], 1):
        node2 = node
    choice = random.choice
    type_ = pset.ret
    if type(type_) is list:
        primitiveList = []
        for type_element in type_:
            for primitive in pset.primitives[type_element]:
                primitiveList.append(primitive)
    else:
        primitiveList = pset.primitives[type_]
    primitives = [p for p in primitiveList if (p.arity == 2 and node1.ret is p.args[0] and node2.ret is p.args[1])]
    if len(primitives) == 0:
        debug_pr('No match for argument list '+str(node1.ret)+', '+str(node2.ret),3)
        return ind1, ind1
    new_node = choice(primitives)
    try:
        new_string1 = new_node.format(ind1, ind2)
        Individual = type(ind1)
        new_individual1 = Individual(gp.PrimitiveTree.from_string(new_string1, pset))
        new_string2 = new_node.format(ind1, ind2)
        new_individual2 = Individual(gp.PrimitiveTree.from_string(new_string2, pset))
        return new_individual1, new_individual2
    except TypeError as inst:
        debug_pr("Exception creating gp.PrimitiveTree from String!",1)
        debug_pr(inst,1)
        return ind1, ind1

def apply_expression(expression, afs_tool, df=None, eval_dict=None):
    if (eval_dict is None):
        eval_dict = afs_tool.data_eval_dict
    if (df is None):
        df = afs_tool.data
    if(expression in eval_dict):
        return eval_dict[expression]
    oper,operand1,operand2 = decompose(expression)
    if(operand1 is None):
        raise ValueError("Bad expression: "+expression)
    func = eval(oper, afs_tool.pset.context)
    if(operand2 is None):
        return func(afs_tool.get_feature_col(operand1, df=df, eval_dict=eval_dict))
    col1 = pd.Series(afs_tool.get_feature_col(operand1, df=df, eval_dict=eval_dict))
    col2 = pd.Series(afs_tool.get_feature_col(operand2, df=df, eval_dict=eval_dict))
    ret_val = func(col1, col2)
    return ret_val

def decompose(expression):
    """
    Decomposes a feature string of format AAA(BBB, CCC)
     into the tuple AAA,BBB,CCC

    Decomposes a feature string of format AAA(BBB)
     into the tuple AAA,BBB,None

    Follows syntax rules such that BBB and CCC can also be feature strings
     in the same format

    Returns tuple of expression terms
    """
    first_open_ind = expression.find('(')
    if(first_open_ind==-1):
        return (expression,None,None)
    AAA = expression[:(first_open_ind)]
    expression = expression[(first_open_ind+1):]
    term_start = 0
    curr_index = term_start
    paren_depth = 1
    BBB=None
    CCC=None
    while (paren_depth > 0):
        open_ind = expression.find('(', curr_index)
        close_ind = expression.find(')', curr_index)
        comma_ind = expression.find(',', curr_index)
        min_pos_ind = min_positive(open_ind, close_ind, comma_ind)
        if (min_pos_ind<0 or min_pos_ind == np.inf):
            raise ValueError('Bad expression format!')
        min_pos_ind = int(min_pos_ind)
        if (open_ind==min_pos_ind): # ( comes first
            paren_depth += 1
        elif (close_ind==min_pos_ind): # ( comes first
            paren_depth -= 1
            if(paren_depth == 0):
                if(BBB == None):
                    BBB = expression[term_start:(min_pos_ind)]
                    term_start = min_pos_ind+1
                else:
                    CCC = expression[term_start:(min_pos_ind)]
        elif (paren_depth==1 and comma_ind==min_pos_ind):
            BBB = expression[term_start:(min_pos_ind)]
            term_start = min_pos_ind+1
        curr_index = min_pos_ind+1
    return (AAA,BBB,CCC)


def min_positive(a,b,c):
    if(a<0):
        a=np.inf
    if(b<0):
        b=np.inf
    if(c<0):
        c=np.inf
    return np.min([a,b,c])