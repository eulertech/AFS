from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import copy
from afs_gp.afsTool import AFSTool

def get_computed_stat(seriesCol, stat_type, group_key=None):
    column_key = seriesCol.name
    if (column_key in AFSTool.data_stats_dict):
        col_stat_dict = AFSTool.data_stats_dict[column_key]
    else:
        col_stat_dict = dict()
        AFSTool.data_stats_dict[column_key] = col_stat_dict
    if (stat_type == 'quantiles'):
        stat_type = 'quantiles_{}'.format(group_key)
    if (stat_type in col_stat_dict):
        return col_stat_dict[stat_type]
    if ('quantiles' in stat_type):
        num_quantiles = group_key
        quantiles = [x * 100.0 / num_quantiles for x in range(1, num_quantiles)]
        col_stat_dict[stat_type] = np.percentile(seriesCol,quantiles)
        return col_stat_dict[stat_type]
    if (stat_type == 'value_counts'):
        col_stat_dict['value_counts'] = seriesCol.value_counts(sort=True, ascending=False, dropna=False)
        return col_stat_dict['value_counts']
    if (stat_type == 'mean'):
        col_stat_dict['mean'] = seriesCol.mean()
        return col_stat_dict['mean']
    if (stat_type == 'std'):
        col_stat_dict['std'] = seriesCol.std()
        return col_stat_dict['std']
    if (stat_type == 'mode'):
        col_stat_dict['mode'] = seriesCol.mode()
        return col_stat_dict['mode']
    if (stat_type == 'groupby'):
        # record entire column stats (to be used as default)
        col_stat_dict['mean'] = seriesCol.mean()
        col_stat_dict['std'] = seriesCol.std()
        col_stat_dict['min'] = seriesCol.min()
        col_stat_dict['max'] = seriesCol.max()
        col_stat_dict['mode'] = seriesCol.mode()
        # record the groupby object
        grouped = seriesCol.groupby(group_key)
        grouped_stat_dict = dict()
        grouped_stat_dict['mean'] = grouped.agg([np.mean])
        grouped_stat_dict['std'] = grouped.agg([np.std])
        grouped_stat_dict['min'] = grouped.agg([np.min])
        grouped_stat_dict['max'] = grouped.agg([np.max])
        col_stat_dict['groupby'] = grouped_stat_dict
        return col_stat_dict['groupby']
    raise KeyError('Requested stat_type {} is invalid.'.format(stat_type))

def single_row_zscore_by_group(x, y, grouped_stat_dict, default_mean, default_std):
    group_std = grouped_stat_dict['std']
    group_mean = grouped_stat_dict['mean']
    if (y in group_std.index):
        mean_ = (group_mean.loc[y])['mean']
        std_ = (group_std.loc[y])['std']
        return (x-mean_)/std_
    else:
        return (x-default_mean)/default_std

def protected_div(x, y):
    return 1 if y == 0 else x / y

def categorical_concat(x, y):
    return x + y

def z_score_by_group(x,y):
    grouped_stat_dict = get_computed_stat(x, 'groupby', y)
    default_mean = get_computed_stat(x, 'mean')
    default_std = get_computed_stat(x, 'std')
    ret=copy.deepcopy(x)
    categories = grouped_stat_dict['std'].index
    for cat in categories:
        ind = (y==cat)
        ret[ind] = (x.loc[ind] - (grouped_stat_dict['mean'].loc[cat])['mean']) \
                    / (grouped_stat_dict['std'].loc[cat])['std']
    ind = ~(y.isin(categories))
    ret[ind] = (x.loc[ind] - default_mean) / default_std
    return ret

def z_score_by_yr_mo(x,y):
    yr_mo = lambda x: str(x.year) + '-' + str(x.month).zfill(2)
    return z_score_by_group(x, y.apply(yr_mo))

def max_by_group(x,y):
    # return x.groupby(y).transform(max)
    grouped_stat_dict = get_computed_stat(x, 'groupby', y)
    default_max = get_computed_stat(x, 'max')
    ret=copy.deepcopy(x)
    categories = grouped_stat_dict['max'].index
    for cat in categories:
        ind = (y==cat)
        ret[ind] = (grouped_stat_dict['max'].loc[cat])['amax']
    ind = ~(y.isin(categories))
    ret[ind] = default_max
    return ret

def min_by_group(x,y):
    # return x.groupby(y).transform(min)
    grouped_stat_dict = get_computed_stat(x, 'groupby', y)
    default_min = get_computed_stat(x, 'min')
    ret=copy.deepcopy(x)
    categories = grouped_stat_dict['min'].index
    for cat in categories:
        ind = (y==cat)
        ret[ind] = (grouped_stat_dict['min'].loc[cat])['amin']
    ind = ~(y.isin(categories))
    ret[ind] = default_min
    return ret

def date_difference(x,y):
    return (x - y).dt.days

def encode_labels(x):
    value_counts = get_computed_stat(x, 'value_counts')
    default_label = 0
    ret = pd.Series(np.zeros(shape=x.shape))
    ret.index = x.index.copy()
    categories = value_counts.index
    for cat in categories:
        ind = (x==cat)
        ind.index = ret.index.copy()
        ret[ind] = value_counts.index.tolist().index(cat)
    ind = ~(x.isin(categories))
    ind.index = ret.index.copy()
    ret[ind] = default_label
    return ret

def protected_numeric_mode_flag(x):
    mode_ = get_computed_stat(x, 'mode')
    if len(mode_) == 0:
        return x!=x
    else:
        return x==x.mode().values[0]

def conjuction(x,y):
    return x & y

def disjuction(x,y):
    return x | y

def negation(x):
    return ~x

def bin_quantiles(x, n_quantiles):
    quantiles = get_computed_stat(x, 'quantiles', n_quantiles)
    ret = copy.deepcopy(x)
    ret[x < quantiles[0]] = '{}of{}'.format(0, n_quantiles)
    for i in range(0,n_quantiles-1):
        ret[x >= quantiles[i]] = '{}of{}'.format(i+1, n_quantiles)
    return ret

def bin_cat_quantiles(x, n_quantiles):
    return bin_quantiles(encode_labels(x), n_quantiles)

def bin_4(x):
    return bin_quantiles(x, 4)

def bin_10(x):
    return bin_quantiles(x, 10)

def cat_bin_4(x):
    return bin_cat_quantiles(x, 4)

def cat_bin_10(x):
    return bin_cat_quantiles(x, 10)

# # the code below doesn't belong here- just used to show typing
# # binary/unary numeric transforms
# pset.addPrimitive(operator.add,[float,float],float)
# pset.addPrimitive(operator.mul,[float,float],float)
# pset.addPrimitive(operator.sub,[float,float],float)
# pset.addPrimitive(protected_div,[float,float],float)
# pset.addPrimitive(np.exp, [float], float)
# pset.addPrimitive(np.log, [float], float)
# pset.addPrimitive(lambda x: x.pow(2), [float],float)
# pset.addPrimitive(lambda x: x.pow(0.5), [float],float)

# # categorical transforms
# pset.addPrimitive(categorical_concat,[str,str],str)
# pset.addPrimitive(encode_labels,[str],float)
# pset.addPrimitive(bin_10,[float],str)
# pset.addPrimitive(cat_bin_4,[str],str)

# # by-group transforms
# pset.addPrimitive(z_score_by_group,[float,str],float)
# pset.addPrimitive(z_score_by_yr_mo,[float,np.datetime64],float)
# pset.addPrimitive(max_by_group,[float,str],float)
# pset.addPrimitive(min_by_group,[float,str],float)

# # dates
# pset.addPrimitive(date_difference,[np.datetime64,np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.dayofyear, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.month, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.year, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.weekofyear, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.weekday, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.dayofyear, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.quarter, [np.datetime64],float)
# pset.addPrimitive(lambda x: x.dt.is_month_end, [np.datetime64],bool)
# pset.addPrimitive(lambda x: x.dt.is_month_start, [np.datetime64],bool)


# # flags
# pset.addPrimitive(lambda x: x==0, [float],bool)
# pset.addPrimitive(lambda x: x==x.min(), [float],bool)
# pset.addPrimitive(lambda x: x==x.max(), [float],bool)
# pset.addPrimitive(lambda x: x==x.mode().values[0],[str],bool)
# pset.addPrimitive(protected_numeric_mode_flag,[float],bool)
# pset.addPrimitive(conjuction,[bool,bool],bool)
# pset.addPrimitive(disjuction,[bool,bool],bool)
# pset.addPrimitive(negation,[bool],bool)

# #TODO:
# # binning by volume/interval
# # truncating continuouse, lump rare categories.