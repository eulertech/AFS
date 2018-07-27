import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

def get_file_names(pop_dir, base_name):
    """
    Gets all the filenames in pop_dir where base_name is a substring. 
    Meant for getting pickled files from AFS tool. 
    
    Parameters 
    -----------------
    pop_dir (str): path to directory where the files are stored.
    base_name (str): substring to search for in pop_dir.  Also look for .p ending. 
    
    Returns 
    -----------------
    file_names (list of str): list with file names in pop_dir 
                                containing base_name.
    """
    
    file_names = []
    for item in os.listdir(pop_dir):
        if re.search(base_name + '.*.p$',item):
        #if base_name in item:
            file_names.append(item)
    file_names.sort()
    return file_names

def convert_names(original_names, column_names):
    """
    Converts the variable names in a deap tree to more meaningful ones
    using the actual variable names from the data.
    
    Parameters 
    -----------------
    original_names (list of str): string representations of deap trees, 
                                    e.g. MUL (ARG 19, ARG 18).
    column_names (list of str): list of column names from dataframe.
    
    Returns 
    -----------------
    List where variables in original names have been replaced by the corresponding
    names in the dataframe.
    """
    convert_names_dict={}
    for i in range(0, len(column_names)):
        convert_names_dict['ARG' + str(i)] = column_names[i]
    pattern = re.compile(r'\b(' + '|'.join(convert_names_dict.keys()) + r')\b')
    if type(original_names)==str:
        return pattern.sub(lambda x: convert_names_dict[x.group()],original_names)
    else:
        return [pattern.sub(lambda x: convert_names_dict[x.group()], item)
                for item in original_names]
    

def get_individual_fitness(population, column_names):
    """
    Gets the fitness for each individual in a population.
    
    Parameters 
    -----------------
    population (deap population): deap population.
    column_names (list of str): list of actual column names.
    
    Returns 
    -----------------
    Pandas dataframe with two columns 'feature' containing a description 
    of the created feature and 'fitness', its fitness value.
    """
    
    df_dict = {}
    df_dict['fitness'] = [x.fitness.values[0] for x in population]
    original_names = [str(x) for x in population]
    df_dict['feature'] = convert_names(original_names, column_names)
    
    df = pd.DataFrame(df_dict)
    #df.set_index('feature', inplace=True)
    df.sort_values('fitness', inplace=True, ascending=False)
    df.drop_duplicates(inplace=True)
    return df
    
def create_combined_df(file_names, pop_dir, column_names=[]):
    """
    Creates a dataframe containing all the information stored in 
    all the pickled files in the AFS tool. 
    
    Parameters 
    -----------------
    file_names (list of str): list of pickled file names to use.
    pop_dir (str): directory where the files are stored.
    column_names (list of str): list of column names from dataframe.
                                If none, pulls from the pickled files. 
    
    Returns 
    -----------------
    Pandas dataframe with the following columns: 
    'feature': string representing deap tree with actual column names instead 
                of 'ARG XX'.
    'fitness': fitness value.
    'epoch': epoch number.
    'bfs_flag': flag for whether that feature is in the best feature set 
                for that epoch. 
    'hof_rank': rank in the Hall of fame for that epoch. 
    'bfs_score': AUC of BFS for that epoch
    """
    
    feature_fitness_df = pd.DataFrame(columns = ['feature', 'fitness', 'epoch'])
    bfs_df = pd.DataFrame(columns = ['feature', 'epoch', 'bfs_flag'])
    hof_df = pd.DataFrame(columns = ['feature', 'epoch', 'hof_rank'])
    for pickled_file in file_names:
        epoch = int(re.search('epoch_(\d*).p', pickled_file).group(1))
        with open(pop_dir + pickled_file, 'r') as f:
            temp = pickle.load(f)
        if len(column_names)==0:
            column_names = list(temp['abt_columns'].values)
        temp_df = get_individual_fitness(temp['population'], column_names)
        temp_df['epoch'] = epoch
        feature_fitness_df = feature_fitness_df.append(temp_df)
        if temp['bfs']:
            bfs_names = convert_names(list(temp['bfs']), column_names)
            temp_bfs_df = pd.DataFrame({'feature':bfs_names, 'epoch':epoch, 'bfs_flag':1})
            if temp['bfs_score']:
                temp_bfs_df['bfs_score'] = temp['bfs_score']
            bfs_df = bfs_df.append(temp_bfs_df)
        if temp['hof']:
            hof_names = convert_names([str(x) for x in temp['hof']], column_names)
            hof_df = hof_df.append(pd.DataFrame({'feature':hof_names, 'epoch':epoch, 'hof_rank':range(0,len(hof_names))}))
    combined_df = feature_fitness_df.merge(bfs_df, on=['feature', 'epoch'], how='left')
    combined_df = combined_df.merge(hof_df, on=['feature', 'epoch'], how='left')
    combined_df['bfs_flag'].fillna(0, inplace=True)
    return combined_df
    
def plot_individual_fitness(fitness_df, title='', num_zoomed=20):
    """
    Creates two plots, one with fitness values for all features in the input 
    df and another with only the top fitness values. 
    
    Parameters 
    -----------------
    fitness_df (pandas dataframe): df containing feature and fitness columns.
    title (str): title for the plot.
    num_zoomed (int): number to plot in the zoomed in version. 
    
    Returns 
    -----------------
    Nothing, but creates two plots. 
    """
    
    for_plot_df = fitness_df.sort_values('fitness', ascending=False)
    for_plot_df.set_index('feature', inplace=True)
    for_plot_df['fitness'].plot(kind='bar', figsize=(20,5))
    if title!='':
        plt.title(title)
    plt.show()

    for_plot_df['fitness'].iloc[0:num_zoomed].plot(kind='bar', figsize=(20,5))
    if title!='':
        plt.title(title + ' top ' + str(top_num))
    plt.show()
    
def plot_feature_stats(combined_df, feature):
    """
    Plots how a single feature changes over epochs. 
    
    Parameters 
    -----------------
    combined_df (pandas dataframe): dataframe containing feature, epoch, 
                                    fitness, hof_rank and bfs_flag. 
    feature (str): name of feature to use for the plots. 
    
    Returns 
    -----------------
    Nothing but plots a feature's fitness, hof rank and bfs flag by epoch. 
    """
    
    combined_df.loc[combined_df['feature']==feature].plot('epoch', 'fitness')
    plt.title(feature + ' fitness')
    plt.show()
    combined_df.loc[combined_df['feature']==feature].plot('epoch', 'hof_rank')
    plt.title(feature + ' hof_rank')
    plt.show()
    combined_df.loc[combined_df['feature']==feature].plot('epoch', 'bfs_flag')
    plt.title(feature + ' bfs_flag')
    plt.show()
    
def plot_pop_fitness(combined_df):
    """
    Plots the avg, min, max fitness of the population over time. 
    
    Parameters 
    -----------------
    combined_df (pandas dataframe): df containing feature, epoch, 
                                    fitness. 
    
    Returns 
    -----------------
    Nothing, but displays a plot. 
    """
    
    combined_df.groupby('epoch').agg(['mean', 'max', 'min'])['fitness'].plot()
    plt.title('Fitness of population over time')
    plt.show()


def plot_bfs_fitness(combined_df):
    """
    Plots the avg, min, max fitness of the bfs over time. 
    
    Parameters 
    -----------------
    combined_df (pandas dataframe): df containing feature, epoch, 
                                    fitness and bfs_flag. 
    
    Returns 
    -----------------
    Nothing, but displays a plot. 
    """
    
    combined_df.loc[combined_df['bfs_flag']==1].groupby('epoch').agg(['mean', 'max', 'min'])['fitness'].plot()
    plt.title('Fitness of BFS over time')
    plt.show()

def plot_bfs_size(combined_df):
    """
    Plots the size of the bfs over time. 
    
    Parameters 
    -----------------
    combined_df (pandas dataframe): df containing feature, epoch, 
                                    fitness and bfs_flag. 
    
    Returns 
    -----------------
    Nothing, but displays a plot. 
    """
    
    combined_df.loc[combined_df['bfs_flag']==1].groupby('epoch').count().rename(columns = {'feature':'volume'})[['volume']].plot()
    plt.title('Volume of BFS over time')
    plt.show()

def plot_bfs_score(combined_df):
    """
    Plots the bfs AUC score over time 
    
    Parameters 
    -----------------
    combined_df (pandas dataframe): df containing epoch and bfs_score. 
    
    Returns 
    -----------------
    Nothing, but displays a plot. 
    """
    
    combined_df[['epoch', 'bfs_score']].groupby('epoch').max().plot()
    plt.title('BFS AUC over time')
    plt.show()