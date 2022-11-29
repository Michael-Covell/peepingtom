import pandas as pd; pd.set_option('display.max_colwidth',3000)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo


def get_key_from_value_list(my_dict,value):
    for k, v_list in my_dict.items():
        for v in v_list:
            if v == value:
                return k

#data (required)
#grouping_variables (required)
#DV (required)
#equal_var (required)
#subset_combinations (optional)
#returns DataFrame containing statistics that describe the group difference in the DV between the "group" and the all other cases within the subset
def group_mean_differences(data,grouping_variables,DV,equal_var,subset_combinations=0):
    import scipy.stats as stats
    import itertools

    #Create dictionary d of {grouping variable:[categories]} and list of categories
    d = {}
    for g in grouping_variables:
        d.update({g:list(data[g].unique())})
    categories = [v for value_lists in d.values() for v in value_lists]

    #create subset_list
    subset_list = []
    for i in range(subset_combinations+1):
        for subset in itertools.combinations(categories, i):
            groupings = [get_key_from_value_list(d,value) for value in subset]
            category_counts = [groupings.count(x) for x in groupings]
            repeated_categories = any(x > 1 for x in category_counts)
            if not repeated_categories and subset not in subset_list:
                subset_list.append(list(subset))
    print(subset_list)

    #Add dummy coded variables
    for gv in grouping_variables:
        groups = data[gv].unique()
        for g in groups:
            data[g] = np.where(data[gv] == g, 1, 0)

    #Create DataFrame containing statistics
    df = pd.DataFrame(columns=['grouping_variable','group','subset','t','p'])
    for gv in grouping_variables:
        groups = data[gv].unique()
        for g in groups:
            #get statistics for subsets of dataset
            for subset in subset_list:

                b1 = data[g]==1
                b2 = data[g]==0
                b_subset = (data[subset]==1).all(axis='columns')
                temp_df = pd.DataFrame({'grouping_variable':[gv],
                                        'group':[g],
                                        'subset':[subset]})

                group1 = data.loc[b1&b_subset,DV]
                group2 = data.loc[b2&b_subset,DV]
                if group1.notna().any() and group2.notna().any():
                    mean_difference = group1.mean() - group2.mean()
                    median_difference = group1.median() - group2.median()
                    ttest = stats.ttest_ind(a=group1, b=group2, equal_var=equal_var)
                    t = ttest[0]
                    p = ttest[1]
                    temp_df['t'] = t
                    temp_df['p'] = p
                    temp_df['mean_difference'] = mean_difference
                    temp_df['median_difference'] = median_difference
                    df = pd.concat([df,temp_df])

    return df.reset_index(drop=True).sort_values('p')


def group_count_differences(data,grouping_variables,DV,subset_combinations=0):
    import scipy.stats as stats
    import itertools

    #Create dictionary d of {grouping variable:[categories]} and list of categories
    d = {}
    for g in grouping_variables:
        d.update({g:list(data[g].unique())})
    categories = [v for value_lists in d.values() for v in value_lists]

    #create subset_list
    subset_list = []
    for i in range(subset_combinations+1):
        for subset in itertools.combinations(categories, i):
            groupings = [get_key_from_value_list(d,value) for value in subset]
            category_counts = [groupings.count(x) for x in groupings]
            repeated_categories = any(x > 1 for x in category_counts)
            if not repeated_categories and subset not in subset_list:
                subset_list.append(list(subset))
#    print(subset_list)

    #Add dummy coded variables
    for gv in grouping_variables:
        groups = data[gv].unique()
        for g in groups:
            data[g] = np.where(data[gv] == g, 1, 0)

    #Create DataFrame containing statistics
    df = pd.DataFrame(columns=['grouping_variable','group','subset'])
    for gv in grouping_variables:
        groups = data[gv].unique()
        for g in groups:
            #get statistics for subsets of dataset
            for subset in subset_list:
                temp_df = pd.DataFrame({'grouping_variable':[gv],
                        'group':[g],
                        'subset':[subset]})

                b = data[g]==1
                b_subset = (data[subset]==1).all(axis='columns')
                group = data.loc[b&b_subset,DV]
                if group.notna().any():
                    temp_df['count'] = group.count()
                    df = pd.concat([df,temp_df])

        for gv in grouping_variables:
            groups = data[gv].unique()
            for subset in subset_list:
                for pair in itertools.combinations(groups, 2):
                    b1 = (df['group'] == pair[0]) & (df['subset'].astype(str) == str(subset))
                    b2 = (df['group'] == pair[1]) & (df['subset'].astype(str) == str(subset))
                    try:
                        group1_cts = df.loc[b1,'count']
                        group2_cts = df.loc[b2,'count']
                        df.loc[b1,'percent_difference_from_'+pair[1]] = (group1_cts - group2_cts)/(group2_cts)
                    except KeyError:
                        pass

    return df.reset_index(drop=True).sort_values('count',ascending=False)


#"items" (required) must be a DataFrame or array where items are represented in columns
#n_factors (optional) may be entered to specify the number of factors in the factor correlation matrix used to create the scree plot
#applies StandardScaler to DataFrame containing items
#prints chi-square and p-value from Bartlett’s Sphericity test
    #H0: the matrix of population correlations is equal to 1
    #H1: the matrix of population correlations is not equal to 1
#prints overall KMO score () for the Kaiser-Meyer-Olkin criterion
    #this statistic represents the degree to which each observed variable is predicted, without error, by the other variables in the dataset. In general, a KMO < 0.6 is considered inadequate)
#returns table and scree plot of original eigenvalues, given the factor correlation matrix
    #eigenvalues >= 1 indicate significant variance among factors (which can be used as a justification to choosing that number of factors for the exploratory factor analysis)
    #a dropoff in the scree plot indicates a drop in variance among factors (which can be used to justify choosing the largest number of factors with relatively high variance for the  exploratory factor analysis)
#for more info, see https://factor-analyzer.readthedocs.io/en/latest/factor_analyzer.html
def adequacy_and_scree_plot(items,n_factors=None):
    scaled_df = pd.DataFrame(data=StandardScaler().fit_transform(items),columns = items.columns)

    chi_square_value,p_value = calculate_bartlett_sphericity(scaled_df)
    print('chi_square_value = ',chi_square_value)
    print('p_value = ',p_value)

    kmo_per_variable,kmo_total = calculate_kmo(scaled_df)
    print('kmo_total = ',kmo_total)

    if n_factors == None:
        n_factors = len(items.columns)
    fa= FactorAnalyzer(n_factors,rotation=None)
    fa.fit(scaled_df)
    original_evs,common_factor_evs = fa.get_eigenvalues()
    display(pd.DataFrame(data= {'number of factors':np.arange(1,len(original_evs)+1),'Original_Eigenvalues':original_evs}).round(2))

    plt.scatter(range(1,scaled_df.shape[1]+1),original_evs)
    plt.plot(range(1,scaled_df.shape[1]+1),original_evs)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

#"items" (required) must be a DataFrame or array where items are represented in columns
#n_factors (required) may be entered to specify the number of factors in the factor correlation matrix used to create the scree plot
#applies StandardScaler to DataFrame containing items
#runs an exploratory factor analysis with default parameters of factor_analyzer.FactorAnalyzer() and listwise deletion
#displays a table of factor loadings from exploratory factor analysis
def EFA(items,n_factors):
    scaled_df = pd.DataFrame(data=StandardScaler().fit_transform(items),columns = items.columns)
    fa = FactorAnalyzer(n_factors = n_factors,impute='drop')
    fa.fit(scaled_df)

    col_names = ['Factor '+str(n) for n in range(1,n_factors+1)]
    loadings = pd.DataFrame(fa.loadings_, columns=col_names, index=items.columns)
    display(loadings.sort_values(col_names,ascending=False).style.background_gradient(cmap='Blues'))
