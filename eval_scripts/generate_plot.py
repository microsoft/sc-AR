import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from scipy.stats import ttest_ind
import argparse
import seaborn as sns
import numpy as np
import os
import re


parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
## Standard training ##
# 20240520-species-oodTrue-variable_conFalse.csv
# 20240521-pbmc-oodTrue-variable_conFalse.csv
# 20240522-lps-hpoly-oodTrue-variable_conFalse.csv

## Variable control setting ##
# 20240619-species-oodTrue-variable_conTrue.csv
# 20240624-pbmc-oodTrue-variable_conTrue.csv
# 20240702-lps-hpoly-oodTrue-variable_conTrue.csv
parser.add_argument("--name", default='20240702-lps-hpoly-oodTrue-variable_conTrue.csv', help="data name")
parser.add_argument("--data", default='lps-hpoly', help="data name") # lps-hpoly
args = parser.parse_args()
config = vars(args)

path = os.getcwd()
root = os.path.abspath(os.path.join(path, os.pardir))
print('root: ', root)
# missingfolder = '/MSR_internship_new_git/sc-uncertainty'
missingfolder = '/sc-uncertainty'
data = pd.read_csv(str(root)+missingfolder+'/result/test/'+args.data+"/"+args.name, header=0)

data.columns = ['id', 'test', 'AR', 'variable_con',
                'con_percent', 'in_dist_group',
                'metric', 'value', 'seed']
colors = [px.colors.qualitative.Dark2[6], 'pink', 'rosybrown', 'blue', 'green']


def extract_variable(input_string):
    # Define the regular expression pattern
    pattern = r'-test-(.*?)-'

    # Use re.search to find the first match
    match = re.search(pattern, input_string)

    if match:
        # Extract the captured group (X)
        variable_X = match.group(1)
        return variable_X
    elif 'random' in input_string:
        # Return None if no match is found
        return 'random'

    
def add_test_column(df):
    lst = df['id']
    cell_list = []
    for l in lst:
        cell_list.append(extract_variable(l))
        
    df['test'] = cell_list
    return df


def create_latex_table(df, col_group='test', file_name=None): 

    f = open(root+missingfolder+'/result/test/'+args.data+'/'+args.name.split('.')[0]+'_latex.csv', "a")
    print(root+missingfolder+'/result/test/'+args.data+'/'+args.name.split('.')[0]+'_latex.csv')
    f.write('\\newpage\\\\\n')
    f.write(args.name.split('.')[0].replace('-', ' ')+'\\\\\n')
    f.write('Variable Control Sample from: '+file_name+'\\\\\n')

    metrics = ['MSE',
               '20 DEG MSE',
               '50 DEG MSE',
               '100 DEG MSE',
               'Cosine Similarity']

    # add a df['model'] column to the dataframe based on value of df['AR']
    df['model'] = df['AR'].apply(lambda x: 'AR' if x == True else 'Standard')
    methods = list(set(df['model']))
    print(methods)

    if 'test' not in df.columns:
        df = add_test_column(df)

    df.sort_values('test')
    column_list = list(set(df[col_group]))
    # Specify the scientific notation format
    scientific_notation_format = '{:.1e}'.format
    
    table = pd.DataFrame(columns=column_list+['Avg'])
    for metric in metrics:
        table.loc[metric + ' Standard'] = [0] * (len(column_list)+1)
        table.loc[metric + ' AR'] = [0] * (len(column_list)+1)
        table.loc[metric + ' p-value'] = [0] * (len(column_list)+1)
        table.loc[metric + ' Standard std'] = [0] * (len(column_list)+1)
        table.loc[metric + ' AR std'] = [0] * (len(column_list)+1)

    for metric in metrics:
        method_values = {}
        method_values['Standard'] = []
        method_values['AR'] = []
        method_values_avg_naive = list()
        method_values_avg_ar = list()
        
        for c in column_list:
            for method in methods:
                # print('Method: ', method, 'Metric: ', metric, 'Column: ', c)

                feature_df = df[(df['metric'] == metric) &
                                (df['model'] == method) &
                                (df[col_group] == c)]
                # print(feature_df.shape)

                method_values[method] = feature_df['value'].values.tolist()
                assert len(method_values[method]) == len(list(set(df['seed']))) == \
                    len(feature_df['value']) == 5

                avg = feature_df['value'].astype(float).mean()
                std = feature_df['value'].astype(float).std()
                table.loc[metric + ' '+method, c] = avg
                table.loc[metric + ' '+method+' std', c] = std

            table.loc[metric + ' p-value', c] = ttest_ind(
                method_values['Standard'],
                method_values['AR'])[1]
   
        # calculate average of each metric
        table.loc[metric + ' Standard', 'Avg'] = table.loc[metric + ' Standard', column_list].mean()
        table.loc[metric + ' AR', 'Avg'] = table.loc[metric + ' AR', column_list].mean()
        table.loc[metric + ' Standard std', 'Avg'] = table.loc[metric + ' Standard std', column_list].mean()
        table.loc[metric + ' AR std', 'Avg'] = table.loc[metric + ' AR std', column_list].mean()

        # calculate p-value of average of cells for each metric and seed
        avg_df = df[(df['metric'] == metric)]
        for seed in set(avg_df['seed']):
            avg_df_seed = avg_df[(avg_df['seed'] == seed)]
            method_values_avg_naive.append(avg_df_seed[avg_df_seed['model'] == 'Standard']['value'].values.mean())
            method_values_avg_ar.append(avg_df_seed[avg_df_seed['model'] == 'AR']['value'].values.mean())
            assert avg_df_seed[avg_df_seed['model'] == 'Standard'].shape[0] == avg_df_seed[avg_df_seed['model'] == 'AR'].shape[0] == len(column_list)
        assert len(method_values_avg_naive) == len(method_values_avg_ar) == len(list(set(df['seed'])))
        table.loc[metric + ' p-value', 'Avg'] = ttest_ind(
            method_values_avg_naive,
            method_values_avg_ar)[1]

    data = table.applymap(scientific_notation_format)

    if col_group == 'con_percent':
        data.columns = [str(col) for col in data.columns]
        data.columns = [col[:3] for col in data.columns]
        data = data.reindex(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', 'Avg'], axis=1)

    f.write(data.to_latex(index=True, multirow=True))
    f.write('\\\\')
    f.close()

    return df


def generate_barplot(data, data_name, metric, group='test', cell='', file_name=''):

    data['model'] = data['AR'].apply(lambda x: 'AR' if x == True else 'Standard')
    # replace values of "Enterocyte.Progenitor" with "Ent.Progenitor" for better visualization
    data['test'] = data['test'].replace('Enterocyte.Progenitor', 'Enterocyte.P')
    
    palette = {'Standard':"darkgray",
               'AR':"brown"}
    
    sub_data = data[data["metric"] == metric]
    # sort sub_data based on values in the "test" column
    sub_data = sub_data.sort_values(group)
    ## comment for variable control setting
    # if 'species' in data_name:
    #     sub_data[group] = [x.capitalize() for x in sub_data[group]]

    plt.clf()
    ax = sns.barplot(data=sub_data,
                      x=group,
                      y="value",
                      hue="model",
                      palette = palette,
                      hue_order=["Standard", "AR"],
                      ci="sd",
                      alpha=0.8,)
    
    # for each group in the barplot, calculate p_values between AR and Naive and add a star if p_value < 0.05
    groups = list(set(sub_data[group]))
    groups.sort()

    for g in groups:
        p_value = ttest_ind(
            sub_data[(sub_data[group] == g) & (sub_data['model'] == 'AR')]['value'].values,
            sub_data[(sub_data[group] == g) & (sub_data['model'] == 'Standard')]['value'].values)[1]

        if p_value < 0.05 and p_value > 0.01:
            ax.annotate('*', xy=(groups.index(g)-0.2, 0.95), xytext=(groups.index(g), 0.98), size=20, horizontalalignment='center')
        elif p_value < 0.01 and p_value > 0.001:
            ax.annotate('**', xy=(groups.index(g)-0.2, 0.95), xytext=(groups.index(g), 0.98), size=20, horizontalalignment='center')
        elif p_value < 0.001 and p_value > 0.0001:
            ax.annotate('***', xy=(groups.index(g)-0.2, 0.95), xytext=(groups.index(g), 0.98), size=20, horizontalalignment='center')
        elif p_value < 0.0001:
            ax.annotate('****', xy=(groups.index(g)-0.2, 0.95), xytext=(groups.index(g), 0.98), size=20, horizontalalignment='center')
    
    # for bar in ax.patches[:-2]:
    #     ax.annotate(format(bar.get_height(), '.2f'), 
    #                 (bar.get_x() + bar.get_width() / 2, 
    #                     0), ha='center', va='center',
    #                 size=12, xytext=(0, 12),
    #                 textcoords='offset points')
    
    # xlabel = "Test Group"
    xlabel = ''
    # if group == 'con_percent':
    #     xlabel = "Control Percentage Included in Training"
    # ax.tick_params(axis='x', labelrotation=20)
    ax.set_ylim(0, 1)
    # ax.set_xlabel(xlabel) 
    print('metric barplot: ', metric)
    if '20 DEG R2' in metric:
        metric = 'DEG R2'
    # ax.set_ylabel(metric)
    # adjust font of legend and legend title
    # plt.setp(ax.get_legend().get_texts(), fontsize='20')
    
    palette = {'Standard':"dimgray",
               'AR':"brown"}
    sns.stripplot(
        x=group, 
        y="value", 
        hue="model", 
        palette = palette,
        data=sub_data, dodge=True, alpha=1.0, ax=ax,
        hue_order=["Standard", "AR"],
    )
    handles, labels = ax.get_legend_handles_labels()
    sns.despine(top=True,right=True, bottom=False, left=False)
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # if cell != '':
    #     ax.set_title(cell)
    # else:
    #     title = data_name.split('_')[1]
    #     if title == 'pbmc':
    #         title = 'PBMC'
    #     if title == 'species':
    #         title = 'Species'
    #     if title == 'lps-hpoly':
    #         title = 'HPoly'
        # ax.set_title(title)
    fig = ax.get_figure()
    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    # adjust font size of x and y ticks
    if 'species' in data_name:
        ax.tick_params(axis='x', labelsize=20, labelrotation=40)
        # capitalize x axis labels

    elif 'lps-hpoly' in data_name or 'pbmc' in data_name:
        ax.tick_params(axis='x', labelsize=12, labelrotation=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.title.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(metric)
    
    original_date=file_name.split('-')[0]
    file_name = '-'.join(file_name.split('-')[1:])
    save_path = root+missingfolder+'/result/dataset_figure/'+data_name.split('_')[1]+"/"+file_name+'/'+original_date
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path+"/"+"barplot-"+metric+"-"+group+"-"+cell+".pdf", bbox_inches='tight', dpi=600)


def generate_boxplot(data, data_name, metric, group='test', cell='', file_name=''):
    sub_data = data[data["metric"] == metric]
    sub_data = sub_data.sort_values(group)
    sub_data['test'] = sub_data['test'].replace('Enterocyte.Progenitor', 'Enterocyte.P')
    palette = {'Standard':"darkgray",
               'AR':"brown"}
    
    plt.clf()
    ## comment for variable control setting
    # if 'species' in data_name:
    #     sub_data[group] = [x.capitalize() for x in sub_data[group]]

    ax = sns.boxplot(x=group, y="value", hue="model", data=sub_data,
                     palette=palette, hue_order=["Standard", "AR"],
                     showfliers=False, boxprops=dict(alpha=.9))

    # if cell != '':
    #     ax.set_title(cell, fontsize=20)
    # else:
    #     title = data_name.split('_')[1]
    #     if title == 'pbmc':
    #         title = 'PBMC'
    #     if title == 'species':
    #         title = 'Species'
    #     if title == 'lps-hpoly':
    #         title = 'HPoly'
    #     ax.set_title(title, fontsize=20)
        
    groups = list(set(sub_data[group]))
    groups.sort()
    
    height = 0
    max_value = 0
    
    if metric == 'Cosine Similarity':
        # set its range from 0 to 1
        ax.set_ylim(sub_data['value'].min()-0.005, 1)
        max_value = 0.9999
        height = 0.9999#  - (sub_data['value'].max()- sub_data['value'].min())/20
    elif 'DEG MSE' in metric:
        max_value = sub_data['value'].max()
        height = max_value - (sub_data['value'].max()- sub_data['value'].min())/4
        ax.set_ylim(sub_data['value'].min(), max_value)
    elif 'MSE' in metric:
        max_value = sub_data['value'].max()
        height = max_value #- (sub_data['value'].max()- sub_data['value'].min())/7
        ax.set_ylim(sub_data['value'].min(), max_value)
        
    print(sub_data['value'])
    print('min value: ', sub_data['value'].min())
    print('max value: ', max_value)
    print(metric, "height: ", height)
    print()

    for g in groups:
        p_value = ttest_ind(
            sub_data[(sub_data[group] == g) & (sub_data['model'] == 'AR')]['value'].values,
            sub_data[(sub_data[group] == g) & (sub_data['model'] == 'Standard')]['value'].values)[1]

        if p_value < 0.05 and p_value > 0.01:
            ax.annotate('*', xy=(groups.index(g), max_value), size=20, horizontalalignment='center')
        elif p_value < 0.01 and p_value > 0.001:
            ax.annotate('**', xy=(groups.index(g), max_value), size=20, horizontalalignment='center')
        elif p_value < 0.001 and p_value > 0.0001:
            ax.annotate('***', xy=(groups.index(g), max_value), size=20, horizontalalignment='center')
        elif p_value < 0.0001:
            ax.annotate('****', xy=(groups.index(g), max_value), size=20, horizontalalignment='center')
            
    handles, labels = ax.get_legend_handles_labels()
    sns.despine(top=True,right=True, bottom=False, left=False)
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    if 'species' in data_name:
        ax.tick_params(axis='x', labelsize=20, labelrotation=40)
    elif 'lps-hpoly' in data_name or 'pbmc' in data_name:
        ax.tick_params(axis='x', labelsize=12, labelrotation=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.title.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.set_xlabel('') 
    if 'DEG MSE' in metric:
        metric = 'DEG MSE'
    elif 'MSE' in metric:
        metric = 'MSE'
    if 'DEG Cosine Similarity' in metric:
        metric = 'DEG Cosine Similarity'
    elif 'Cosine Similarity' in metric:
        metric = 'Cosine Similarity'
    ax.set_ylabel(metric)
    
    plt.setp(ax.get_legend().get_texts(), fontsize='20')

    # xlabel = "Test Group"
    # if group == 'con_percent':
    #     xlabel = "Control Percentage Included in Training"
    # ax.set_xlabel(xlabel)
    # print('metric: ', metric)
    # if '20 DEG R2' in metric:
    #     metric = 'DEG R2'
    # if '50 DEG MSE' in metric:
    #     metric = 'DEG MSE'
    # if '100 DEG Cosine Similarity' in metric:
    #     metric = 'DEG Cosine Similarity'
    # ax.set_ylabel(metric)
    fig = ax.get_figure()
    
    original_date=file_name.split('-')[0]
    file_name = '-'.join(file_name.split('-')[1:])
    save_path = root+missingfolder+'/result/dataset_figure/'+data_name.split('_')[1]+"/"+file_name+'/'+original_date
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path+"/"+"boxplot-"+metric+"-"+group+"-"+cell+".png", bbox_inches='tight', dpi=300)
        
    return


def generate_plot(data, data_name):

    if ('variable_conFalse' in args.name):
        print(data)
        file_name = args.name.split('.')[0]
        data = create_latex_table(data, 'test', file_name)

        generate_barplot(data, data_name, 'R2', 'test', '', file_name)
        # generate_barplot(data, data_name, 'Diff R', 'test', '', file_name)
        generate_barplot(data, data_name, '20 DEG R2', 'test', '', file_name)
        # generate_barplot(data, data_name, '50 DEG R2', 'test', '', file_name)
        # generate_barplot(data, data_name, '100 DEG R2', 'test', '', file_name)
        # generate_barplot(data, data_name, '20 DEG Diff R', 'test', '', file_name)

        generate_boxplot(data, data_name, 'MSE', 'test', '', file_name)
        generate_boxplot(data, data_name, '20 DEG MSE', 'test', '', file_name)
        generate_boxplot(data, data_name, 'Cosine Similarity', 'test', '', file_name)
        # generate_boxplot(data, data_name, '20 DEG Cosine Similarity', 'test', '', file_name)

    elif 'variable_conTrue' in args.name:
        f = open(root+missingfolder+'/result/test/'+args.data+'/'+args.name.split('.')[0]+'_latex.csv', "a")
        f.write('\\newpage\\\\\n')
        f.write(args.name.split('.')[0].replace('-', ' ')+'\\\\\n')
        file_name = args.name.split('.')[0]
        
        # data = add_test_column(data)
        cell = list(set(data['test']))
        label = 'con_percent'
        for c in cell:
            print('Cell: ', c)
            df_cell = data[(data['test'] == c)]
            # df_cell = create_latex_table(df_cell, col_group=label, cell=c)
            generate_barplot(df_cell, data_name, 'R2', label, c, file_name)
            generate_barplot(df_cell, data_name, '20 DEG R2', label, c, file_name)
            generate_barplot(df_cell, data_name, '50 DEG R2', label, c, file_name)
            generate_barplot(df_cell, data_name, '100 DEG R2', label, c, file_name)
            
            # generate_barplot(df_cell, data_name, 'Diff R', label, c, file_name)
            # generate_barplot(df_cell, data_name, '20 DEG Diff R', label, c, file_name)
            
            generate_boxplot(df_cell, data_name, 'MSE', label, c, file_name)
            generate_boxplot(df_cell, data_name, '20 DEG MSE', label, c, file_name)
            generate_boxplot(df_cell, data_name, 'Cosine Similarity', label, c, file_name)

    return

experiment_name = args.name.split('-')[0]
generate_plot(data, experiment_name+"_"+args.data)