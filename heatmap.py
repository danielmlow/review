#!/usr/bin/env python3

'''
Author: Daniel M. Low
license: Apache 2.0

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import maxabs_scale


def preprocess_table(path = None):
    df = pd.read_excel(path, sheet_name='figure2',header=1)
    df = df.replace(np.nan, 0, regex=True)
    matrix = []
    for row in df.iloc[:,2:].values:
        row_new = []
        # print(row)
        for cell in row:
            if cell == 0:
                row_new.append(0)
            else:
                papers = cell.split(';')
                values = [int(n.split(',')[0].replace(' ','')) for n in papers]
                # mean
                # score = np.mean(values)
                # alternative methods takes amount of papers into account:
                score = np.sum(values) * (1/1+(values.count(0)))
                row_new.append(score)
        matrix.append(row_new)
    matrix = pd.DataFrame(matrix, columns=df.columns[2:], index=df.iloc[:,1])
    return matrix


def plot_heatmap(output_dir , df_corr, row_names = None, output_file_name = 'similarity_experiment', with_corr_values=True,
                 value_range=[-1,1],
                 font_scale = None, tight_layout=1.8, xlabel = 'Psychiatric disorders', ylabel='Acoustic features',
                 x_rotation=45 , label_size = 8):
    if value_range == [-1,1]:
        vmin = -1.0
        vmax = 1.0
        ticks = [-1., -0.5, 0.0, 0.5, 1.0]
    elif value_range == [0, 1]:
        vmin = 0.
        vmax = 1.0
        ticks = [0, 0.25, 0.5, 0.75, 1.0]
    plt.clf()
    sns.set(font_scale=font_scale)
    g = sns.heatmap(df_corr,cmap="RdBu_r", vmin = vmin, vmax=vmax , cbar_kws={"ticks":ticks,"shrink": 0.5 }, annot=with_corr_values, xticklabels=True, yticklabels=True, )
    g.figure.axes[-1].yaxis.label.set_size(label_size)
    # plt.tight_layout(tight_layout)
    plt.yticks(rotation=0, fontsize=font_scale)
    plt.xticks(rotation=x_rotation, fontsize=font_scale)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
              'r', 'r', 'r', 'r', 'r', 'r','r',
              'k','k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', ]
    g.axes.set_yticklabels(row_names)
    for xtick, color in zip(g.axes.get_yticklabels(), colors):
        xtick.set_color(color)
    cbar = g.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=label_size)
    plt.tight_layout(tight_layout)
    plt.savefig(output_dir + output_file_name+ '.eps', format='eps', dpi=600)
    return


def plot_cluster_map(output_dir = './' , dataframe = pd.DataFrame(),output_file_name = 'figure2', mask = None, x_rotation=45, xlabel = None , ylabel = None, label_size = 10, tight_layout=1.8, font_scale=1):
    plt.clf()
    sns.set(font_scale=font_scale)
    cg = sns.clustermap(dataframe, method='ward', metric='euclidean', cmap="RdBu_r", vmin=-1., vmax=1.0,
                        cbar_kws={"ticks": [-1., -0.5, 0.0, 0.5, 1.0]},
                        mask = mask)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=x_rotation)
    ax = cg.ax_heatmap
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    colors = ['b', 'b', 'b', 'b', 'b', 'b',
              'r', 'r', 'r', 'r', 'r','r',
              'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', ]
    # find new position of ytick labels
    d = {}
    for i, color in enumerate(colors):
        d[i]=color
    new_index = list(cg.dendrogram_row.reordered_ind)
    new_colors = []
    for i, index in enumerate(new_index):
        color = d.get(index)
        new_colors.append(color)
    # set color to yticklabels
    for xtick, color in zip(cg.ax_heatmap.yaxis.get_majorticklabels(), new_colors):
        xtick.set_color(color)
    plt.tight_layout(tight_layout)
    cg.savefig(output_dir + output_file_name+ '.eps', format='eps', dpi=600)
    return

def hardlim(x):
    if x == 0:
        y = 0
    elif x<0:
        y = -1
    elif x > 0:
        y = 1
    return y


if __name__ == "__main__":
    input_dir = 'data/inputs/'
    output_dir = 'data/outputs/'
    input_file = 'speech_psychiatry_figure2.xlsx'
    # Clean dataframe
    matrix = preprocess_table(path = input_dir+input_file)
    # Scale
    matrix_flat = matrix.values.flatten()
    matrix_scaled = maxabs_scale(matrix_flat,axis=0)
    matrix_scaled = np.reshape(matrix_scaled, matrix.shape)
    matrix_scaled = pd.DataFrame(matrix_scaled, columns=matrix.columns, index=matrix.index)

    # Plot heatmap from figure 2
    output_file_name = 'features_by_disorders_maxabs_scaler'
    plot_heatmap(output_dir = output_dir, df_corr = matrix_scaled, row_names = matrix.index,
                 output_file_name = output_file_name, with_corr_values=False,
                 value_range=[-1,1],
                 xlabel = 'Psychiatric disorders', 
                 ylabel='Acoustic features',
                 font_scale=6,
                 x_rotation=45 , 
                 label_size = 8, 
                 tight_layout = 0.1)

    # Plot cluster heatmap from figure 3
    matrix_scaled_hardlim  = matrix_scaled.applymap(hardlim) # Apply hardlim to avoid effects
    # Remove disorders with few studies
    matrix_scaled_hardlim_big_disorders = matrix_scaled_hardlim.drop(['OCD', 'Bulimia', 'Anorexia', 'PTSD'], axis=1)
    # Remove features only present in dropped disorders
    matrix_scaled_hardlim_big_disorders = matrix_scaled_hardlim_big_disorders.drop(['Pause variability', 'Maximum phonation time', 'F2 variability',
                                                                                'F1 variability', 'ADR', 'FDR', 'Tremor', 'HNR', ], axis=0)
    plot_cluster_map(output_dir = output_dir, dataframe = matrix_scaled_hardlim_big_disorders, 
                    output_file_name = 'disorders_clustermap_hardlim_big_disorders', 
                    mask= None, 
                    x_rotation=45, 
                    xlabel = 'Similarity between psychiatric disorders given acoustic features', 
                    ylabel = 'Empirical ontology of acoustic features given psychiatric traits', 
                    label_size = 12, 
                    tight_layout = 1.8)
