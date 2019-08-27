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


def plot_heatmap(path = None, output_dir = None , df_corr = None, row_names = None, output_file_name = 'similarity_experiment', with_corr_values=True,
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

    # Mask
    df = pd.read_excel(path, sheet_name='figure2', header=1).iloc[:,2:]
    df.index = df_corr.index
    mask = df.isnull()

    # Plot
    g = sns.heatmap(df_corr,mask=mask, cmap="RdBu_r", vmin = vmin, vmax=vmax , cbar_kws={"ticks":ticks,"shrink": 0.5 }, annot=with_corr_values, xticklabels=True, yticklabels=True, )
    g.set_facecolor('xkcd:white')
    g.figure.axes[-1].yaxis.label.set_size(label_size)
    # plt.tight_layout(tight_layout)
    plt.yticks(rotation=0, fontsize=font_scale)
    plt.xticks(rotation=x_rotation, fontsize=font_scale)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
              'r', 'r', 'r', 'r', 'r', 'r','r',
              'purple',
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

if __name__ == "__main__":
    input_dir = 'data/inputs/'
    output_dir = 'data/outputs/'
    input_file = 'speech_psychiatry_figure2.xlsx'
    path = input_dir + input_file
    # Clean dataframe
    matrix = preprocess_table(path = path)
    # Scale
    matrix_flat = matrix.values.flatten()
    matrix_scaled = maxabs_scale(matrix_flat,axis=0)
    matrix_scaled = np.reshape(matrix_scaled, matrix.shape)
    matrix_scaled = pd.DataFrame(matrix_scaled, columns=matrix.columns, index=matrix.index)

    # Plot heatmap from figure 2
    output_file_name = 'features_by_disorders_maxabs_scaler'
    plot_heatmap(path= path, output_dir = output_dir, df_corr = matrix_scaled, row_names = matrix.index,
                 output_file_name = output_file_name, with_corr_values=False,
                 value_range=[-1,1],
                 xlabel = 'Psychiatric disorders', 
                 ylabel='Acoustic features',
                 font_scale=7,
                 x_rotation=45 , 
                 label_size = 8, 
                 tight_layout = 0.1)

