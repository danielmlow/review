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
	df = pd.read_excel(path, sheet_name='heatmap',header=1)
	df = df.replace(np.nan, 0, regex=True)
	matrix = []
	# matrix_amount = []
	for row in df.iloc[:,2:].values:
		row_new = []
		row_new_amount = []
		for cell in row:
			if cell == 0:
				row_new.append(0)
				row_new_amount.append(0)
			else:
				papers = cell.split(';')
				values = [int(n.split(',')[0].replace(' ','')) for n in papers]
				# mean
				# score = np.sum(values)
				# nulls have to be subtracted if total score_wo_nulls is positive, and added if total score is negative
				null_weight = values.count(0)
				sum_wo_nulls = np.sum([n for n in values if n != 0])

				if sum_wo_nulls  > 0:
					score = sum_wo_nulls - null_weight

				elif sum_wo_nulls  < 0:
					score = sum_wo_nulls + null_weight

				elif sum_wo_nulls == 0:
					score = 0

				row_new.append(score)
				# row_new_amount.append(amount_of_studies)
		matrix.append(row_new)
		# matrix_amount.append(row_new_amount)
	matrix = pd.DataFrame(matrix, columns=df.columns[2:], index=df.iloc[:,1])
	# matrix_amount = pd.DataFrame(matrix_amount, columns=df.columns[2:], index=df.iloc[:, 1])
	# return matrix, matrix_amount
	return matrix


def plot_heatmap(path = None, output_dir = None , df_corr = None, row_names = None, output_file_name = 'similarity_experiment', with_corr_values=True,
				 value_range=[-1,1],
				 font_scale = None, tight_layout=1.8, xlabel = 'Psychiatric disorders', ylabel='Acoustic features',
				 x_rotation=45 , label_size = 8, plot_for_presentation=False):
	vmin = value_range[0]
	vmax = value_range[1]
	ticks = np.linspace(vmin,vmax,5)
	plt.clf()
	sns.set(font_scale=font_scale)

	# Mask
	df = pd.read_excel(path, sheet_name='heatmap', header=1).iloc[:,2:]
	df.index = df_corr.index
	mask = df.isnull()


	# Plot

	g = sns.heatmap(df_corr, mask=mask, cmap="RdBu_r", vmin = vmin, vmax=vmax , cbar_kws={"ticks":ticks,"shrink": 0.5 }, annot=with_corr_values, xticklabels=True, yticklabels=True )


	g.set_facecolor('xkcd:silver')
	g.figure.axes[-1].yaxis.label.set_size(label_size)
	# plt.tight_layout(tight_layout)
	plt.yticks(rotation=0, fontsize=font_scale)
	plt.xticks(rotation=x_rotation, fontsize=font_scale)
	plt.xlabel(xlabel, fontsize=label_size)
	plt.ylabel(ylabel, fontsize=label_size)
	if not plot_for_presentation:
		colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
				  'r', 'r', 'r', 'r', 'r', 'r','r',
				  'purple',
				  'k','k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', ]
		g.axes.set_yticklabels(row_names)
		for xtick, color in zip(g.axes.get_yticklabels(), colors):
			xtick.set_color(color)

	if plot_for_presentation:

		colors = ['k','k','w','w','w','w','w','w','w',
				  'k', 'w','w','w','w','w','w',
				  'w',
				  'k','k', 'w', 'w', 'k','w','w','w','k','w','w','w','w','w','w','k']
		g.axes.set_yticklabels(row_names)
		for xtick, color in zip(g.axes.get_yticklabels(), colors):
			xtick.set_color(color)


	cbar = g.collections[0].colorbar
	# here set the labelsize by 20
	cbar.ax.tick_params(labelsize=label_size)
	plt.tight_layout(tight_layout)
	plt.savefig(output_dir + output_file_name+ '.'+format, format=format, dpi=600)
	return





if __name__ == "__main__":
	# Config
	plot_for_presentation = False
	format = 'eps'
	input_dir = 'data/inputs/'

	output_dir = 'data/outputs/'
	input_file = 'speech_psychiatry_heatmap.xlsx'
	path = input_dir + input_file

	# Clean dataframe
	matrix = preprocess_table(path = path)

	# Find most important features for each disorder
	nlargest_features = {}
	for disorder in matrix.columns:
		features = list(matrix.nlargest(3,disorder).index)
		nlargest_features[disorder] = features

	print('3 highest absolute scores per disorder', nlargest_features)

	# Plot heatmap

	output_file_name = 'features_by_disorders_maxabs_scaler'

	max_abs_value = np.max(np.max(np.abs(matrix)))

	if plot_for_presentation:
		format = 'png'
		plot_heatmap(path= path, output_dir = output_dir, df_corr = matrix, row_names = matrix.index,
					 output_file_name = output_file_name, with_corr_values=False,
					 value_range=[-max_abs_value ,max_abs_value ],
					 xlabel = 'Psychiatric disorders',
					 ylabel='Acoustic features',
					 font_scale=10,
					 x_rotation=45 ,
					 label_size = 10,
					 tight_layout = 0.1,
					 plot_for_presentation = plot_for_presentation)
	else:
		# As in manuscript
		plot_heatmap(path= path, output_dir = output_dir, df_corr = matrix, row_names = matrix.index,
					 output_file_name = output_file_name, with_corr_values=False,
					 value_range=[-max_abs_value ,max_abs_value ],
					 xlabel = 'Psychiatric disorders',
					 ylabel='Acoustic features',
					 font_scale=7,
					 x_rotation=45 ,
					 label_size=8,
					 tight_layout=0.1,
					 plot_for_presentation=plot_for_presentation)