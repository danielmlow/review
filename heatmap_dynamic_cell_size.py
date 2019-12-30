#!/usr/bin/env python3

'''
Author: Daniel M. Low and here: https://github.com/drazenz/heatmap/blob/master/heatmap.py
license: Apache 2.0

'''

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def preprocess_table(path = None):
	df = pd.read_excel(path, sheet_name='heatmap',header=1)
	df = df.replace(np.nan, 0, regex=True)
	matrix = []
	matrix_amount = []
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
				score = np.mean(values)
				amount_of_studies = len(values)

				row_new.append(score)
				row_new_amount.append(amount_of_studies)
		matrix.append(row_new)
		matrix_amount.append(row_new_amount)
	matrix = pd.DataFrame(matrix, columns=df.columns[2:], index=df.iloc[:,1])
	matrix_amount = pd.DataFrame(matrix_amount, columns=df.columns[2:], index=df.iloc[:, 1])
	return matrix, matrix_amount




def custom_heatmap(x, y, **kwargs):
	if 'color' in kwargs:
		color = kwargs['color']
	else:
		color = [1]*len(x)

	if 'palette' in kwargs:
		palette = kwargs['palette']
		n_colors = len(palette)
	else:
		n_colors = 256 # Use 256 colors for the diverging color palette
		palette = sns.color_palette("Blues", n_colors)

	if 'color_range' in kwargs:
		color_min, color_max = kwargs['color_range']
	else:
		color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

	def value_to_color(val):
		if color_min == color_max:
			return palette[-1]
		else:
			val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
			val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
			ind = int(val_position * (n_colors - 1)) # target index in the color palette
			return palette[ind]

	if 'size' in kwargs:
		size = kwargs['size']
	else:
		size = [1]*len(x)

	if 'size_range' in kwargs:
		size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
	else:
		size_min, size_max = min(size), max(size)

	size_scale = kwargs.get('size_scale', 500)

	def value_to_size(val):
		if size_min == size_max:
			return 1 * size_scale
		else:
			val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
			val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
			return val_position * size_scale
	if 'x_order' in kwargs:
		x_names = [t for t in kwargs['x_order']]
	else:
		x_names = [t for t in sorted(set([v for v in x]))]
	x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

	if 'y_order' in kwargs:
		y_names = [t for t in kwargs['y_order']]
	else:
		y_names = [t for t in sorted(set([v for v in y]))]
	y_to_num = {p[1]:p[0] for p in enumerate(y_names)}


	plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
	ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

	marker = kwargs.get('marker', 's') #square

	kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
		 'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
	]}
	# Set features with 0 studies to -1 so they don't have a box
	size2 = []
	for i in size:
		if i==0:
			size2.append(-1)
		else:
			size2.append(i)

	ax.scatter(
		x=[x_to_num[v] for v in y],
		y=[y_to_num[v] for v in x],
		marker=marker,
		s=[value_to_size(v) for v in size2],
		c=[value_to_color(v) for v in color],
		edgecolors='k',
		linewidth='0.01',
		**kwargs_pass_on
	)
	ax.set_xticks([v for k,v in x_to_num.items()])
	ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right',size=8)
	ax.set_yticks([v for k,v in y_to_num.items()])
	ax.set_yticklabels([k for k in y_to_num],size=7)

	blue = 'dodgerblue'
	red = 'firebrick'
	if plot_for_presentation:
		# show less for in slide
		colors = ['k', 'k', 'w', 'w', 'w', 'w', 'w', 'w', 'w',
		          'k', 'w', 'w', 'w', 'w', 'w', 'w',
		          'w',
		          'k', 'k', 'w', 'w', 'k', 'w', 'w', 'w', 'k', 'w', 'w', 'w', 'w', 'w', 'w', 'k']
	else:
		colors = [blue,blue,blue, blue,blue,blue,blue,blue,blue,
				  red, red, red, red, red, red, red,
				  'purple',
				  'k','k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', ]

	colors = colors[::-1]

	for xtick, color in zip(ax.get_yticklabels(), colors):
		xtick.set_color(color)

	ax.set_ylabel('Acoustic features')
	ax.set_xlabel('Psychiatric disorders')


	ax.grid(False, 'major') # grid within cells
	ax.grid(True, 'minor', color = 'lightgrey', linewidth = 0.01) #grid between cells
	ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True) #show grid between cells
	ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
	ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
	ax.set_facecolor('#ffffff')  # white #BACKGROUND COLOR

	# Add color legend on the right side of the plot
	if color_min < color_max:
		ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot
		col_x = [0]*len(palette) # Fixed x coordinate for the bars
		bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars
		bar_height = bar_y[1] - bar_y[0]
		ax.barh(
			y=bar_y,
			width=[2]*len(palette), # Make bars 5 units wide
			left=col_x, # Make bars start at 0
			height=bar_height,
			color=palette,
			linewidth=0,
		)
		ax.set_xlim(1.99, 1) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
		ax.set_ylim(-1, -0)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
		ax.grid(False) # Hide grid
		ax.set_facecolor('white') # Make background white
		ax.set_xticks([]) # Remove horizontal ticks
		ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
		ax.yaxis.tick_right()  # Show vertical ticks on the right

	plt.savefig('./data/outputs/heatmap.'+format, dpi=1200, bbox_inches='tight')


def corrplot(matrix, amount_of_studies, size_scale=500, marker='s'):
	corr = pd.melt(matrix.reset_index(), id_vars='index')
	corr.columns = ['x', 'y', 'value']
	corr2 = pd.melt(amount_of_studies.reset_index(), id_vars='index')
	corr2.columns = ['y', 'x', 'value']
	custom_heatmap(
		x = corr['x'], y = corr['y'],
		color=corr['value'],
		color_range=[-1, 1],
		palette=sns.diverging_palette(220, 20, n=256, center='light'),
		size=corr2['value'].abs(), size_range=[0,np.max(np.max(amount_of_studies.abs()))],
		marker=marker,
		x_order=matrix.columns,
		y_order=matrix.index[::-1],
		size_scale=size_scale,
	)



if __name__ == "__main__":
	# Config
	plot_for_presentation = False
	output_file_name = 'features_by_disorders_dynamic'
	format = 'pdf' # I then convert to jpeg
	input_dir = 'data/inputs/'

	output_dir = 'data/outputs/'
	input_file = 'speech_psychiatry_heatmap.xlsx'
	path = input_dir + input_file

	# Clean dataframe
	matrix, amount_of_studies = preprocess_table(path = path)
	matrix = pd.DataFrame(matrix, index= matrix.index.rename(name='index'))
	amount_of_studies = pd.DataFrame(amount_of_studies , index=amount_of_studies .index.rename(name='index'))

	matrix = matrix.reindex(index=matrix.index)
	corrplot(matrix, amount_of_studies, size_scale=90)
