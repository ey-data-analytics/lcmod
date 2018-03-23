import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from lcmod.policy import *
#import projection_funcs as pf
#import policy_tools as pt
#import inspect
from copy import deepcopy
#


def bigplot(scens, res_df, shapes_df, name=None, _debug=False):
	'''Makes three plots

	   Shapes, based on passed shapes_df (or will make one)

	Cumulative spend, based on past results df

	Annual diffs vs first scenario
	'''
	if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
	if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

	if shapes_df is None:
shapes_df = make_shapes(scens, flat=True, multi_index=True).sort_index(axis=1)


# MAKE A TABLE WITH PARAMETERS & SUMMARY
params_table1 = make_params_table(scens).append(res_df.groupby(res_df.index.year).sum().iloc[:5,:])
	params_table1


	fig = plt.figure(figsize=(10,10), dpi=200)
legend = list(shapes_df.columns.levels[0])
	max_y = shapes_df.max().max()*1.1*144

	pad = 25

	if _debug: print('columns to plot are'.ljust(pad), shapes_df.columns)   

# get only lines we want
	right_lines = [x for x in shapes_df.columns.levels[1] if '_init' not in x]
	if _debug: print("right_lines".ljust(pad), right_lines)

# get the df sorted etc
sorted_df = shapes_df.sort_index(axis=1)

	for i, line in enumerate(right_lines): 
# this is the crucial operation which reorganises the df across scenarios
# eg grouping together EoL spendlines across baseline, option1, option2
# NB annualising here
		if _debug: print("\n" + "+"*10 + "\nLINE is".ljust(pad), line)
		if _debug: print("index is".ljust(pad), i)
		sub_df = sorted_df.xs(line, level=1, axis=1) *144

		if '_init' in line: 
		if _debug: print('exiting as contains init')
		break

		if _debug: print('sub_df'); print(sub_df.head(), "\n")
# make the plot
ax = plt.subplot2grid((3, 3),(0,i))
#         ax = plt.subplot2grid((4, 4),(3,i), rowspan=0)
	for j in sub_df.columns:
	if _debug: print('\nnow in sub_df col'.ljust(pad), j)


#  these are now double-annualised
	if j == 'baseline': # treat the baseline separately
	if _debug: print('plotting dfcol (base)'.ljust(pad), j)
	if _debug: print('data'); print(sub_df[j].head())
	ax.plot(sub_df.index/12, sub_df[j], color='black')
	else:
	if _debug: print('plotting dfcol (not base)'.ljust(pad), j)
	if _debug: print('data'); print(sub_df[j].head())
ax.plot(sub_df.index/12, sub_df[j], alpha=0.75)

	ax.set_title(line + " cohorts")
	ax.set_xlabel('years post launch')
ax.set_ylim(0,max_y)
	if i == 0: 
ax.legend(legend)
#     if i == 0: ax.legend([p for p in pols])
	ax.set_ylabel('£m, annualised')
	else: ax.yaxis.set_tick_params(label1On=False)

# SECOND ROW: cumulative spend



ax = plt.subplot2grid((3, 3),(1,0), colspan=2)
#     ax = plt.subplot2grid((4, 4),(0,2), rowspan=2, colspan=2)
	plot_cumspend_line(res_df, plot_pers=60, annualise=True, ax=ax) # annualise
	ax.set_title('Annualised net spend on future launches')
ax.legend(legend)
	ax.set_ylabel('£m, annualised')

# THIRD ROW: annual diffs
# get data grouped by scenario (aggregating over spendlines)
	data = deepcopy(res_df.groupby(axis=1, level=0).sum())
ax = plt.subplot2grid((3, 3),(2,0), colspan=2)
#     ax = plt.subplot2grid((4, 4),(2,2), rowspan=3, colspan=2)
plot_ann_diffs(data, ax=ax, net_spend=True, legend=legend[1:], table=True)

fig.subplots_adjust(hspace=0.6, wspace=0.3)
	if name is not None:
	fig.savefig('figs/' + name + '.png')

##_________________________________________________________________________##


	def plot_cumspend_line(res_df, annualise=True, net_spend=False, plot_pers=None,
			fig=None, ax=None, figsize=None, return_fig=False, save_path=None, _debug=True):
		'''Plots a  line graph of scenarios, summing across spendlines.

		   Input is a dataframe of results.  Will be summed for scenarios (level 0 of col multi-index)

Can either generate a new plot, or add to existing axis (in which case pass ax)

	Can either generate projections and index from the policy, or use existing if passed

	Limit time interval by specifying plot_pers
	'''
	if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
	if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")   

	pad=20

# need to avoid actually changing res_df
	ann_factor = 1
	if annualise: ann_factor = 12

	if plot_pers is None: plot_pers = len(res_df)
	if _debug: print('plot pers'.ljust(pad), plot_pers)

	ind = res_df.index.to_timestamp()[:plot_pers]

# sum for the scenarios - highest level of column multi-index
	scen_lines = res_df.groupby(level=0, axis=1).sum().iloc[:plot_pers, :] * ann_factor
	if _debug: print('scen_lines:\n', scen_lines.head())

# create fig and ax, unless passed (which they will be if plotting in existing grid)
	if fig is None and ax is None:
fig, ax = plt.subplots(figsize=figsize)

	for i, p in enumerate(scen_lines):
		if i==0: 
		ax.plot(ind, scen_lines[p].values, color='black') 
		else:
			ax.plot(ind, scen_lines[p].values, alpha=0.75)  

			for t in ax.get_xticklabels():
				t.set_rotation(45)

				ax.legend(scen_lines.columns)
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	title_str = ""
	if net_spend: title_str = " net"

	ax.set_title("Accumulated{} spend".format(title_str))

	if save_path is not None:
fig.savefig(save_path)

	if _debug: print("\nLEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
	if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

	if return_fig: return(fig)


##_________________________________________________________________________##


	def plot_ann_diffs(projs, max_yrs=5, fig=None, ax=None, figsize=None, 
			table=False, legend=None, net_spend=False, return_fig=False, save_path=None, _debug=True):
	'''Plots a bar chart of annual data, subtracting the first column
	   Can either generate a new plot, or add to existing axis (in which case pass ax)
	   '''
	   if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
	   if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

		      diffs = projs.iloc[:,1:].subtract(projs.iloc[:,0], axis=0)
		      diffs = diffs.groupby(diffs.index.year).sum().iloc[:max_yrs,:]

		      ind = diffs.index

# set the name of the counterfactual
		      col_zero = projs.columns[0]
		      if isinstance(col_zero, tuple):
			      counterfactual_name = col_zero[0]
			      else: counterfactual_name = col_zero

# create fig and ax, unless passed (which they will be if plotting in existing grid)
			      if fig is None and ax is None:
fig, ax = plt.subplots(figsize=figsize)

num_rects = len(diffs.columns)
	rect_width = 0.5
	gap = 0.45
	for i, x in enumerate(diffs):
	print(diffs[x])
	rect = ax.bar(diffs.index + ((i/num_rects)*(1-gap)), diffs[x], 
			width=rect_width/num_rects) 
	title_str = ""
	if net_spend: title_str = " net"

	ax.set_title("Difference in{} annual spend vs ".format(title_str) + counterfactual_name +", £m")
	ax.tick_params(axis='x', bottom='off')
	ax.grid(False, axis='x')
# for t in ax.get_xticklabels():
#     t.set_rotation(45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	if legend is not None:
ax.legend(legend)

	else:
	ax.legend(diffs.columns)
if len(diffs.columns)>2:  ax.legend(diffs.columns)

	if table:
ax.set_xticks([])

	rows = []
	for x in diffs:
	rows.append(["{:0,.0f}".format(y) for y in diffs[x]])

	row_labs = None
	if legend: row_labs = legend
	else: row_labs = diffs.columns

	c_labels = list(diffs.index)
	tab = ax.table(cellText=rows, colLabels=c_labels, rowLabels= row_labs)
	tab.set_fontsize(12)
tab.scale(1,2)
	tab.auto_set_font_size

	if save_path is not None:
fig.savefig(save_path)

	if _debug: print("\nLEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
	if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

	if return_fig: return(fig)


##_________________________________________________________________________##
