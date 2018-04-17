import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from copy import deepcopy
import inspect

from lcmod.policy import make_shapes, make_params_table
from lcmod.core import r_trend



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
    plot_cumspend_line(res_df, plot_pers=60, annualise=True, ax=ax, _debug=_debug) # annualise
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
                        fig=None, ax=None, figsize=None, return_fig=False, save_path=None, _debug=False):
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
                    table=False, legend=None, net_spend=False, return_fig=False, save_path=None, _debug=False):
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

##____________________________________________________________________________________________________________##


def plot_rset_projs(rs_dict_in, rsets=None, selection=None, num_plots=12, n_pers=120, xlims=None, show_phases=True,
                    max_sel=True, agg_filename=None, file_suffix=None, out_folder='figs/test_plot_rset/', _debug=False):
    
    '''Helper function for plotting projections of individual products in rulesets.

    For a dict of rulesets, sorts by max spend and takes num_plots highest.
    Then for each generates a separate plot containing each product in the high spend set.

    PARAMETERS

    rs_dict_in [reqd]     : a dict of rulesets (i.e. the normal core output from a projection)

    rsets                 : can pass a list of keys (eg 'biol_sec'), and will use only those, 
                            rather than going through whole of rs_dict_in

    selection             : pass a list of product names instead of finding products with max spend

    num_plots             : number of plots for each ruleset

    agg_filename          : puts together in a single image file if pass a filename 
                             - most useful wwhen passing a selection, to get a single file of output
                             
                             [makes the code a bit complex so dev notes:]
                             - will then find instances of the selection elements across all the rulesets
                             - so need to extend selection names to include all instances, eg a name may occur in several. 
                             - Do this with ext_selection. 
                             - Also use the length of this to set the number of plots, and 
                             - use ext_selection to hold the annotated names (identifying rset they were found it)
                             - also use an additional iterator, agg_i, which keeps incrementing across different rsets

    n_pers                 : number of periods to project for

    xlims                  : x axis limits for plots

    show_phases            : will make coloured areas showing uptake and launch phases

    file_suffix            : gets added onto output file names.  Useful if want to distinguish from prev plots    

    out_folder             : destination folder

    max_sel                : sort a selection by max spend, if passed
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")   

    rcParams.update({'font.size': 8})

    pad = 35

    # first select only passed rsets  
    if rsets is not None:
        rs_dict = {x:rs_dict_in[x] for x in rsets}

    # otherwise get rid of fut
    else:
        rs_dict = {x:rs_dict_in[x] for x in rs_dict_in.keys() if 'fut' not in x}

    if _debug: print(rs_dict.keys())

    # set up to make a single figure if aggregating
    if agg_filename is not None:
        if selection is not None:

            # tricky.  Need to know how many across all rsets
            ext_selection = []
            for r in rs_dict_in:
                ext_selection.extend([x + " - " + r for x in rs_dict_in[r].past.index.get_level_values('molecule') 
                                        for s in selection if s in x])

            if _debug: print('extended list\n', ext_selection)

            num_plots = len(ext_selection)
            if _debug: print('num_plots (selection)'.ljust(pad), num_plots)

        else: 
            num_plots = len(rs_dict * num_plots)
            if _debug: print('num_plots (rulesets)'.ljust(pad), num_plots)

        # set up a single pdf output file (if not agg, will do one anew for each ruleset)
        pdf = PdfPages(out_folder + agg_filename + '.pdf')

    # make an iterator for if aggregating in a single fig
    agg_i = 0

    # iterate non future rsets
    for r in [x for x in rs_dict.keys() if not x.endswith('_fut')]:
        # select data by max spend
        print('\nin rset'.ljust(pad), r)

        selected = None

        # use selection if passed
        if selection is not None:
            selected = rs_dict[r].past.loc[selection]

            if _debug:
                print('index without sort'.ljust(pad), selected.index.get_level_values(0))

            if max_sel:
                selected = selected.loc[selected.max(axis=1).sort_values(ascending=False).index]
                if _debug:
                    print('index with sort'.ljust(pad),
                        selected.index.get_level_values(0))

        # otherwise take top by spend
        else:
            selected = rs_dict[r].past.loc[rs_dict[r].past.max(axis=1).sort_values(ascending=False).head(num_plots).index]
        
        if _debug: print('length of selection'.ljust(pad), len(selected))
        #get into £m annualised
        selected *=12/10**8
                      
        # get the LIST OF dfs from r_trend with an _out_type='df' flag
        df_list = r_trend(selected, n_pers=n_pers,
                                   shed = rs_dict[r].f_args['shed'],
                                   term_gr = rs_dict[r].f_args['term_gr'],
                                   loe_delay = rs_dict[r].f_args['loe_delay'],
                                   _out_type='df')
        

        plat_dur = rs_dict[r].f_args['shed'].plat_dur
        total_dur = rs_dict[r].f_args['shed'].uptake_dur + plat_dur
        if _debug: print('plat length is '.ljust(pad), plat_dur)
        if _debug: print('total length is '.ljust(pad), total_dur)
        
        # make pdf for this ruleset (remember in a loop here already), if not already made one for aggregate
        if agg_filename is None:
            if file_suffix is None: fs = ""
            else: fs = file_suffix
            pdf = PdfPages(out_folder + r + fs + '.pdf')

        # now loop through the returned dataframes - one per product (each with 3 cols / lines to plot)
        for i, df in enumerate(df_list):

            if selected.iloc[i].name[0].lower() == 'not applicable':
                print('found product named "not applicable" - ignoring')
                continue

            # first get rid of zeros for cleaner plotting
            df = df[df!=0]

            if _debug: print('\ndf number'.ljust(pad), i)
            if _debug: print('..corresponding product name'.ljust(pad), selected.iloc[i].name[0])

            loe_date = pd.Period(selected.iloc[i].name[6], freq='M')
            if _debug: print('..with loe'.ljust(pad), loe_date)

            # make the index, from the selected input dataframe, adding n_pers
            ind_start = selected.columns[0]
            if _debug: print('ind start'.ljust(pad), ind_start)
            ind = pd.PeriodIndex(start=ind_start, periods = len(selected.columns) + n_pers).to_timestamp()
            if _debug: print('ind end'.ljust(pad), ind[-1])
            if _debug: print('total periods'.ljust(pad), len(ind))

            # snip df to length of index - THERE IS A STRAY PERIOD COMING FROM SOMEWHERE

            if _debug: print('length of dfs'.ljust(pad), len(df))
            if len(df) > len(ind):
                if _debug: print('snipping df.. ..')
                df = df.iloc[:len(ind), :]
                if _debug: print("length now".ljust(pad), len(df))

            ind_end = pd.Period(ind[-1], freq='M')
            if _debug: print('index end'.ljust(pad), ind_end)

            # # make an axes iterator that works with case if single plot
            # ax_it = None
            # if num_plots == 1:  ax_it = ax
            # elif agg_filename:  ax_it = ax[agg_i]
            # else:               ax_it = ax[i]

            # make a figure for this df (remember, one product, 3 lines)
            fig, ax = plt.subplots(dpi=200)

            # and now loop through the actual columns in the dataframe for the product
            for col in df:
                line = ax.plot(ind, df[col], linewidth=1)
                if col == 'projected': line[0].set_color('darkorchid')

            plot_name = selected.iloc[i].name[0]
            if agg_filename: plot_name = ext_selection[agg_i]
            else: plot_name = selected.iloc[i].name[0]


            ax.set_title(plot_name + ", loe: " + str(loe_date))
            if i%4 == 0:
                ax.legend(['actual', 'mov ave.', 'projected'])
            pat_exp = pd.Period(selected.iloc[i].name[6], freq='M')
            lim_0 = max(ind_start, (pat_exp - total_dur)).to_timestamp()
            lim_1 = max(ind_start, (pat_exp - plat_dur)).to_timestamp()
            lim_2 = min(ind_end, max(ind_start, (pat_exp))).to_timestamp()
            lim_3 = None

 
            if lim_1 > ind_start.to_timestamp() and show_phases: 
                ax.axvspan(lim_0, lim_1, facecolor='g', alpha=0.1)
                # only draw the line if in scope
                if lim_1 < ind_end.to_timestamp():
                    ax.axvline(x=lim_1, linestyle='--', linewidth=1, color='gray')

            if lim_2 > ind_start.to_timestamp(): 
                if show_phases: ax.axvspan(lim_1, lim_2, facecolor='r', alpha=0.1)
                
                # only draw the line if in scope
                if lim_2 < ind_end.to_timestamp():
                    ax.axvline(x=lim_2,  linewidth=2, linestyle='--', color='seagreen')

            ax.set_ylim(0)

            if xlims is not None:
                ax.set_xlim(xlims)

            agg_i +=1

            # save to the pdf, and clear the plot
            pdf.savefig()
            plt.close()

        if agg_filename is None:
            pdf.close()

    if agg_filename is not None:
        pdf.close()

    if _debug: print("\nLEAVING:  ", inspect.stack()[0][3])





##_________________________________________________________________________##
