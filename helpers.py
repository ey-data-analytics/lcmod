import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

from lcmod.classes import *
from lcmod.core import *


def load_spend_dset(path='C://Users//groberta//Work//data_accelerator//refactoring//adjusted_consol_dset_15MAR18.pkl', 
                    add_start_m=True, trim_to_financial_year='2017-03', _debug=True):
    '''Helper function to load the spend dataset using latest inputs - which have been adjusted to match DH Finance (on latest iteration)

    Option to trim to the last month of Finance projections - which is the last point at which we have a proper multiplier

    Currently adds a start month (optionally, default true).  This is needed to make slices of the df in the rulesest, 
    but may be better done in generating the original spend dset. (TODO) 
    '''
    pad = 40

    if _debug:
        print('Path for inputs spend df\n-->', path)

    df = pd.read_pickle(path)

    if _debug: 
        print('\nlast period is'.ljust(pad), df.columns[-1], end="")

    if trim_to_financial_year:
        df = df.loc[:,:trim_to_financial_year]

        if _debug: 
            print('\ntrimmed, last period now'.ljust(pad), df.columns[-1])

    # add a start month
    if add_start_m:
        start_m_ind = pd.PeriodIndex(df.index.get_level_values(level='adjusted_launch_date'), freq='M', name='start_month')
        df.set_index(start_m_ind, append=True, inplace=True)
        if _debug:
            print('\nadded a start month, index now\n-->', df.index.names)

    return df

##____________________________________________________________________________________________________________##


def make_rsets(df, params_dict, 
                xtrap=False, return_all_sums = False, return_setting_sums=False, return_sum = False,
                trim=False, _debug=False):
    '''Helper function to make rulesets objects based on input parameters

    Default just returns a dict of rulesets with the params added.

    Setting `xtrap` flag will return dict of rulesets populated with results of calling xtrap on each
     - i.e. it will actually do the extrapolation functions and assign to .past, .fut, .joined and .sum

    Setting `return_sum` or `return_setting_sums` flags will return summed datasets as expected.

    Setting `trim` with a tuple or list (start, end) will trim the output sum dfs
    '''

    cut_off = params_dict['cut_off']
    biol_shed = params_dict['biol_shed']
    non_biol_shed = params_dict['non_biol_shed']
    loe_delay = params_dict['loe_delay']
    biol_term_gr = params_dict['biol_term_gr']
    non_biol_term_gr = params_dict['non_biol_term_gr']
    biol_coh_gr = params_dict['biol_coh_gr']
    non_biol_coh_gr = params_dict['non_biol_coh_gr']
    n_pers = params_dict['n_pers']

    rsets = {}
    # existing product rulesets - NB MUST SET CUTOFF TO MINUS 1 TO AVOID OVERLAP
    rsets['biol_sec'] = RuleSet(df, name='biol_sec', 
                               index_slice = dict(biol=True, setting='secondary', start_month=slice(None, cut_off-1, None)),
                               func = r_trend,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, loe_delay=loe_delay))

    rsets['nonbiol_sec'] = RuleSet(df, name='nonbiol_sec', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(None, cut_off-1, None)),
                               func = r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, loe_delay=loe_delay))

    rsets['biol_prim'] = RuleSet(df, name='biol_prim', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = r_trend,
                               f_args = dict(shed=biol_shed, term_gr=biol_term_gr, loe_delay=loe_delay))

    rsets['nonbiol_prim'] = RuleSet(df, name='nonbiol_prim', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(None, cut_off-1, None)),
                               func = r_trend,
                               f_args = dict(shed=non_biol_shed, term_gr=non_biol_term_gr, loe_delay=loe_delay))

    # future launches rulesets
    rsets['biol_sec_fut'] = RuleSet(df, name='biol_sec_fut', 
                               index_slice = dict(biol=True, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = r_fut_tr,
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr, cut_off=cut_off))

    rsets['nonbiol_sec_fut'] = RuleSet(df, name='nonbiol_sec_fut', 
                               index_slice = dict(biol=False, setting='secondary', start_month=slice(cut_off, None, None)),
                               func = r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr, cut_off=cut_off))

    rsets['biol_prim_fut'] = RuleSet(df, name='biol_prim_fut', 
                               index_slice = dict(biol=True, setting='primary', start_month=slice(cut_off, None, None)),
                               func = r_fut_tr,
                               f_args = dict(shed=biol_shed, loe_delay=loe_delay, term_gr=biol_term_gr, coh_gr=biol_coh_gr, cut_off=cut_off))

    rsets['nonbiol_prim_fut'] = RuleSet(df, name='nonbiol_prim_fut', 
                               index_slice = dict(biol=False, setting='primary', start_month=slice(cut_off, None, None)),
                               func = r_fut_tr,
                               f_args = dict(shed=non_biol_shed, loe_delay=loe_delay, term_gr=non_biol_term_gr, coh_gr=non_biol_coh_gr, cut_off=cut_off))
    
    if xtrap or return_all_sums or return_setting_sums or return_sum: 
        for r in rsets:
            if _debug: print('xtrapping rset ', r, end=" ")
            rsets[r].xtrap(n_pers, _debug=False)
            if _debug: print(' ..OK')

    # if any sums reqd, make the full set
    if return_all_sums or return_setting_sums or return_sum:
        if _debug: print('making all sums')
        sums = pd.concat([rsets[x].summed for x in rsets], axis=1)
        if trim:
            sums = sums.loc[slice(pd.Period(trim[0], freq='M'), pd.Period(trim[1], freq='M'),None),:]

    # if all sums reqd, just return
    if return_all_sums:
        return sums

    # if sums by setting reqd
    elif return_setting_sums:
        if _debug: print('making sums by setting')
        sums = sums.groupby(lambda x: 'sec' in x, axis=1).sum()
        sums.columns = ['Primary', 'Secondary']
        return sums

    # if a single sum reqd
    elif return_sum: 
        if _debug: print('making single sum')
        sums = sums.sum(axis=1)
        if return_setting_sums or return_all_sums:
            print('Returning single sum only - to return all sums, or by setting you need to turn off the `return_sum` flag')
        return sums


    # default returns the whole rulesets (with or without dfs depending on if xtrap was called)    
    else: return rsets


##____________________________________________________________________________________________________________##


def prod_info(drugs, df, details=False, pdf_save_path=None, save_path=None):
    '''Returns summary info and plots for a list of drugs in a non-indexed dataframe (i.e. all columns).

    Requirements for df - just needs to be able to find the following column headings:
    ['molecule', 'date', 'setting', 'vol', 'spend']
    '''

    # First test if it's a list.  If it's a single product name string, then make it a list    
    if not isinstance(drugs, list):
        drugs = [drugs]

    colors = dict(primary='seagreen', secondary='darkred')

    if pdf_save_path is not None:
        pdf = PdfPages(pdf_save_path)

    fig, ax = plt.subplots(nrows=len(drugs), ncols=3, figsize=(15,4*len(drugs)))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, drug in enumerate(drugs):
        print(drug)

        # process the data
        if details: print(df.loc[df['molecule']==drug].iloc[0])
        drug_df = (df.loc[df['molecule']==drug, ['date', 'setting', 'vol', 'spend']]
                   .set_index(['date', 'setting'], drop=True))
        
        loe_date = df.loc[df['molecule']==drug].iloc[0]['loe_date']

        vols = drug_df.pivot_table(values='vol', columns='setting', index='date')
        spend_gbp = 0.01*drug_df.pivot_table(values='spend', columns='setting', index='date')
        price_gbp = spend_gbp / vols
        ind = spend_gbp.index
        
        # make the plot

        # SPEND
        if len(drugs) == 1: it = (0)
        else: it = (i,0)
  
        for s in spend_gbp.columns:
            line = ax[it].plot(spend_gbp.index, spend_gbp[s]*12/1000000, alpha=0.7, color=colors[s])

        if i%5==0: ax[it].set_title('spend, £m annualised')
        ax[it].legend(spend_gbp.columns)
        for t in ax[it].get_xticklabels():
            t.set_rotation(45)
        ax[it].set_ylim(0)
        ax[it].set_ylabel(drug + ", loe  " + str(loe_date)[:7])

        if loe_date > price_gbp.index[0] and loe_date < price_gbp.index[-1]:
            ax[it].axvline(loe_date)

        # PRICE
        if len(drugs) == 1: it = (1)
        else: it = (i,1)
     
        for j,s in enumerate(price_gbp.columns):
            if j==1:
                ax2 = ax[it].twinx()
                ax2.plot(price_gbp.index, price_gbp[s], alpha=0.7, color=colors[s])
                ax2.set_ylim(0)
            else:
                line = ax[it].plot(price_gbp.index, price_gbp[s], alpha=0.7, color=colors[s])

        if i%5==0: ax[it].set_title('price, £')
        # else: ax[it].set_title('loe= ', loe_date)
        ax[it].legend(price_gbp.columns)
        for t in ax[it].get_xticklabels():
            t.set_rotation(45)    
        ax[it].set_ylim(0)
        ax[it].legend([])

        if loe_date > price_gbp.index[0] and loe_date < price_gbp.index[-1]:
            ax[it].axvline(loe_date)


        # VOL
        if len(drugs) == 1: it = (2)
        else: it = (i,2)

        for j,s in enumerate(vols.columns):
            if j==1:
                ax2 = ax[it].twinx()
                ax2.plot(vols.index, vols[s], alpha=0.7, color=colors[s])
                ax2.set_ylim(0)
            else:
                line = ax[it].plot(vols.index, vols[s], alpha=0.7, color=colors[s])

        if i%5==0: ax[it].set_title('volume')
        ax[it].legend(vols.columns)
        for t in ax[it].get_xticklabels():
            t.set_rotation(45)    
        ax[it].set_ylim(0)
        ax[it].legend([])

        if loe_date > price_gbp.index[0] and loe_date < price_gbp.index[-1]:
            ax[it].axvline(loe_date)


    if pdf_save_path is not None:
        pdf.savefig()
        pdf.close()

    if save_path is not None:
        fig.savefig(save_path)




##_________________________________________________________________________##

def plot_hi_lo_base(in_dict, df=None, 
                        trim=('1-2010', '12-2023'), figsize=(12,6), ybarlims=None,
                        outfile=None, fig=None, ax=None, bars=True, return_fig=False, _debug=False):

    '''Pass in a dictionary of scenarios with structure as follows:

    in_dict = { 'hi': {'params': <parameters for constructing rsets and getting projections>,
                       'sums':   <summed projections>,
                       'color':  <color to plot this scenario>,
                       'legend': <how this should be labelled in the legend>},

                'baseline': {'params'.. etc},

                'lo': {'params'.. etc}
                }
    
    Keys of in_dict can be whatever - used as the names of the scenarios.  
      (If one has 'base' it gets a thick line.)

    Need either sums or (params plus a dataframe - which can calc sums).  
    See Sensitivity Analysis notebook.

    '''

    pad = 30

    # make a df of sums
    sums_list = []

    for s in in_dict:

        this_sum = None

        if _debug: print('in rset'.ljust(pad), s)

        # check if missing sums, and get them if reqd.  Append to the df
        if in_dict[s].get('sums', None) is None:

            if _debug: print('no sum detected, so making one')

            if df is None:
                print('Need a df and params for ', s, ' so stopping here'); return

            elif in_dict[s]['params'] is None:
                print('Need params for ', s, ' so stopping here'); return

            else: 
                this_sum = make_rsets(df, in_dict[s]['params'], return_sum=True, trim=trim)
                if _debug: print('got this\n', this_sum.head())

        else: 
            if _debug: print('found sums for this, so just going to trim it')
            this_sum = in_dict[s]['sums'].loc[slice(pd.Period(trim[0], freq='M'), pd.Period(trim[1], freq='M'),None)]

        this_sum.name = s

        # at this point must have the sums
        sums_list.append(this_sum)

    # so we have a df of results    
    res_df =  pd.concat(sums_list, axis=1)

    num_plots = 1
    if bars: num_plots = 2

    # FIRST THE LINE GRAPH
    if fig is None and ax is None:
        fig, ax = plt.subplots(num_plots, figsize=(figsize[0], figsize[1]*num_plots))

    ind = res_df.index.to_timestamp()

    ax_it = ax
    if bars: ax_it = ax[0]

    for i, s in enumerate(res_df):
        line = ax_it.plot(ind, res_df[s]*12/10**11, alpha=0.5)
        if in_dict[s].get('color', None) is not None:
            line[0].set_color(in_dict[s]['color'])
        if 'base' in s.lower(): line[0].set_linewidth(3)

    # do filling in
    # first get hold of the baseline
    base_out_name = [x for x in res_df.columns if 'base' in x.lower()]

    if len(base_out_name)==1: 
        base_out_name = base_out_name[0]
        for s in res_df.drop(base_out_name, axis=1):
            fill_color = in_dict[s].get('color', 'grey')
            ax_it.fill_between(ind, res_df[s]*12/10**11, res_df[base_out_name]*12/10**11, color=fill_color, alpha=0.15)

    leg = [in_dict[s].get('legend',s) for s in in_dict]
    ax_it.legend(leg)

    if bars: ax_it.set_xticks([])

    #  NOW THE BAR CHART
    if bars:
        ann_df = res_df.groupby(res_df.index.year).sum()
        diffs_df = ann_df.drop('baseline', axis=1).subtract(ann_df['baseline'], axis=0)/10**11

        ax_it = ax[1]

        bar_w = 1; gap = 0.4
        
        for i,s in enumerate(diffs_df.columns):
            rect = ax_it.bar(diffs_df.index, diffs_df[s].values, width=bar_w-gap, color=in_dict[s]['color'], alpha=0.3)

        if ybarlims is not None: ax_it.set_ylim(ybarlims)

        ax_it.set_xticks([])
        ax_it.set_xlim(diffs_df.index[0]-0.5, diffs_df.index[-1]+0.5)
        # ax_it.legend([l for l in leg if 'base' not in l])


        tab = ax_it.table(colLabels=diffs_df.index, 
                          cellText=[diffs_df[x].round(2).values for x in diffs_df],
                          rowLabels=diffs_df.columns)

        tab.set_fontsize(12)
        tab.scale(1,2)
        # tab.auto_set_font_size

        fig.subplots_adjust(hspace=0.05, wspace=0.3)

    if outfile: fig.savefig(outfile)

    if return_fig: return fig

##_________________________________________________________________________##

