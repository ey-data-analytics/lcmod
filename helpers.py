import pandas as pd

from lcmod.classes import *
from lcmod.core import *
from lcmod.misc import *


def load_spend_dset(path='c://Users//groberta//Work//data_accelerator/spend_data_proc/consol_ey_dset/adjusted_consol_dset_15MAR18.pkl', 
                    add_start_m=True, trim_to_financial_year='2017-03', _debug=True):
    '''Helper function to load the spend dataset using latest inputs - which have been adjusted to match DH Finance (on latest iteration)

    Option to trim to the last month of DH Finance projections - which is the last point at which we have a proper multiplier

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