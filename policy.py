import pandas as pd
import numpy as np
from openpyxl import load_workbook
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from collections import namedtuple
import inspect
import os

from lcmod.core import make_shape, get_forecast

def spend_mult(sav_rate, k1=None, k2=None, ratio=None):
    '''Returns the multiplier to apply to baseline spend 
    to reflect a change in threshold (k1->k2), given the savings (drugs substitution)
    as a proportion r of the original costs

    k1 = (C1 - s)/dQ : k is cost/QALY, c is cost of drug, s is savings
    s = rC1, r=s/C1  : r is savings as proportion of initial costs
    k1 = (1-r)C1     : assume dQ is 1 as effects are all relative
    C1 = k1/(1-r)
    k2 = C2 -rC1; C2 = k2 + rC1 = k2 + rk1/(1-r)
    C2/C1 = (k2 + rk1/(1-r))*(k1/(1-r)) = (1-r)(k2/k1) + r
    '''
    
    r = sav_rate

    if ratio is None:
        if (k1 is None) or (k2 is None): 
            print('need a ratio or k1 and k2')
            return
        
        else: ratio = k2/k1

    return ((1-r)*(ratio)) + r


##_________________________________________________________________________##



def dump_to_xls(res_df, in_dict, outfile, append=False, prefix="", shapes=None, log=True, 
    npv_rs=None, npv_yrs_lim=None, _debug=False):
    '''Dump a results DataFrame to an xls, with option to make shapes from a  
     dict of scenarios or from a scenario alone, and/or passing shapes

     PARAMETERS

     res_df         : the dataframe of results

     in_dict        : the scenarios dictionary used to generate the dataframe of results

     outfile        : the target file.  NOTE NEEDS TO END xlsx - will be coerced

     append         : if True will add sheets to the outfile target

     prefix         : helpful if adding sheets - will be prepended to sheet names
     
     shapes         : if a df of individual lifecycle shapes already exists, it can 
                      be passed in this parameter
    
     log            : if True will append any log info to the parameters header
   
     npv_rs         : pass a list of discount rates to get NPVs for all scenarios vs baseline
   
     npv_yrs_lim    : if NPVs are being calculated, this optionally limits the number of years.
                      Otherwise will be calculated over whole projection period

    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    # coerce if not xls.x
    if outfile.split('.')[-1] != 'xlsx':
        print('coercing outfile to .xlsx')
        outfile = outfile.split('.')[0] + '.xlsx'
        print('new name is ', outfile)

    # initialise vars
    params_header = None
    shapes_body = None

    # first get the params header and shapes, depending on input
    # if a policy is passed
    params_header = make_params_table(in_dict, log=log, _debug=_debug)
    shapes_body = make_shapes(in_dict, flat=True, multi_index=True)

    if _debug: print('\ngot header and shapes from in_dict:\n')
    if _debug: print("params_header\n", params_header, end="\n\n")
    if _debug: print("shapes_body\n", shapes_body.head(), end="\n")


    # if shapes are passed
    if shapes is not None:
        if _debug: print('preparing to overwrite shapes')
        shapes_body = shapes
        if _debug: print("new shapes_body\n", shapes_body.head(), end="\n")


    # assemble outputs
    shapes_out = params_header.append(shapes_body)
    main_out = params_header.append(res_df)
    annual_out = params_header.append(res_df.groupby(res_df.index.year).sum())
    scen_sums_out = res_df.groupby(level=0, axis=1).sum()
    scen_sums_ann_out = scen_sums_out.groupby(res_df.index.year).sum()
    scen_sums_ann_diff_out = scen_sums_ann_out.subtract(scen_sums_ann_out['baseline'],
              axis=0).drop('baseline', axis=1)

    if npv_rs is not None:

        # get limit of yrs to plot - full df unless limited by npv_yrs
        if npv_yrs_lim is None:
            npv_yrs_lim = len(scen_sums_ann_out)

        npv_df = pd.DataFrame(index=npv_rs, columns = scen_sums_ann_diff_out.columns)
        for npv_r in npv_rs:
            npv_df.loc[npv_r,:] = scen_sums_ann_diff_out.iloc[:npv_yrs_lim,:].multiply((1 - npv_r) ** np.arange(npv_yrs_lim), axis=0).sum()

        npv_df.index.name = 'disc rate'

    # write out - either appending or making a new one, or not appending
    if append:
        if os.path.isfile(outfile):
            book = load_workbook(outfile)
            writer = pd.ExcelWriter(outfile, engine="openpyxl")
            writer.book = book
        else: 
            print("could not find file to append to.  will make a new one")
            writer = pd.ExcelWriter(outfile)

    # not appending
    else: writer = pd.ExcelWriter(outfile)

    shapes_out.to_excel(writer, prefix + 'shapes')
    main_out.to_excel(writer, prefix + 'main')
    annual_out.to_excel(writer, prefix + 'annual')
    scen_sums_out.to_excel(writer, prefix + 'scen_sums')
    scen_sums_ann_out.to_excel(writer, prefix + 'scen_sums_ann')
    scen_sums_ann_diff_out.to_excel(writer, prefix + 'scen_sums_ann_diff')

    if npv_rs is not None:
        npv_df.to_excel(writer, prefix + 'NPVs')


    writer.save()

    if _debug: print("\LEAVING:  ", inspect.stack()[0][3])


##_________________________________________________________________________##

def dump_shapes(scens):
    '''for dictionary of scenarios, dumps a single df with all the shapes 
     - constructed from the Sheds etc, by make_shapes()

     TODO make this cope with dicts or list, with whatever depth
    '''
    
    all_shapes = pd.DataFrame()
    for s in scens:
        for l in scens[s]:
            all_shapes[s,l] = make_shapes([scens[s][l]])
    all_shapes.columns = pd.MultiIndex.from_tuples(all_shapes.columns)
    all_shapes.columns.names = ['scenario', 'spendline']
    all_shapes.index.name = 'period'
    return all_shapes


##_________________________________________________________________________##


def first_elem(in_struct):
    '''Returns the type first element of the input, be it a list or a dict
    '''

    if isinstance(in_struct, list): 
        return in_struct[0]
    
    elif isinstance(in_struct, dict):
        return in_struct[list(in_struct.keys())[0]]


##_________________________________________________________________________##


def flatten(struct, _debug=False):
    '''Return a flat dict of spendlines, keys are tuples of the hierarchical name
    '''
    out_dict = {}
    
    def _dig(in_struct, name=None):
        # if _debug: print('entering dig with', in_struct)
        if name is None: name = []
        


        if isinstance(first_elem(in_struct), dict):
            if _debug: print('in a dict')
            for s in in_struct:
                if _debug: print('digging to ', s)
                _dig(in_struct[s], name + [s])
                
        elif isinstance(first_elem(in_struct), list):
            if _debug: print('in a list')
            for s in in_struct:
                _dig(s, name + ['list'])     
                
        else: # can't get it to use isinstance to id spendline here so have to do by default :/
            if _debug: print('in a spendline I ASSUME - type ', type(first_elem(in_struct)))
            for l in in_struct:
                if _debug: print('element is ', l)
                out_dict[tuple(name + [l])] = in_struct[l]

   
    _dig(struct)

    return out_dict

###_________________________________________________________________________###


def make_log(in_dict, _debug=False):
    '''Return a df with log contents of an input dict containing spendlines
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    # first get a flat dict
    flat = flatten(in_dict)

    # instantiate the dataframe with the keys
    df = pd.DataFrame(columns=flat.keys())

    # now go through each spend line, and then each log entry
    for line in flat:
        if _debug: print('now in spend line'.ljust(pad), line)
        if _debug: print('log is'.ljust(pad), flat[line].log)
        for entry in flat[line].log:
            if _debug: print('now in log entry'.ljust(pad), entry)
            df.loc[entry, line] = flat[line].log[entry]
    
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    if _debug: print('leaving ', inspect.stack()[0][3])

    return df

##_________________________________________________________________________##

def make_params_table(pol_dict, index=None, log=False, _debug=False):
    '''Constructs a dataframe with parameters for spendlines in an input dict.

    There's a default set of row names - pass your own index if reqd

    TODO auto pick up index
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    if index is None:
        index = """peak_spend_pa peak_spend icer sav_rate 
                    uptake_dur plat_dur gen_mult launch_delay launch_stop
                    term_gr_pa term_gr_pm coh_gr_pa coh_gr_pm""".split()

    df = pd.DataFrame(index=index)

    flat_dict = flatten(pol_dict)

    for q in flat_dict:
        params = [flat_dict[q].peak_spend*12*12, # double annualised
                 flat_dict[q].peak_spend,
                 flat_dict[q].icer,
                 flat_dict[q].sav_rate,

                 int(flat_dict[q].shed.uptake_dur),
                 int(flat_dict[q].shed.plat_dur),
                 flat_dict[q].shed.gen_mult,
                 flat_dict[q].launch_delay,
                 flat_dict[q].launch_stop,

                 flat_dict[q].term_gr*12,
                 flat_dict[q].term_gr,
                 flat_dict[q].coh_gr*12,
                 flat_dict[q].coh_gr]

        df[q] = params

    
    if log: df = df.append(make_log(pol_dict, _debug=_debug))

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    if _debug: print('leaving ', inspect.stack()[0][3])

    return df


#_________________________________________________________________________##


def make_shapes(scens, term_dur=1, start_m=None, flat=True, synch_start=False, 
                net_spend=True, multi_index=True, _debug=False):
    '''For an input dict of scenarios (scens), return a df of shapes,
    ensuring differential launch times are handled.
    
    term_dur    : minimum number of terminal periods to plot
    start_m     : optional start month, otherwise range index
    synch_start : option to move index start according to negative launch delays
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    if flat: scens = flatten(scens, _debug=False)

    pads = [25] + [8]*4
    out = pd.DataFrame()

    # work out the global dimensions: maximum overhang beyond zero, and max length of shed
    # for x in scens: 
    # if _debug: print("in line".ljust(pad), x)
    max_hang = min([scens[x].launch_delay for x in scens])
    max_len = max([(scens[x].shed.uptake_dur + scens[x].shed.plat_dur)  for x in scens])
    if _debug: print("max hang".ljust(pad), max_hang)
    if _debug: print("max len".ljust(pad), max_len)

    # use the global dimensions to construct consistent dimensions for each shape
    for x in scens:
        if _debug: print('launch_delay'.ljust(pads[0]), scens[x].launch_delay)
        shed_len = scens[x].shed.uptake_dur + scens[x].shed.plat_dur 
        z_pad = scens[x].launch_delay - max_hang
        term_pad = max_len + term_dur - scens[x].launch_delay - shed_len
        total = z_pad + shed_len + term_pad
        
        if _debug: 
            print('line'.ljust(pad), x)
            print('shed_len'.ljust(pad), shed_len)
            print('l_delay'.ljust(pad), scens[x].launch_delay)
            print('z_pad'.ljust(pad), z_pad)
            print('term_pad'.ljust(pad), term_pad)
            print('total'.ljust(pad), total)

        
        out[x] = make_shape(shed = scens[x].shed, 
                               z_pad = z_pad, 
                               peak_spend = scens[x].peak_spend, 
                               sav_rate = scens[x].sav_rate,
                               net_spend = net_spend,
                               term_pad=term_pad, 
                               term_gr=scens[x].term_gr,
                               _debug = _debug)
                  

    if start_m is not None:
        if synch_start: 
            start_m = pd.Period(start_m, freq='M') + max_hang
            if _debug: print('new start m ', start_m)
        
        out.index = pd.PeriodIndex(freq='M', start = pd.Period(start_m, freq='M'), periods=total)   

    if multi_index: out.columns = pd.MultiIndex.from_tuples(out.columns)     

    if _debug: print("\LEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")      

    return out


###_________________________________________________________________________##



def project_policy(policy, start_m='1-2019', n_pers=120, shapes=None,
                    synch_start=False, diffs_out=False, annual=False, net_spend=True,
                    nans_to_zero=True, multi_index=True, _debug=False):  
    '''For a list of spendlines (with a start month and number of periods),
    returns a df with projections.
    
    The function orients individual  projections according to the launch delays
    across the whole set of spendlines, inserting spacers as necessary.

    Does not apply any savings assumptions - these need to happen before or after.

    Note this can't work on just shapes - needs terminal growth and coh growth
    
    PARAMETERS
    
    policy     : a list of spendlines (see SpendLine class)
    
    start_m    : start month for the projections 
                    (i.e. period zero for the spendline with lowest launch delay)
               : format eg '3-2019' is march 2019 (pd.Period standard)
    synch_start:
               : if the spendlines have a negative minimum launch delay
               :  (i.e. if one involves bringing launch forward) then move the 
               :  actual start month forward from the input.
               :  This is like saying that start_m is actually period zero, but 
               :  earlier periods are possible.  This lets different sets of policies be
               :  oriented correctly in time.
               : Note this goes through make_shapes
              
    n_pers     : number of periods to project.  Normally months
    
    '''
    pad = 25
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    out = pd.DataFrame()

    policy = flatten(policy, _debug=_debug)
    if _debug: print('flattened keys'.ljust(pad), policy.keys())

    # need to work out the max overhang.  NB this is also done in make_shapes
    max_hang = min([policy[x].launch_delay for x in policy])
    if _debug: print("max hang is".ljust(pad), max_hang)


    # get the shapes - keys will be same as col heads
    # NB MUST PASS NET SPEND FLAG.  The policy carries the sav rates
    shapes = make_shapes(policy, term_dur=1, start_m=start_m, net_spend=True,
                        flat=True, multi_index=False, synch_start=synch_start, _debug=_debug)
    if _debug: print('shapes returned:\n', shapes.head())

    # work out new time frame, given any extension for negative launch delay
    # len_reqd = n_pers - max_hang
    # if _debug: print('new timeframe'.ljust(pad), len_reqd)

    # can now iterate through shapes, and cross ref the spendline
    for s in policy:
        if _debug: print('column: '.ljust(pad), s)
        out[s] = get_forecast(shapes[s], term_gr = policy[s].term_gr,
                                             coh_gr = policy[s].coh_gr,  
                                             l_stop = policy[s].launch_stop,
                                             n_pers = n_pers, 
                                             _debug=_debug)

    if _debug: print('length returned'.ljust(pad), len(out))

    # the index may be longer than the n pers, because of any overhangs erc
    ind = pd.PeriodIndex(start=start_m, freq='M', periods=len(out.index))
    out.index = ind

    if multi_index: out.columns=pd.MultiIndex.from_tuples(out.columns)

    if diffs_out:
        out = out.iloc[:,1:].subtract(out.iloc[:,0], axis=0)

    if annual:
        out = out.groupby(out.index.year).sum()

    if nans_to_zero: out[out.isnull()] = 0

    if _debug: print("\LEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    return out


