import pandas as pd
import numpy as np
import inspect

from lcmod.classes import *

def trend(prod, interval=24, launch_cat=None, life_cycle_per=0,
          shed=None, loe_delay=0, term_gr = 0,  threshold_rate=0.001, n_pers=12, 
          _out_type='array', start_m=None, _debug=False, name=None):

    '''Takes input array, with parameters, and returns a trend-based projection.

    Key parameters:

        prod            An input array of spend

        interval        The number of periods (back from last observation) that are used
                        to calculate the trend

        life_cycle_per  Where the current period (i.e. last obs) lies in the lifecycle 

        loe_delay       The periods by which actual fall in revenue is delayed beyond
                        patent expiry

        _out_type       Pass 'df' to return a df with raw past, mov ave past and projection.  
                        (also pass a start_m to add a PeriodIndex to the df)


    Notes on use of loe_delay
    -------------------------

    The loe_delay is used to extend plat_dur.  This has two effects:  
        
        1. It may change the classification of the product's lifecycle phase.
             
             For example, if the life_cycle_per is 100 and uptake_dur + raw plat_dur
             is 98, then the product would be classified to terminal phase.

             But if an loe_delay of 6 was added, uptake_dur + plat_dur is 104.
             That puts the product in plateau phase.

             This is desirable, IF the loe_delay is correct - as, in the above eg,
             at period 100 the product would not yet have had a drop in spend.

             However it does put a lot of faith in the loe_delay when loe is close to
             to the current period.  In particular, if it is too low and the product 
             is classified to terminal phase when it has not yet had a spend reduction, 
             it will lead to a large over-estimate of spend (assuming the drop would be large).

             There is a smaller problem in the other direction - if loe_delay is too high,
             the product will be classified to plateau even though it has already had a spend 
             reduction.  It will then get another (but, presumably, this error will affect a
             smaller starting spend level).

             Potential solutions could include:
                - identifying whether the product actually has had a drop
                - allowing manual assignment to phase (eg as a var in the df)
                - using gradual erosion (maybe)

        2. It extends the plateau duration projection (if there is one)

            This is more obvious, and less problematic.  Once the product is assigned to 
            plateau phase, that will be extended by the amount of loe_delay

    '''


    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n")

    pad, lpad, rpad = 45, 35, 20

    if _debug: print('\nPROCESSING ARRAY INPUT')

    if isinstance(prod, pd.Series):
        prod = np.array(prod)
        if _debug: print('found a series, len'.ljust(pad), str(len(prod)).rjust(rpad))

    elif isinstance(prod, tuple):
        prod = np.array(prod)
        if _debug: print('found a tuple, len'.ljust(pad), str(len(prod)).rjust(rpad))

    elif isinstance(prod, np.ndarray):
        prod = np.array(prod)
        if _debug: print('found an ndarray, len'.ljust(pad), str(len(prod)).rjust(rpad))
        if len(prod) == 1:
            if _debug: print('unpacking array of unit length (i.e. actual array nested in a list with len==1)')
            prod = prod[0]
            if _debug: print('array len now'.ljust(pad), str(len(prod)).rjust(rpad))

    else:
        print("DON'T KNOW WHAT HAS BEEN PASSED - make sure its not a dataframe")
        return

    #---
    if _debug: print('\nPROCESSING LIFECYCLE INPUTS')

    uptake_dur = shed.uptake_dur
    # NB  this is the critical use of loe_delay
    plat_dur = shed.plat_dur + loe_delay
    gen_mult = shed.gen_mult
    plat_gr = shed.plat_gr

    if _debug: 
        print(" - uptake_dur".ljust(pad), str(uptake_dur).rjust(rpad))
        print(" - plat_dur".ljust(pad), str(plat_dur).rjust(rpad))
        print(" - plat_gr".ljust(pad), str(plat_gr).rjust(rpad))
        print("(after adding loe_delay of)".ljust(pad), str(loe_delay).rjust(rpad))
        print(" - gen_mult".ljust(pad), str(gen_mult).rjust(rpad))
        print("lifecycle period".ljust(pad), str(life_cycle_per).rjust(rpad))

    # make an annual moving average array
    prod[np.isnan(prod)]=0
    prod_ma = mov_ave(prod, 12)

    #---
    if _debug: print('\nANALYSING PAST SPEND')

    max_spend        = np.nanmax(prod)
    max_spend_per    = np.nanargmax(prod)
    last_spend       = (prod[-1])

    max_spend_ma     = np.nanmax(prod_ma)
    max_spend_ma_per = np.nanargmax(prod_ma)
    last_spend_ma    = (prod_ma[-1])

    total_drop_ma    = max(0, max_spend_ma - last_spend_ma)
    
    if not max_spend_ma == 0:
        total_drop_ma_pct = total_drop_ma/max_spend_ma  

    # get linear change per period over interval of periods
    # TODO calculate this on recent averages
    interval = min(interval, len(prod))
    interval_delta = prod[-1] - prod[-(1 + interval)]
    interval_rate = interval_delta / interval
    interval_rate_pct = None

    if not prod[-(1 + interval)] == 0:
        interval_rate_pct = interval_rate / prod[-(1 + interval)]
 
    if _debug:
        print("max spend in a single period".ljust(pad), "{:0,.0f}".format(max_spend).rjust(20))
        print("period of max spend".ljust(pad), "{}".format(max_spend_per).rjust(rpad))
        print("spend in last period".ljust(pad), "{:0,.0f}".format(last_spend).rjust(rpad))
        print("max of mov ave spend".ljust(pad), "{:0,.0f}".format(max_spend_ma).rjust(rpad))
        print("period of max mov ave spend".ljust(pad), "{}".format(max_spend_ma_per).rjust(rpad))
        print("last obs mov ave spend".ljust(pad), "{:0,.0f}".format(last_spend_ma).rjust(rpad))
        print("drop in mov ave".ljust(pad), "{:0,.0f}".format(total_drop_ma).rjust(rpad))
        print("drop in mov ave pct".ljust(pad), "{:0,.0f}%".format(total_drop_ma_pct*100).rjust(rpad))
        print("interval for calculating linear trend".ljust(pad), "{}".format(interval).rjust(rpad))
        print("change over that interval".ljust(pad), "{:0,.0f}".format(interval_delta).rjust(rpad))
        print("change per period over interval".ljust(pad), "{:0,.0f}".format(interval_rate).rjust(rpad))
        print("change per period over interval pct".ljust(pad), "{:0,.0f}%".format(interval_rate_pct*100).rjust(rpad))


    #--- work out phase
    if _debug: print('\nCLASSIFYING TO PHASE')

    if life_cycle_per <= uptake_dur: 
        phase = 'uptake'

    # note that plat_dur has been extended by the loe delay
    elif life_cycle_per <= uptake_dur + plat_dur:
        phase = 'plateau'

    else: phase = 'terminal'

    if _debug: print('Classified as'.ljust(pad), phase.rjust(rpad))


    #---  make array for building output
    if _debug: print('\nCONSTRUCTING PROJECTION ARRAY')

    out = np.array([last_spend_ma]) # this period overlaps with past, will be snipped later
    if _debug: print('initial stub of proj. array'.ljust(pad), out)
    
    if phase == 'terminal':
        # this is really a shortcut where we know it's in terminal
        if _debug: print('\nIn terminal phase, so creating a terminal array')
        out = out[-1] * ((1 + term_gr) ** np.arange(1, n_pers+1))
        if _debug: 
            print('First 10 periods of terminal array:')
            print(out[:10], end="\n")

    else:

        # This is the main work.  For each phase make an array, and append to the out array
        if _debug: print('\nGenerating pre-terminal phases')

        # compute remaining UPTAKE periods and generate an array
        uptake_pers = min(max(uptake_dur - life_cycle_per, 0),n_pers - (len(out)-1))
        uptake_out = out[-1] + (interval_rate * np.arange(1,uptake_pers))
        
        # move the lifecycle period along to the end of uptake phase
        life_cycle_per += uptake_pers

        if _debug:
            print("\nRemaining UPTAKE periods".ljust(pad), str(uptake_pers).rjust(rpad))
            print("--> lifecycle period moved to".ljust(pad), str(life_cycle_per).rjust(rpad))

        # append the uptake array to the out array
        out = np.append(out, uptake_out)

        # compute remaining PLATEAU periods, and generate an array 
        # Note that plat_dur has been extended by loe_delay
        plat_pers = min(max((uptake_dur + plat_dur) - life_cycle_per, 0), n_pers - (len(out)-1))
        plat_out = out[-1] * ((1 + plat_gr) ** np.arange(plat_pers))
        life_cycle_per += plat_pers

        if _debug:
            print("\nRemaining PLATEAU periods".ljust(pad), str(plat_pers).rjust(rpad))
            plat_ch = 100*(plat_out[-1]-plat_out[0])/plat_out[0]
            print("Total growth over plateau".ljust(pad), "{:0,.1f}%".format(plat_ch).rjust(rpad))

            print("--> lifecycle period moved to".ljust(pad), str(life_cycle_per).rjust(rpad))

        # append the plateau array to the out array
        out = np.append(out, plat_out)

        # compute remaining TERMINAL periods and generate an array
        term_pers = max(n_pers - (len(out)-1), 0)
        term_out = out[-1] * gen_mult * ((1 + term_gr) ** np.arange(1, term_pers+1))

        if _debug:
            print("\nRemaining TERMINAL periods".ljust(pad), str(term_pers).rjust(rpad))

        # append the terminal array to the out array
        out = np.append(out, term_out)

        # eliminate any negatives
        out[out<0] = 0



    #---output
    if _out_type == 'df':
        if _debug: print('\nGenerating df output')
        spacer = np.empty(len(prod))
        spacer[:] = np.nan
        out=np.insert(out, 0, spacer)
        df=pd.DataFrame([prod, prod_ma, out], index=['raw', 'mov_ave', 'projected']).T

        # add an index if a start month was passed
        if start_m is not None:
            df.index = pd.PeriodIndex(start=pd.Period(start_m, freq='M'), periods=len(df))

        # get rid of the ugly trajectory of mov_ave from zero
        df['mov_ave'][:interval] = np.nan
        if _debug: print("\nLEAVING:  ", inspect.stack()[0][3])
        return df

    else:
        if _debug: print("\nLEAVING:  ", inspect.stack()[0][3])
        return out[1:]



##_________________________________________________________________________##


def r_trend(df, n_pers, *, shed=None, uptake_dur=None, plat_dur=None, gen_mult=None, term_gr=0, 
            loe_delay=None, threshold_rate=0.001, _interval=24, _out_type='array', _debug=False):
    
    '''Iterates through an input df, calling trend(), returning a df of projections.

    Key logic is calculation of lifecycle period, which is passed to trend() to orient the projection.
    This is currently done with reference to the loe date.  

    Eg if last observation is 1-2017, and loe date for a product is 1-2020, then the lifecycle period is 
    36 periods before the loe lifecycle period (which is uptake_dur + plat_dur).

    So if uptake_dur=56, plat_dur=100, lifecycle period is 120 (56+120-36).  When passing to trend(), another
    36 periods of plateau will be projected, and then the loe drop will be applied.

    To reflect a lag in erosion, therefore need to position product further back in lifecyle.  
    In above example, if lag was 6m, pass the lifecycle period of 114, so that 42 periods of plateau are applied.

    To do this, pass an loe_delay parameter that in turn goes to trend() and extends plat_dur.  
    Could include this loe_delay parameter in the lifecycle model.  

    _out_type='array' specifies that trend() returns an array, obv, which is required for the actual projections.
    But can pass 'df' to get the dataframe output (showing mov ave etc) if calling to visualise projections etc.  
    In this case, r_trend() will return a list of those dfs.
 
    '''
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad = 35
    out=[]


    #  housekeeping - assign lifecycle variables depending on what was passed
    if shed is not None:
        if _debug: print('using passed shed:')
        uptake_dur = shed.uptake_dur
        plat_dur = shed.plat_dur
        gen_mult = shed.gen_mult

    # define the lifecycle period in which loe will occur, according to input shed or shape data
    # - this is used to fix the actual lcycle period
    loe_lcycle_per = uptake_dur + plat_dur

    # enter the key loop through rows in the input df
    for row in df.itertuples():

        # TODO make params a dict, rather than have to look up by index number
        params = row[0]
        data = row[1:]

        if _debug: print('\nMolecule'.ljust(pad), params[df.index.names.index('molecule')])
        if _debug: print('Setting'.ljust(pad), params[df.index.names.index('setting')])

        # get loe month
        loe_month = pd.Period(params[df.index.names.index('loe_date')], freq='M'); 
        if _debug: print('taking raw loe date'.ljust(pad), loe_month)

        # get date of last month in actual data
        last_month = df.columns[-1]
        if _debug: print('last_month'.ljust(pad), last_month)

        # get time after / before loe (negative if befote loe)
        pers_post_loe = last_month - loe_month
        if _debug: print('pers_post_loe'.ljust(pad), pers_post_loe)

        # infer the lifecycle period from this
        life_cycle_per = loe_lcycle_per + pers_post_loe
        if _debug: print('life_cycle_per'.ljust(pad), life_cycle_per)

        # call trend
        out_array = trend(data, _interval, n_pers=n_pers, life_cycle_per=life_cycle_per, shed=shed,
                        loe_delay=loe_delay, name=params[0], term_gr=term_gr,  
                        _out_type=_out_type)

        out.append(out_array)

    # just return this out list of dataframes if passing 'df' as _out_type (eg for visualisations)
    if _out_type == 'df':
        if _debug: print("\LEAVING:  ", inspect.stack()[0][3])
        return out

    # Otherwise build the df index and columns and return a single df

    cols = pd.PeriodIndex(start=last_month+1, periods=n_pers, freq='M')

    if _debug: print("\LEAVING:  ", inspect.stack()[0][3])

    return pd.DataFrame(out, index=df.index, columns=cols)



##_________________________________________________________________________##


def r_fut_tr(df, n_pers, cut_off, shed=None, loe_delay=None,
             coh_gr=0, term_gr=0, name='future', _debug=False):
    
    '''Generates a projection of spend on future launches, based on cumulation
    of a lifecycle profile (itself imputed from observations), and scaled using observations.

    Note only returns the future projection, not the input past observations

    1. Make a shape corresponding to the passed shed
    2. Use this to project a forecast (unscaled)
    3. Scale the forecast to the last period of actual observations
    4. Slice the forecast to give the future only

    '''

    pad = 25

    # INCREMENT PLAT_DUR BY LOE DELAY before passing to make_shape()
    # The non future trend functions use this differently, as they need to calculate the loe_month for
    # extending observations etc, based on observed loe date.  Think it's ok this way but need to be aware.

    # for future want to probably include this as part of shed.  Though there's an argument it's
    # really part of the plat_dur (in effect), and the need to fiddle around is only when you are working out 
    # the plat_dur from an external loe date.
    
    if loe_delay is not None:
        if _debug: print('remaking shed to add loe delay\n')
        shed = Shed(shed.shed_name + '_1', 
                       shed.uptake_dur,
                       shed.plat_dur + loe_delay,
                       shed.gen_mult)
        if _debug: 
            print("shed now")
            print(shed)
   

    # will be working with the sum of the input df
    df=df.sum()

    # get useful dates
    cut_off = pd.Period(cut_off)
    last_date = df.index[-1]


    #--- 1. Make a shape from the passed shed
    shape = make_shape(shed=shed)

    #--- 2. Use this to project a forecast.
    # - need to project for n_pers plus the overlap with actual
    overlap = last_date - cut_off
    if _debug: 
        print('cut off:'.ljust(pad), cut_off)
        print('last_date:'.ljust(pad), last_date)
        print('overlapping periods:'.ljust(pad), overlap)
        print('shape length'.ljust(pad), len(shape))
        print('shed'.ljust(pad), shed)

    fut = get_forecast(shape, term_gr=term_gr, coh_gr=coh_gr, n_pers=n_pers+overlap, name=name)

    #--- 3. Scale the forecast
    #  Take cumulative sum for last period in slice passed
    last_sum = df[-1]
    if _debug: print('spend at last period'.ljust(pad), last_sum)

    # to scale, want the period just before the fut forecast to equal last_sum.  
    if _debug: print('spend at overlap period'.ljust(pad), fut.iloc[overlap])
    scaler=last_sum/fut.iloc[overlap-1]
    if _debug: print('scaler to apply'.ljust(pad), scaler)
    
    fut = (fut*scaler)
    if _debug: print("\ntail of actual:\n", df.tail(), "\n")
    if _debug: print("\nscaled fut at overlap:\n", fut[overlap-5:overlap+5].head(), "\n")

    #--- 4. Slice the forecast to give the future only
    out = fut[overlap:]

    out.index=pd.PeriodIndex(start=last_date+1, periods=len(out), freq='M')

    return pd.DataFrame(out)



##_________________________________________________________________________##


def make_shape(spendline=None, shed=None, z_pad=0, peak_spend=1, annualised=False, sav_rate=0, 
                net_spend=False, term_pad=1, term_gr=0, ser_out=True, name=None, _debug=False):
    '''Flexible function for generating shapes from sheds or spendlines.

    Everything but cohort growth - this is nonaccumulated
    Optionally add pads before and after (z_pad, term_pad) 
    Optionally return a pandas series or numpy array
    '''
    
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad = 30

    # set a multiplier if the shape is to be annualised
    ann_factor = 1
    if annualised:
        ann_factor = 144
        if_debug: print('annualising')
    elif _debug: print('not annualising')

    # process the lifecycle inputs
    if spendline is not None and shed is not None:
        print('PASSED A SHED AND A SPENDLINE TO ', inspect.stack()[0][3])
        print('using the shed')
        spendline = None

    # note the launch delay is translated into z_pad, the amount of zero padding the array gets
    if spendline is not None:
        if _debug: print('using a passed spendline.   Note will not synchronise for negative launch delays\n')
        shed = spendline.shed
        peak_spend = spendline.peak_spend
        sav_rate = spendline.sav_rate
        term_gr = spendline.term_gr

        if spendline.launch_delay < 0: 
            print("Can't use negative launch delays - setting this to zero")
            print("Pass a shed with z_pad appropriate to the set of spendlines if doing this for anything other than display\n")
            z_pad = 0
        else: z_pad = spendline.launch_delay

    elif shed is not None:
        if _debug: print('using a passed shed.\n')

    else: print('need a spendline or a shed'); return 1

    # use the final values to build components of the shape
    zeros = np.zeros(z_pad)
    uptake = np.arange(1, shed.uptake_dur+1)
    plat = np.ones(shed.plat_dur) * shed.uptake_dur
    term = shed.gen_mult * (shed.uptake_dur * (1+term_gr) ** np.arange(1, term_pad+1))

    # now put components together
    base = np.concatenate([zeros, uptake, plat, term])

    if _debug:
        print('shed: '.ljust(pad), shed)
        print('zpad passed: '.ljust(pad), z_pad)
        print('zeros arr: '.ljust(pad), zeros)
        print('uptake arr: '.ljust(pad), uptake)
        print('plat arr: '.ljust(pad), plat)
        print('term arr: '.ljust(pad), term)
        print('net spend: '.ljust(pad), net_spend)
        print('sav_rate: '.ljust(pad), sav_rate)
        print('\n-->base arr: '.ljust(pad))
        print(base, "\n")

    # on to processing of the shape - first, need to net off savings?
    if not net_spend: 
        sav_rate=0
        if _debug: print('not netting, so rate is'.ljust(pad), sav_rate)

    else: 
        if _debug: print('netting, sav_rate is'.ljust(pad), sav_rate)
    
    # do all the scaling.  Remembert uptake_dur is peak, due to the way
    # base assembled with np.arange for the uptake period
    scaling_factor = peak_spend * ann_factor * (1-sav_rate) / shed.uptake_dur

    base *= scaling_factor

    if _debug: 
        print('\n1 - effective sav_rate'.ljust(pad), 1 - sav_rate)
        print('\npeak_spend'.ljust(pad), peak_spend)
        print('\nann_factor'.ljust(pad), ann_factor)
        print('\nuptake_dur divisor'.ljust(pad), shed.uptake_dur)
        print('\n-->scaling factor'.ljust(pad), scaling_factor)
        print('\nbase after scaling'.ljust(pad), "\n", base)
  
    if ser_out:
        base = pd.Series(base, name=name)   

    if _debug: print("\nLEAVING FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..returning to:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    return base



##_________________________________________________________________________##


def get_forecast(shape, term_gr, coh_gr, n_pers=120, 
                  l_stop=None, name=None, _debug=False, _logfile=None):

    '''For a given input shape, plus cohort parameters, generate an accumulation of
    spend over a period of successive launches with that shape.

    Option to limit the duration of launches by passing l_stop. This specifies
    the number of periods over which new launches occur - while still accumulating them 
    over subsequent periods (until the end of the projection, at n_pers)

    Pass a file path to _logfile and get disaggregated output, with each launch cohort
    shown separately (and then summed).
    '''
    # General strategy is to make a single array of the shape (plus terminal period)
    # and then overlay copies, shifting forward one period each time (and applying any 
    # cohort growth rate

    # First make the starting array - the shape, extended for the number of periods
    # note this is snipped off at n_pers, which may mean never reach terminal
    if _debug: print("\nIN FUNCTION:  ".ljust(20), inspect.stack()[0][3])
    if _debug: print("..called by:  ".ljust(20), inspect.stack()[1][3], end="\n\n")

    pad = 25

    if _debug: 
        if isinstance(shape, np.ndarray):
            print("processing np array shape, len".ljust(pad), len(shape))

        elif isinstance(shape, pd.Series):
            print("processing pd series shape, name".ljust(pad), shape.name)
            print("..........................length".ljust(pad), len(shape))

        else: print('processing something but not sure what')

    # get last spend value - used to scale the terminal period
    last_val = None

    try:
        last_val = shape[-1]
        if _debug: print('got last period using index -1')

    except:
        if _debug: print('could not get last period using index -1')

        if isinstance(shape, pd.Series):
            last_val = shape.iloc[-1]
            if _debug: print('Last val assigned'.ljust(pad), last_val)
        
        else:
            print("can't get a last period value")

    # make the terminal stretch that will extend the input shape
    term_per = (1 + term_gr) ** np.arange(1, n_pers - len(shape) + 1) * last_val

    if _debug: 
        print('\nInput shape head\n', shape[:10])
        print('Term_per ', term_per)

    # assemble the array, adding the terminal extension
    base_arr = np.concatenate([shape, term_per])[:n_pers]

    if _debug: print('Base_arr head\n', base_arr[:10])


    # instantiate a new copy of the array to build on, adding layers 
    # (don't want to mutate the starting array)
    res = base_arr.copy()
    
    # use a factor to reflect cohort growth
    growth_factor = 1
    
    if _debug: 
        df_out = pd.DataFrame(base_arr, columns=[0])

    # determine whether the launches are going to stop within proj period 
    if l_stop is not None:
        l_pers = min(l_stop, n_pers)
        if _debug: print('curtailing launches at', l_pers)
    else:
        l_pers = n_pers
        if _debug: print('launches for full interval')

    # MAIN JOB: iterate through remaining periods (already have zero)
    for per in range(1, l_pers):

        # first calculate the new growth factor
        growth_factor *= (1 + coh_gr)
        
        # make a layer, shifting base to right and adding zeroes at start
        layer = np.concatenate([np.zeros(per), base_arr[:-per]]) * growth_factor
        if _debug: df_out[per] = layer

        # add the layer to the base
        res += layer

    if _debug: 
        df_out['dfsum'] = df_out.sum(axis=1)
        df_out['result'] = res
        df_out['diff'] = df_out['dfsum']  - df_out['result'] 
        #print('debug df info: ', df_out.info())
        if _logfile is not None:
            df_out.to_pickle(_logfile + '.pkl')
    
    return pd.Series(res, name=name)


###_________________________________________________________________________###


def mov_ave(in_arr, window):
    '''Parameters:  
        
            in_arr: an input array (numpy, or anything that can be coerced by np.array())
            window: the window over which to make the moving average

        Return:

            array of same length as in_arr, with mov ave
    '''
    
    # first coerce to numpy array 
    in_arr = np.array(in_arr)    

    # now turn nans to zero
    in_arr[np.isnan(in_arr)] = 0

    a = np.cumsum(in_arr) # total cumulative sum
    b = (np.cumsum(in_arr)[:-window]) # shifted forward, overlap truncated
    c = np.insert(b, 0, np.zeros(window))  # start filled to get to line up

    return (a-c) / window

    
 ###_________________________________________________________________________###

