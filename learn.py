import numpy as np
import pandas as pd

from copy import deepcopy
from pprint import pprint
from collections import OrderedDict
import inspect

from lcmod.classes import Shed
from lcmod.core import make_shape, get_forecast


class Path:
    '''Container for learning optimisation searches. For a given set of variable 
    parameters, will seek a path to best fit with a set of observations.

    CONSTRUCTOR ARGUMENTS:

    panel           : a dict with a shed, term_gr and coh_gr elements
                      (i.e. enough to make a forecast shape)

    params_in       : a dict with the parameters to be varied and control info.  
                      Structure of this dict is:

                      { parameter1 : (init_value, incr_method, incr),
                      { parameter2 : (init_value, incr_method, incr)..}

                      where the value is a tuple or list of:

                        init_value  - the initial value of the variable
                                      (will overwrite value in panel)

                        incr_method - the method for incrementing
                                      from {'add_int', 'multiply'}

                        incr        - the increment to apply
    
    obs             : a np.array of observations to which the learning process
                      will attempt to generate a best fit

    cont_pat        : boolean flag instructing the algorithm to maintain constant
                      patent length.  True by default.

                      eg if uptake_dur is reduced by 1, plat_dur would be increased
                      by 1, if this flag is set


    ATTRIBUTES

    params          : a dict of control variables, with one entry for each parameter
                      being varied

    hist            : a pd.DataFrame with the historical path for the algorithm
                      (also serves as the main data structure driving the algorithm)


    METHODS

    extend()        : attempt a round of optimisation (which will attempt to improve 
                      fit by changing EACH variable parameter, both up and down)

    is_active()     : returns True if ALL variable parameters are reported as inactive
                      (that is, they were not improved in last extension)
    
    '''

    def __init__(self, panel=None, params_in=None, obs=None, const_pat=True, _debug=False):
        self.pad = 30

        # PARAMS structure for holding control data about parameters 
        self.params = dict()
        for p in params_in:
            self.params[p] = dict()
            self.params[p]['key'] = p
            self.params[p]['init_val'] = params_in[p][0]
            self.params[p]['incr_method'] = params_in[p][1]
            self.params[p]['incr'] = params_in[p][2]
            self.params[p]['ascending'] = True
            self.params[p]['active'] = True

        # store the observation data, against which test scenarios are compared
        self.obs = obs 
        
        # flag for whether to keep patent length constant
        self.const_pat = const_pat

        # set up the HISTORY dataframe - first making a dict from inputs
        data = OrderedDict(uptake_dur = panel['shed'].uptake_dur,
                           plat_dur   = panel['shed'].plat_dur,
                           plat_gr    = panel['shed'].plat_gr,
                           gen_mult   = panel['shed'].gen_mult,
                           term_gr    = panel['term_gr'],
                           coh_gr     = panel['coh_gr'],
                           diff       = 0,
                           action     = True)

        # - overwrite the inputs if different initial values provided
        for k in params_in: data[k]   = params_in[k][0]

        # - initialise history dataframe
        self.hist = pd.DataFrame(data=data, index=[0])

        # - write in the diff result
        diff = get_diff(self.hist.iloc[-1], obs=self.obs)
        self.hist.loc[0, 'diff'] = diff
        if _debug: print("\nInitial History\n", str(self.hist))

    # method for extending the search path
    def extend(self, _debug=False):
        pad = self.pad

        if not self.is_active():
            return False

        # go through the test parameters
        for p in self.params:

            # get last row of history dif
            last_row = self.hist.iloc[-1,:]
            row      = self.hist.index[-1]
            #print("\nLast row:\n" + str(last_row))

            if _debug: 
                print("\nTesting".ljust(pad), p)
                print(" - current val:".ljust(pad), last_row[p])
                print(" - last diff:".ljust(pad), "{:.4e}".format(last_row['diff']))

            # try with current direction
            #  - first get a new value TODO pass just last_row[p] and  params dict?
            new_val = incr_val(last_row[p], self.params[p]['incr'],
                               self.params[p]['incr_method'], self.params[p]['ascending'])

            if _debug: print(" - new value".ljust(pad), new_val)

            # - then try result, and append if works out
            res = get_diff(params=last_row, obs=self.obs, chg_params={p:new_val}, ret_ser=True)
            if _debug: print(" - result in curr direction:".ljust(pad), "{:.4e}".format(res['diff']))

            if res['diff'] < last_row['diff']:
                if _debug: print("Improvement in curr direction")

            # - else get value in different direction
            else:
                if _debug: print(' ------ trying other direction ------ ')
                # make new val, this time passing opposite direction
                new_val = incr_val(last_row[p], self.params[p]['incr'],
                                   self.params[p]['incr_method'], not(self.params[p]['ascending']))

                if _debug: print(" - new value".ljust(pad), new_val)

                res = get_diff(params=last_row, obs=self.obs, chg_params={p:new_val}, ret_ser=True)
                if _debug: print(" - result in new direction:".ljust(pad), "{:.4e}".format(res['diff']))

                
            # now check what's happened - is the new diff better?
            if res['diff'] < last_row['diff']:
                res['action'] = True
                self.hist.loc[row+1] = res
                if _debug: print("--> Improvement - adding to history")

                # need to work out if direction changed
                if ((self.params[p]['ascending'] and new_val < last_row[p]) 
                        | (not(self.params[p]['ascending']) and new_val > last_row[p])):
                    self.params[p]['ascending'] = not(self.params[p]['ascending'])

            # if res is not an improvement, append unchanged copy to keep record 
            else: 
                if _debug:
                    print("--> No improvement possible for {}, in row".format(p), )

                self.hist.loc[row+1] = last_row
                self.hist.loc[row+1, 'action'] = False
                self.params[p]['active'] = False

        return True


    def is_active(self, _debug=False):
        check_window = 2 * len(self.params)
        if _debug:
            print("checking last {}".format(check_window))
            print(self.hist['action'][-check_window:])
        return self.hist['action'][-check_window:].any()


    def get_params(self):
        return self.params


    def __repr__(self):
        return str(tuple(self.params[p]['init_val'] for p in self.params))


    def __str__(self):
        rpad = 30
        out_list = []
        out_list.append("PARAMS:")
        for p in self.params: # each key / parameter, eg uptake_dur
            out_list.append("\n" + p)
            for q in self.params[p]:
                out_list.append(" - " + q.ljust(rpad) + str(self.params[p][q]))

        return "\n".join(out_list)


    def get_forecast(self, comparison=False):
        '''Return a projection for the most recent parameter set, 
        scaled to agree with the observations set.

        Optionally return a comparison of the scaled projection and the observations
        '''
        params = self.hist.iloc[-1]

        shed = Shed('x', uptake_dur = params.uptake_dur,
                         plat_dur = params.plat_dur,
                         plat_gr = params.plat_gr,
                         gen_mult = params.gen_mult)

        term_gr = params.term_gr
        coh_gr = params.coh_gr

        # make a shape, get a forecast
        shape = make_shape(shed=shed)
        unscaled = get_forecast(shape, term_gr=term_gr, coh_gr=coh_gr, n_pers=len(self.obs))
        
        # scale it
        scale_factor = self.obs.iloc[-12:].mean() / unscaled.iat[-6]
        scaled = unscaled * scale_factor

        if comparison:
            return pd.DataFrame(data=[scaled, self.obs.values], index=['learned', 'obs']).T

        else:
            return scaled



# -----------------------------------------------------------------------------------------------------

def get_diff(params, obs, chg_params=None, ret_ser=False, const_pat=True, _debug=False):
    '''For an input params (pandas series) with labels corresponding to shed parameters 
   required to get a spend forecast shape (unscaled), return the sum of squares of differences
   between the shape of that forecast and the shape of a passed observation.

   Can also optionally overwrite any of the parameters passed in pd_in with those passed in params.
    '''
    
    pad = 30

    # if patent length to be held constant, get it now
    if const_pat: pat_len = params['uptake_dur'] + params['plat_dur']

    # make sure don't change the inputs outside
    params = deepcopy(params)

    # overwrite if reqd
    if chg_params is not None:
        for k in chg_params:
            params[k] = chg_params[k]

            # check if need another change to keep pat length constant
            if k == 'uptake_dur' and const_pat:
                params['plat_dur'] = pat_len - params['uptake_dur']
                if _debug: print("adjusting plat_dur to".ljust(pad), params['plat_dur'])

            if k == 'plat_dur' and const_pat:
                params['uptake_dur'] = pat_len - params['plat_dur']
                if _debug: print("adjusting uptake_dur".ljust(pad), params['uptake_dur'])

        if _debug: print("Changed parameters - ", end=" ")


    if _debug: print("params now\n" + str(params))

    shed = Shed('x', uptake_dur = params.uptake_dur,
                     plat_dur = params.plat_dur,
                     plat_gr = params.plat_gr,
                     gen_mult = params.gen_mult)

    term_gr = params.term_gr
    coh_gr = params.coh_gr

    # make a shape, get a forecast
    shape = make_shape(shed=shed)
    unscaled = get_forecast(shape, term_gr=term_gr, coh_gr=coh_gr, n_pers=len(obs))
    
    # scale it
    scale_factor = obs.iloc[-12:].mean() / unscaled.iat[-6]
    scaled = unscaled * scale_factor

    # return the sum of squares of difference vs the observation

    diff = sum((scaled - obs.values)**2)

    # insert in params and return as series if required
    if ret_ser:
        params['diff'] = diff
        return params

    # or just return the result
    else: return diff

# -----------------------------------------------------------------------------------------------------

def incr_val(curr_val, incr, incr_method, ascending):

    if ascending: mult =  1
    else:         mult = -1
    
    if incr_method == 'add_int':
        new_val = curr_val + (incr * mult)
        if new_val < 1: new_val = 1

    if incr_method == 'multiply':
        new_val = curr_val * (1 + (incr * mult))

    return new_val

