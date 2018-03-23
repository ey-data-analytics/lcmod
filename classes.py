# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from collections import namedtuple
import inspect

from lcmod.core import *


class RuleSet:

    def __init__(self, parent_df, name='generic', string_slice=None, index_slice=None,
                func_str="", func=None, f_args={}, join_output=True, log=None):
            
        '''Creates an instance of a RuleSet - which allows the application of a 
        projection rule to a defined set of products.


        INPUT PARAMETERS

        parent_df       : the DataFrame containing spend on existing products.
                        : [the only argument required on instantiation]

        string_slice, index_slice
                        : define the subset of the parental DataFrame
                        : string_slice is in a format compatible with the web app, and
                        is translated to index_slice, which is actually used to slice the df,
                        by the slicify_string method
                        : index_slice is a dict NOT AN ACTUAL INDEXSLICE!
                        : TODO support passing an actual indexslice

        func_str, func  : define the projection function to be applied
                        : func_str is just the string for the function,
                        : func is the object itself

        join_output     : this is used as a hack to get around the problem that projections for future launches
                        are based on a series of past spend (i.e. one line), which creates problems when sticking 
                        together with future spend.  NEEDS FIXING, although it currently does work.



        KEY METHODS

        set_slice()      : sets index_slice, using a dictionary of parameter-argument pairs,
                        where parameters correspond with index levels / labels in the parent df, 
                        and arguments are the slicing rules

                        eg dict(is_BIOL=True) will create a slice of all biologicals (assuming they're identified)


        set_args()      : sets the arguments passed to the projection function

        xtrap()         : executes the projection calculation, and creates outputs



        OUTPUTS

        A set of pandas DataFrames or Series:

            past        : df with past spend on products in the slice

            fut         : df with future, projected spend on products in the slice

            joined      : df with past and future

            summed      : series with the sum of spend across all products

        '''

        self.pad = 25

        self.name = name
        self.parent_df = parent_df
        
        # initialise the index slice to the names in the parent index
        if index_slice is None:
            self.index_slice = {i:None for i in parent_df.index.names}
        else:
            self.index_slice = index_slice

        # string slice holds slicing info in format used by input form
        self.string_slice = string_slice

        self.func = func
        self.func_str = func_str
        self.f_args = f_args
        self.past = None
        self.fut = None
        self.join_output=join_output
        self.joined = None
        self.summed = None
        self.out_fig = ""

        if log is not None:
            self.log = log


    def set_slice(self, input_slice):
        '''Set the index_slice dictionary.

        Takes a dictionary input, and sets matching elements in self.index_slice
        to equal the values in the input_slice'''

        for k in input_slice:
            if k in self.index_slice:
                self.index_slice[k] = input_slice[k]
            else:
                print('{', k, ':', input_slice[k], '} not in index')


    def slicify_string(self):
        for i in self.index_slice:
             self.index_slice[i] = slicify(self.string_slice[i])


    def set_args(self, a_args):
        '''Set the f_args dictionary - passed to self.func()'''
        if self.func==None:
            print("no function yet")
            return

        wrong_keys = [k for k in a_args.keys() 
                        if k not in inspect.getfullargspec(self.func)[4]]

        if wrong_keys:
            print("Args aren't parameters of ", self.func.__name__,": ", wrong_keys)

        else: self.f_args = a_args



    def get_params(self):
        #todo: parse out in best way
        if self.func==None:
            print("no function yet")


        else:
            return inspect.getfullargspec(self.func)[4]

    
    def xtrap(self, n_pers, _debug=False):

        slice= get_ix_slice(self.parent_df, self.index_slice)

        if _debug: print('got slice\n', slice)

        self.past = self.parent_df.loc[slice,:]

        if _debug: print('got')

        if self.func == None:
            print("no function assigned")
            return

        
        self.fut = self.func(self.past, n_pers, **self.f_args)

        if self.join_output:
            try:
                self.joined = self.past.join(self.fut)
                self.summed = pd.DataFrame(self.joined.sum(), columns=[self.name])
            except:
                self.joined = self.summed = pd.DataFrame(self.past.sum().append(self.fut).sum(axis=1), 
                                                columns=[self.name])

         
        
    def __repr__(self):
        outlist = []
        self._info = {"name": self.name,
             "string_slice": self.string_slice,
             "index_slice": self.index_slice,
             "func": self.func,
             "f_args keys": list(self.f_args.keys()),
             "f_args": self.f_args,
             "past type": type(self.past),
             "fut type": type(self.fut),
             "joined type": type(self.joined),
             "summed type": type(self.summed),
             "join_output": self.join_output}
        for key in self._info:
            temp_string = key.ljust(27) + str(self._info[key]).rjust(10)
            outlist.append(temp_string)
        return "\n".join(outlist)
    
##_________________________________________________________________________##

def get_ix_slice(df, in_dict):
    '''make a pd.IndexSlice
    args:   - a dataframe (with named index)
            - dict of index names:value pairs to be sliced (in any order)
            
    returns a pd.IndexSlice with the desired spec
            
    eg, if index contains the boolean 'is_cool' and ints 'year'
       'is_cool = True' will generate a slice where 'is_cool'  is set to True
       'year = slice(2006, 2009, None)' will select years 2006 to 2009 
       'year = slice(2006, None, None)' will select years 2006 onward 
       'year = [2008, 2012]' will select just those two years
       
    simply print the returned output to see what's going on

    Can pass output directly to iloc.  Eg

        ixs = pf.get_ix_slice(df_main, dict(is_biol=False, launch_year=[2015,2016]))
        df_main.loc[ixs,:]

    '''
    # first turn any None entries of the input into slices
    for i in in_dict:
        if in_dict[i] is '' or in_dict[i] is None:
            in_dict[i] = slice(None, None, None)
            

    return tuple((in_dict.get(name, slice(None,None,None)))
                     for name in df.index.names)


##_________________________________________________________________________##

class Shed():
    '''The parameters to define a linear 'shed'-like lifecycle profile, 
    in a named tuple.
    
    .shed_name  : the name of the shed shape
    .uptake_dur : the number of periods of linear growth following launch
    .plat_dur   : the number of periods of constant spend following uptake
    .gen_mult   : the change on patent expiry (multiplier)
    '''
    def __init__(self, name="", uptake_dur=None, plat_dur=None, gen_mult=None):

        self.uptake_dur = uptake_dur
        self.plat_dur = plat_dur
        self.gen_mult = gen_mult
        self.shed_name = name


    def __repr__(self):
        outlist = []
        self._info = {"name": self.shed_name,
             "uptake_dur": self.uptake_dur,
             "plat_dur": self.plat_dur,
             "gen_mult": self.gen_mult}
        for key in self._info:
            temp_string = key.ljust(27) + str(self._info[key]).rjust(10)
            outlist.append(temp_string)
        return "\n".join(outlist)



    namedtuple('shed', 'shed_name uptake_dur plat_dur gen_mult')

##_________________________________________________________________________##


class SpendLine():
    '''Stores information required to define a line of spending, eg a set of products
    in a particular scenario.

    CONSTRUCTOR ARGUMENTS

    name            : the name of the spendline [REQD]

    shed            : namedtuple describing the lifecycle profile [REQD]
     .shed_name      : the name of the shed profile
     .uptake_dur     : duration of (linear) uptake period
     .plat_dur       : duration of (flat) plateau
     .gen_mult       : impact of patent expiry (multiplier, eg 0.2)
    
    term_gr         : the rate of growth to be applied (indefinitely) 
                    :  after the drop on patent expiry

    launch_delay    : negative if launch brought fwd
    coh_gr          : cohort growth rate
    peak_spend      : peak spend to which profile is normalised
    icer            : the incremental cost-effectiveness ratio (£cost/QALY)
    sav_rate        : proportion of spend that substitute other drug spend

    log             : general dict for holding whatever

    All time units are general - usually implied to be months.  
    But no annualisation within this class except in the string.

    '''
    def __init__(self, name, shed, loe_delay=0,
                       term_gr=0, launch_delay=0, launch_stop=None, coh_gr=0, 
                       peak_spend=1, icer=None, sav_rate=0,
                       log=None):

        self.name = name
        self.shed = shed
        self.loe_delay = loe_delay
        self.term_gr = term_gr
        self.coh_gr = coh_gr
        self.peak_spend = peak_spend
        self.launch_delay = launch_delay
        self.launch_stop = launch_stop
        self.icer = icer
        self.sav_rate = sav_rate

        # create a general purpose dict for holding whatever
        if log==None: 
            self.log = {}
        else: self.log = log

    
    def __str__(self):
        pad1 = 35
        pad2 = 10

        if self.icer is None:
            icer_str = str(None)

        else:
            icer_str = "{:0,.0f}".format(self.icer)

        return "\n".join([
            "SpendLine name:".ljust(pad1) + str(self.name).rjust(pad2),
            "peak spend pm of monthly cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_spend).rjust(pad2),
            "peak spend pa of annual cohort:".ljust(pad1) + "{:0,.2f}".format(self.peak_spend*12*12).rjust(pad2),
            "ICER, £/QALY:".ljust(pad1) + icer_str.rjust(pad2),
            "savings rate:".ljust(pad1) + "{:0,.1f}%".format(self.sav_rate*100).rjust(pad2),
            "shed:".ljust(pad1),
            "  - shed name:".ljust(pad1) + self.shed.shed_name.rjust(pad2),
            "  - uptake_dur:".ljust(pad1) + str(self.shed.uptake_dur).rjust(pad2),
            "  - plat_dur:".ljust(pad1) + str(self.shed.plat_dur).rjust(pad2),
            "  - gen_mult:".ljust(pad1) + str(self.shed.gen_mult).rjust(pad2),
            "loe_delay:".ljust(pad1) + str(self.loe_delay).rjust(pad2),
            "term_gr:".ljust(pad1) + str(self.term_gr).rjust(pad2),
            "launch_delay:".ljust(pad1) + str(self.launch_delay).rjust(pad2),
            "launch_stop:".ljust(pad1) + str(self.launch_stop).rjust(pad2),
            "cohort gr, pm:".ljust(pad1) + "{:0,.2f}%".format(self.coh_gr*100).rjust(pad2),
            "cohort gr, pa:".ljust(pad1) +  "{:0,.1f}%".format(self.coh_gr*100*12).rjust(pad2),
            "log keys:".ljust(pad1) + ", ".join(self.log.keys()).rjust(pad2),
            "\n\n"

        ])
 

    def __repr__(self):
        return self.__str__()

    def get_shape(self):
        return make_shape(peak_spend=self.peak_spend, shed=self.shed)

