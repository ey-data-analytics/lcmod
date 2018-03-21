
# Probably try to eliminate these, or integrate with one of the other files
import numpy as np

###_________________________________________________________________________###

def variablise(string):
    '''Turns an input string into a variable, if it can.

    Variables tried: bool, int, float, pd.Period
    '''
    if string is None:
        print('variablising but None for string')
        return None

    if string.strip().lower() == 'true':
        return True
    elif string.strip().lower() == 'false':
        return False
    
    else:
        try: 
            return int(string)
        except:
            try:
                return float(string)
            except:
                try:
                    return pd.Period(string)
                except: return string.strip()


 ###_________________________________________________________________________###


def slicify(in_string):
    '''
    Processes index_slice input strings from web form into variables useable by 
    get_ix_slice. 
    '''
    print('calling slicify on ', in_string)

    if in_string == "" or in_string == None:
        return slice(None, None, None)

    # see if string can be variablised (is a bool, float, int or pd.Period)
    v = variablise(in_string)
    if not isinstance(v, str):
        print("..variablised as ", type(v))
        return v
    
    # now deal with list-likes
    if in_string.strip()[0] == "[" and in_string.strip()[-1] == "]":
        out_list = [variablise(i) for i in in_string.strip()[1:-1].split(',')]
        print("..listified as ", out_list)
        return out_list
    
    # finally deal with slices
    if in_string.strip().startswith('slice:'):
        print('looks like a slice.. ', end = "")
        slice_out = in_string[6:].strip().split(' ')
        
        if slice_out[0] == 'to':
            print(' with to only')
            return slice(None, variablise(slice_out[1]),None)

        if slice_out[-1] == 'to':
            print(' with from and to')
            return slice(variablise(slice_out[0]),None,None)

        if slice_out[1] == 'to' and len(slice_out)==3:
            print(' with from only')
            return slice(variablise(slice_out[0]),variablise(slice_out[2]),None)
        
        else:
            print('could not slicify slice.  Returning nothing from slicify()', in_string)

    else:
        print('could not slicify string, returning unprocessed ', in_string)
        return in_string



###_________________________________________________________________________###




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
    in_arr[np.isnan(in_arr)]=0

    a = np.cumsum(in_arr) # total cumulative sum
    b=(np.cumsum(in_arr)[:-window]) # shifted forward, overlap truncated
    c = np.insert(b,0,np.zeros(window))  # start filled to get to line up
    return(a-c)/window

    
 ###_________________________________________________________________________###

