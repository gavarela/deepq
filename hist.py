## Draw histogram on terminal
## ~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

def to_bag(lst):
    ''' lst should be a list-like object (list or array) with occurences of values (be these numbers or strings). to_bag will return a dictionary with the values as keys and the number of occurences for each value as the value. '''
    
    # Make lst a one-dim array of scalars
    lst = np.array(lst)
    
    if len(lst.shape) > 1:
        for dim in range(1, len(lst.shape)):
            assert dim == 1, "lst must be array or list of scalars"
        lst = lst.reshape(-1)
    
    # Make bag
    bag = {}
    vals = []
    for a in set(lst):
        val = np.sum(lst == a)
        bag[a] = val
        vals.append(val)
    
    bag['total'] = len(lst)
    bag['max'] = max(vals)
    
    return bag


def all_nums(lst):
    ''' Get list. Returns True if all elements are numbers. '''

    string = ''
    for a in lst:
        string += str(a)
    
    return string.isdigit()
    

def binned(bag, num_bins):
    ''' Separates into given number of bins'''
    
    # Get keys
    keys = list(bag.keys())
    keys.remove('total')
    keys.remove('max')
    keys.sort()
    
    # Put into bins
    maxkey = max(keys)
    minkey = min(keys)
    step = (maxkey - minkey)/num_bins
    
    i = 0
    bins = np.zeros(num_bins)
    labs = []
    for b in range(num_bins):
        rnge = (minkey + b*step, minkey + (b+1)*step)
        while rnge[0] <= keys[i] < rnge[1]:
            bins[b] += bag[keys[i]]
            i += 1
        if b == num_bins-1:
            bins[b] += bag[keys[i]]
        # Label to 2 decimal places
        labs.append(str(int(((rnge[0] + rnge[1])/2)*100)/100.0))
    
    return labs, bins
        
        


def hist(lst, num_bins = -1, title = '', allnames = False, max_height = 20):
    ''' Prints a simple histogram of the lst values in the terminal. '''
    
    # Make bag and get keys
    bag = to_bag(lst)
    
    # Set number of bins and get bins
    if num_bins != -1:
        labs, bins = binned(bag, num_bins)
    
    ## Make string from bottom up
    
    ## X labels and axis
    xaxis = ''
    xlabs = ''
    
    # Spaces before y - axis
    spaces_till_y = len(str(bag['max']))
    xaxis += ' '*(spaces_till_y + 1)
    xlabs += ' '*(spaces_till_y + 1)
    
    # Some spaces
    xaxis += '--'
    xlabs += '  '
    
    # Two spaces for each bin plus one at the end
    xaxis += '-'*(2 * num_bins + 4)
    xlabs += ' '*(2 * num_bins + 4)
    
    # Add labels
    if allnames:
        assert True
    else:
        for i in (0, len(labs)//2, len(labs) - 1):
            xlabs = xlabs[:spaces_till_y + 3 + 2*i + 1] + \
                        str(labs[i]) + \
                        xlabs[spaces_till_y + 3 + 2*i + 1 + len(str(labs[i])):]
    
    ## Work up from there
    strs = []
    for h in range(max_height):
        strs.append('')
        
        # y-label
        ymax = min(bins) + (max(bins) - min(bins))/20*(h+1)
        ymin = min(bins) + (max(bins) - min(bins))/20*h
        
        ylab = (ymax + ymin)/2
        ylab = str(int(ylab))
        
        # Space before y-label 
        strs[h] += ' '*(spaces_till_y-len(ylab)) + ylab + '|   '
        
        # Add counters (' o' if bin reaches here, '  ' if not)
        for b in range(num_bins):
            if bins[b] > ymin:
                strs[h] += ' o'
            else:
                strs[h] += '  '
    
    # Print it all
    print('\n' + ' '*(spaces_till_y + 3) + title + '\n')
    for line in reversed(strs):
        print(line)
    print(xaxis)
    print(xlabs)


if __name__ == "__main__":
    
    b = [1, 2, 1, 2, 3, 2, 3, 2, 1, 2, 3, 4, 3, 4, 3, 2, 2, 2, 2, 3, 2, 3, 4, 5, 6, 5, 5, 6, 7, 8, 10, 8, 4, 4, 3, 2, 4, 2, 2, 2, 2, 3, 2, 5, 5, 1, 2, 1, 2, 2, 3]
    
    hist(b, 5, 'Histogram with five bins')
    
    print()
    
    hist(b, 10, 'Hist with 10 bins')