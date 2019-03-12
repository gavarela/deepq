## ~~~~~~~~~~~~~~~~~~~~~~~~
#  Memory Class for QPlayer
## ~~~~~~~~~~~~~~~~~~~~~~~~


import random

class Memory(object):
    
    def __init__(self, max_len):
        
        self.memory = []
        self.max_len = max_len
        
    def __getitem__(self, i):
        
        return self.memory[i]
        
    def __len__(self):
        
        return len(self.memory)
    
    def is_full(self):
        
        return len(self) == self.max_len
    
    def append(self, items, multiple = True, drop_rand = False):
        ''' Adds \items to memory. 
            \multiple indicates if \items is a list of items to added individually, or a single item to be added. 
            \drop_rand indicates whether, in case the memory exceeds the maximum length, to drop items at random or to drop the oldest ones. '''
        
        if multiple:
            n_items = len(items)
        else:
            n_items = 1
        
        overload = (len(self) + n_items) - self.max_len
        if overload > 0:
            if drop_rand:
                for _ in range(overload):
                    rand_ind = random.choice(range(len(self)))
                    del self.memory[rand_ind]
            else:
                self.memory = self.memory[overload:]
            
        if multiple:
            for item in items: self.memory.append(item)
        else:
            self.memory.append(items)
    
    def clear(self):
        
        self.memory = []
        