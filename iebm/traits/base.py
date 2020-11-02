from abc import ABC, abstractmethod
import numpy as np


class Trait(ABC):
    
    def __init__(self, population, params):
        """ Base individual-level trait. All trait subclasses inherit
        from this base class.

        """        
        self.name = params['name']
        self.population = population
           
        if 'track' in params:
            self.track = params['track']
        else:
            self.track = False
            
        self.gene_dict = {}

    @abstractmethod
    def get_value(self, actor_id):
        "function to overwrite with subclass method to get value of actor"
        pass
    
    @abstractmethod
    def inherit_value(self, parent_id):
        "function to overwrite with subclass method to get value of parent"
        pass
    
    def track_values(self):
        "helper funtion to keep track of trait values"
        vals, counts = np.unique(self.population.df[str(self)].to_numpy(column=0),
                                 return_counts=True)
        return vals, counts

    def __repr__(self):
        return self.name