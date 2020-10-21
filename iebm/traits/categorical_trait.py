import numpy as np
from .base import Trait

class CategoricalTrait(Trait):
    
    def __init__(self, population, params):
        """ Constructor for a static individual-level trait.

        Parameters
        ----------

        population : Population
            reference to individual-level population class

        params : dict
            dictionary with trait parameters. The parameters determine the
            type of trait. If key is omitted, considered None
            required keys:
                name : unique identifier (str)
                value : trait value (float)
        """
        
        super().__init__(population, params)
        
        self.categories = params['categories']
        init_fractions = params['fractions']
        
        cats = [c for t, frac in enumerate(init_fractions) 
                for c in [self.categories[t]]*int(frac*self.population.size)]
        cats = np.array(cats)
        np.random.shuffle(cats)
        self.population.df[str(self)] = np.array(cats)
        
    def get_value(self, actor_id):
        
        actor_idx = self.population._get_actor_idx(actor_id)
        if actor_idx is not None:
            # return value at row number and trait column name
            return self.population.df[actor_idx, str(self)]
        else:
            return None
    
    def inherit_value(self, parent_id):
        
        parent_idx = self.population._get_actor_idx(parent_id)
        if parent_idx is not None:
            # return value at row number and trait column name
            return self.population.df[parent_idx, str(self)]
        else:
            return None