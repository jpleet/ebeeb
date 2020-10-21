import numpy as np
from .base import Trait

class StaticTrait(Trait):
    
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
        
        self.value = params['value']
        
        values = [self.value] * self.population.size
        self.population.df[str(self)] = np.array(values)
        
    def get_value(self, actor_id):
        "simply return the static value"
        return self.value
    
    def inherit_value(self, parent_id):
        "simply return the static value"
        return self.value
        