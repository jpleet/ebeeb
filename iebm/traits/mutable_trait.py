import numpy as np
from .base import Trait

class MutableTrait(Trait):
    
    def __init__(self, population, params):
        """ Constructor for a mutable individual-level trait.

        Parameters
        ----------

        population : Population
            reference to individual-level population class

        params : dict
            dictionary with trait parameters. The parameters determine the
            type of trait. If key is omitted, considered None
            Primary mutatable trait keys:
                name : unique trait identifier (str)
                value : starting initial trait value (float)
                min_value: smallest value trait can evolve towards (float)
                max_value: largest value trait can evolve towards (float)
                mutate_rate : Poisson distribution rate
                mutate_step : jump value (multiplied by random Poisson value)

        """
        
        super().__init__(population, params)
        
        self.value = params['value']
        
        if 'min_value' in params:
            self.min_value = params['min_value']
        else:
            self.min_value = None
        
        if 'max_value' in params:
            self.max_value = params['max_value']
        else:
            self.max_value = None
        
        if 'mutate_rate' in params:
            self.mutate_rate = params['mutate_rate']
            self.mutate_step = params['mutate_step']
        else:
            self.mutate_rate = None
            self.mutate_step = None
        
        values = [self.value] * self.population.size
        values = [self.mutate(v) for v in values]
        self.population.df[str(self)] = np.array(values)
        
    def get_value(self, actor_id):
        """function to get and return trait value for a specific actor inidividual in
        a population.

        Parameters
        ----------

        actor_id : int
            identifier interger to a specific acting individual

        """
        
        actor_idx = self.population._get_actor_idx(actor_id)
        if actor_idx in not None:
            # return value at row number and trait column name
            return self.population.df[actor_idx, str(self)]
        else:
            return None
    
    def inherit_value(self, parent_id):
        """function to pass trait values from a parent to an offspring,
        includes a mutation possibility

        Parameters
        ----------

        parent_id : int
            identifier interger for the parent
        """

        parent_idx = self.population._get_actor_idx(parent_id)
        parent_val = self.population.df[parent_idx, str(self)]
        off_val = self.mutate(parent_val)
        return off_val
        
    
    def mutate(self, value):
        """Given a value and a mutable trait, possibly change value. Otherwise,
        the passed value is returned as is.

        Parameters
        ----------

        value : float
            value that might mutate
        """

        if self.mutate_rate:
            # draw a random step size
            step = np.random.poisson(self.mutate_rate)
            # if there is a mutation, adjust value
            if step > 0:
                # add a chance that the step size could go down
                if np.random.rand() < 0.5:
                    step = -step
                # update value based on step size and step value
                val = value + step * self.mutate_step
                # check to make sure new val in proper range
                if self.min_value:
                    if val < self.min_value:
                        val = self.min_value
                if self.max_value:
                    if val > self.max_value:
                        val = self.max_value
                # return new val
                return val
        # otherwise, return initial passed value
        return value        
        