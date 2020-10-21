import numpy as np
from .base import Trait

class LinkedTrait(Trait):
    
    def __init__(self, population, params):
        """ Constructor for a linked individual-level trait.

        Parameters
        ----------

        population : Population
            reference to individual-level population class

        params : dict
            dictionary with trait parameters. The parameters determine the
            type of trait. If key is omitted, considered None
            Linked trait keys:
                name : trait identifier (str)
                link_trait : string of linked trait
                link_func : lambda function to get this trait value from link
        """
        super().__init__(population, params)
        
        self.link_trait = params['link_trait']
        self.link_func = params['link_func']
        # get initial values for all individuals
        pop_ids = self.population.df.to_numpy(
            column=self.population.df.colindex('id'))
        values = [self.get_value(p) for p in pop_ids]
        self.population.df[str(self)] = np.array(values)
        
    def get_value(self, actor_id):
        """function to get and return trait value for a specific actor inidividual in
        a population.

        Parameters
        ----------

        actor_id : int
            identifier interger to a specific acting individual

        """
        
        link_val = self.population.trait_dict[
            self.link_trait].get_value(actor_id)
        # apply link function to linked value and return
        return self.link_func(link_val)
    
    def inherit_value(self, parent_id):
        """function to pass trait values from a parent to an offspring,
        mutation would depend on linked trait

        Parameters
        ----------

        parent_id : int
            identifier interger for the parent
        """
        return self.get_value(parent_id)