import numpy as np
import datatable as dt

from .base import Event


class DeathEvent(Event):
    """ Individual-level event that:
    handle(parmas): removes an individual from the population.
    set_next(id): calculates and sets next death time for an individual.
    DeathEvent is a good example of an event that could be secondary ---
    prey might have a death event after interacting with predators. Death by
    consumptions is a secondary event and not written in the dataframe.
    """

    def __init__(self, population, params):
        """Construct Death Event with given hard-coded name

        Parameters
        ----------

        population : class Population
            the Population class that performs the actions

        params : dict
            *must contain:
            - 'death_rate' key and value
            - 'current_time'

        """
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None

        # initialize base event
        super().__init__(population, params['name'], 
                         params['is_primary'], triggers)
        
        # check if this is a primary event
        if self.is_primary:
            # get individual death rates
            death_rates = self.population.df.to_numpy(
                column=self.population.df.colindex('death_rate'))
            # calculate and set individual death times
            self.population.df[f'{self}_time'] =  (np.random.exponential(1 / death_rates)
                                                   + params['current_time'])


    def set_next(self, params):
        """ function to set next event time for an individual

        Parameters
        ----------

        """

        # check if a primary event and should be set
        if self.is_primary:
            # get row number for actor in population dataframe
            actor_id = params['actor_id']
            actor_idx = self.population._get_actor_idx(actor_id)
            # make sure actor does exist
            if actor_idx is not None:
                # extract individual death rate
                death_rate = self.population.df[actor_idx, 'death_rate']
                # draw random death_time from death_rate
                death_time = (np.random.exponential(1 / death_rate) +
                              params['current_time'])
                # assign next death time to individual
                self.population.df[actor_idx, f'{self}_time'] = death_time


    def handle(self, params):
        """ function that performs the event action. DeathEvent remove an
        individual row from the population dataframe and lowers the population

        Parameters:
        -----------

        params : dict
            dictionary containing 'actor_id' as key with the actual actor id as
            the value

        Returns
        -------

        empty list - no new events for the dead individual
        """

        # get actor id from dictionary
        actor_id = int(params['actor_id'])
        actor_idx = self.population._get_actor_idx(actor_id)
        
        new_events = []
        
        if actor_idx is not None:

            # call trigger first, before removing individual
            if self.triggers:
                new_events += self.triggers(params)
                
            # remove row with id
            del self.population.df[actor_idx, :]
            # decrease population count
            self.population.size -= 1
        

        
        return new_events