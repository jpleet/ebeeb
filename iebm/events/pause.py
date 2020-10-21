import numpy as np
import datatable as dt

from .base import Event

class Pause2DEvent(Event):
    
    def __init__(self, population, params):
        """Construct Pause Event with given hard-coded name

        Parameters
        ----------

        population : class Population
            the Population class that performs the actions

        params : dict
            *must contain:
            - 'current_time'

        """
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None

        # initialize base event
        # hard-coded as a secondary event
        super().__init__(population, params['name'], 
                         False, triggers)
            
        if 'ignore_list' in params:
            self.ignore_list = [f'{c}_time' for c in params['ignore_list']]
        else:
            self.ignore_list = []
        self.ignore_list += [f'{str(self)}_time']
        self.set_list = []
        
        
    def set_next(self, params):
        
        # get actor id from dictionary
        actor_id = params['actor_id']
        actor_idx = self.population._get_actor_idx(actor_id)
        
        new_events = []
        
        if actor_idx is not None:

            cv, cvx, cvy, pt, status = self.population.df[actor_idx, 
                                                  ['velocity', 'vel_x', 'vel_y', 
                                                   f'{str(self)}', 'status']].to_numpy()[0]
            
            if status == 'active':
            
                self.population.df[actor_idx, ['velocity', 'vel_x', 'vel_y']] = 0 
                self.population.df[actor_idx, self.set_list] = np.nan
                self.population.df[actor_idx, 'status'] = 'inactive'

                end_time = params['current_time'] + pt 
                new_params = dict(actor_id = actor_id, 
                                  current_time = end_time,
                                  cv = cv,
                                  cvx = cvx,
                                  cvy = cvy)
                
                event_hash = hash(f'{end_time}_{self}_' + 
                                  '_'.join(str(params[k]) for k in params))

                new_events += [(end_time, event_hash, self, new_params)]
                new_events += [self.population.get_next_event(actor_id)]
            
        return new_events
        
    
    def handle(self, params):
        """ function that performs the event action. StopEvent sets the velocity
        and velocity components of an individual to zero

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
        actor_id = params['actor_id']
        actor_idx = self.population._get_actor_idx(actor_id)
        
        new_events = []
        
        if actor_idx is not None:
            
            self.population.df[actor_idx, 'velocity'] = params['cv']
            self.population.df[actor_idx, 'vel_x'] = params['cvx']
            self.population.df[actor_idx, 'vel_y'] = params['cvy']
            self.population.df[actor_idx, 'status'] = 'active'
            del params['cv']
            del params['cvx']
            del params['cvy']
            
            if self.triggers:
                new_events += self.triggers(params)
        
        return new_events