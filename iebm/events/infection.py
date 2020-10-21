import numpy as np
from .base import Event

class InfectionSIREvent(Event):
    
    def __init__(self, population, params):
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None
        
        super().__init__(population, params['name'], 
                         params['is_primary'], triggers)
        
    def set_next(self, params):

        new_events = []
        
        actor_id = params['actor_id']
        actor_idx = self.population._get_actor_idx(actor_id)
        
        if actor_idx is not None:
            recovery_rate = self.population.df[actor_idx, f'recovery_rate']
            recovery_time = np.random.exponential(1 / recovery_rate) + params['current_time']
            params['recover'] = True
            
            event_hash = hash(f'{recovery_time}_{self}_' + 
                              '_'.join(str(params[k]) for k in params))
            
            new_events += [(recovery_time, event_hash, self, params)]
            
        return new_events

    def handle(self, params):
        
        new_events = []
        
        actor_id = int(params['actor_id'])
        actor_idx = self.population._get_actor_idx(actor_id)
        
        if actor_idx is not None:
            
            # get status
            status = self.population.df[actor_idx, str(self)]
            
            # check if collision and extra param exists
            if 'extra' in params:
                other_id = params['extra']
                other_idx = self.population._get_actor_idx(other_id)
                other_status = self.population.df[other_idx, str(self)]
            else:
                other_status = None
            
            if status == 'susceptible':
                
                if other_status == 'infected':
                    self.population.df[actor_idx, str(self)] = 'infected'
                    new_events += self.set_next(params)
                    
            if status == 'infected':
                
                if other_status == 'susceptible':
                    self.population.df[other_idx, str(self)] = 'infected'
                    other_params = params.copy()
                    other_params['actor_id'] = other_id
                    new_events += self.set_next(other_params)
                    
                if 'recover' in params:
                    if params['recover']:
                        self.population.df[actor_idx, str(self)] = 'recovered'
                        del params['recover']

        return new_events