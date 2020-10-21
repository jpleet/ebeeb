import numpy as np
import rtree
from scipy.spatial.distance import cdist

from .base import Event


class RotateEvent(Event):
    """ Event to turn (rotate) moving individuals towards a
    stationary individuals stored in an Rtree. 

    TO DO: rotate towards close stationary or moving
    individuals. stationary is easier, can use an Rtree.
    maybe store moving locations in an IVF flat from FAISS
    """

    def __init__(self, population, params):
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None
        
        # initialize base event
        super().__init__(population, params['name'], 
                         params['is_primary'], triggers)
        
        self.attract_pop = params['attract_population']
        
        self.attract_index = rtree.index.Index()
        for i,x,y in self.attract_pop.df[:, ['id', 'x', 'y']].to_tuples():
            self.attract_index.insert(i, (x,y))
        
        if self.is_primary:
            rotate_rates = self.population.df.to_numpy(
                column=self.population.df.colindex(f'{self}_rate'))
            rotate_times = (np.random.exponential(1 / rotate_rates) +
                           params['current_time'])
            self.population.df[f'{self}_time'] = rotate_times
        
        
    def set_next(self, params):
        if self.is_primary:
            actor_id = params['actor_id']
            actor_idx = self.population._get_actor_idx(actor_id)
            if actor_idx is not None:
                rotate_rate = self.population.df[actor_idx, f'{self}_rate']
                rotate_time = (np.random.exponential(1 / rotate_rate) + params['current_time'])
                self.population.df[actor_idx, f'{self}_time'] = rotate_time
        
    def handle(self, params):
        new_events = []
        
        actor_id = params['actor_id']
        actor_idx = self.population._get_actor_idx(actor_id)
        
        if actor_idx:
            
            actor_x, actor_y, actor_z = self.population.df[actor_idx, ['x', 'y', 
                                                                       f'{self}_radius']].to_numpy()[0]
            search_xmin = actor_x - actor_z
            search_ymin = actor_y - actor_z
            search_xmax = actor_x + actor_z
            search_ymax = actor_y + actor_z
            
            neighs_id = self.attract_index.intersection((search_xmin, search_ymin,
                                                         search_xmax, search_ymax))
            neighs_idx = [self.attract_pop._get_actor_idx(n) for n in neighs_id]
            
            if len(neighs_idx) > 0:
            
                neigh_points = self.attract_pop.df[neighs_idx, ['x','y']].to_numpy()
                min_arg = cdist([[actor_x, actor_y]], neigh_points).argmin()
                min_x, min_y = neigh_points[min_arg]
                new_ang = np.arctan2(*(min_y - actor_y, min_x - actor_x))
                
                self.population.df[actor_idx, 'angle'] = new_ang
                
                if self.triggers:
                    new_events += self.triggers(params) 
        
        # set next, if primary
        if self.is_primary:
            self.set_next(params)
            new_events += [self.population.get_next_event(actor_id)]

        return new_events

    def add_attracted(self, params):
        attracted_id = int(params['actor_id'])
        attracted_idx = self.attract_pop._get_actor_idx(attracted_id)
        x,y = self.attract_pop.df[attracted_idx, ['x', 'y']].to_numpy()[0]
        self.attract_index.insert(attracted_id, (x,y))
        return []

    def remove_attracted(self, params):
        attracted_id = int(params['actor_id'])
        attracted_idx = self.attract_pop._get_actor_idx(attracted_id)
        x,y = self.attract_pop.df[attracted_idx, ['x', 'y']].to_numpy()[0]
        self.attract_index.delete(attracted_id, (x,y))
        return []
    
        
        
        