import numpy as np
np.seterr(divide='ignore')
import datatable as dt
from datatable import f

from .base import Event

import warnings

class Interact2DEvent(Event):
    
    def __init__(self, population, params):
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None
        
        super().__init__(population, params['name'], 
                         params['is_primary'], triggers)
        
        if 'trigger_set_next' in params:
            self.trigger_set_next = params['trigger_set_next']
        else:
            self.trigger_set_next = False

        if 'other' in params:
            self.other = params['other']
            t1, t2 = self.get_interact_times_all_main_all_other()
        else:
            self.other = None
            t1, t2 = self.get_interact_times_all_same()
        
        interact_times = np.minimum(t1, t2) + params['current_time']    
        _, p1_ids = np.where(~np.isnan(interact_times))
        p1_ids = np.unique(p1_ids)
        p2_ids = np.nanargmin(interact_times[:, p1_ids], axis=0)   
        
        if self.other is not None:
            other_ids = self.other.df[p2_ids, 'id']
        else:
            other_ids = self.population.df[p2_ids, 'id']
        
        interact_df = dt.Frame({
            "id" : self.population.df[p1_ids, 'id'], 
            f"{self}_extra" : other_ids, 
            f"{self}_time"  : interact_times[(p2_ids, p1_ids)] + params['current_time']})
        interact_df.key = 'id'
        self.population.df = self.population.df[:, :, dt.join(interact_df)]
        
        
    def get_interact_times_all_main_all_other(self):
        
        if 'vel_x' in self.population.df.names:
            p_vx = self.population.df.to_numpy(column=self.population.df.colindex('vel_x'))
            p_vy = self.population.df.to_numpy(column=self.population.df.colindex('vel_y'))
        else:
            p_vx = np.zeros(self.population.df.shape[0])
            p_vy = np.zeros(self.population.df.shape[0])
        p_x = self.population.df.to_numpy(column=self.population.df.colindex('x')) 
        p_y = self.population.df.to_numpy(column=self.population.df.colindex('y'))
        p_r = self.population.df.to_numpy(column=self.population.df.colindex(f'{str(self)}_radius'))
        
        if 'vel_x' in self.other.df.names:
            n_vx = self.other.df.to_numpy(column=self.other.df.colindex('vel_x')).reshape(-1,1)
            n_vy = self.other.df.to_numpy(column=self.other.df.colindex('vel_y')).reshape(-1,1)
        else:
            n_vx = np.zeros(self.other.df.shape[0]).reshape(-1,1)
            n_vy = np.zeros(self.other.df.shape[0]).reshape(-1,1)
        n_x = self.other.df.to_numpy(column=self.other.df.colindex('x')).reshape(-1,1)
        n_y = self.other.df.to_numpy(column=self.other.df.colindex('y')).reshape(-1,1)
        n_r = self.other.df.to_numpy(column=self.other.df.colindex(f'{str(self)}_radius')).reshape(-1,1)
        
        return self.calculate_interact_times(p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r)
    
    
    def get_interact_times_single_main_all_other(self, actor_idx):
        
        if 'vel_x' in self.population.df.names:
            p_vx = self.population.df[actor_idx, 'vel_x']
            p_vy = self.population.df[actor_idx, 'vel_y']
        else:
            p_vx = 0
            p_vy = 0
        p_x = self.population.df[actor_idx, 'x']
        p_y = self.population.df[actor_idx, 'y']
        p_r = self.population.df[actor_idx, f'{str(self)}_radius']
                
        if 'vel_x' in self.other.df.names:
            n_vx = self.other.df.to_numpy(column=self.other.df.colindex('vel_x'))
            n_vy = self.other.df.to_numpy(column=self.other.df.colindex('vel_y'))
        else:
            n_vx = 0
            n_vy = 0
        n_x = self.other.df.to_numpy(column=self.other.df.colindex('x'))
        n_y = self.other.df.to_numpy(column=self.other.df.colindex('y'))
        n_r = self.other.df.to_numpy(column=self.other.df.colindex(f'{str(self)}_radius'))
        
        return self.calculate_interact_times(p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r)
    

    def get_interact_times_all_main_single_other(self, other_idx):
        
        if 'vel_x' in self.population.df.names:
            p_vx = self.population.df.to_numpy(column=self.population.df.colindex('vel_x'))
            p_vy = self.population.df.to_numpy(column=self.population.df.colindex('vel_y'))
        else:
            p_vx = np.zeros(self.population.df.shape[0])
            p_vy = np.zeros(self.population.df.shape[0])
        p_x = self.population.df.to_numpy(column=self.population.df.colindex('x')) 
        p_y = self.population.df.to_numpy(column=self.population.df.colindex('y'))
        p_r = self.population.df.to_numpy(column=self.population.df.colindex(f'{str(self)}_radius'))

        if 'vel_x' in self.other.df.names:
            n_vx = self.other.df[other_idx, 'vel_x']
            n_vy = self.other.df[other_idx, 'vel_y']
        else:
            n_vx = 0
            n_vy = 0
        n_x = self.other.df[other_idx, 'x']
        n_y = self.other.df[other_idx, 'y']
        n_r = self.other.df[other_idx, f'{str(self)}_radius']

        return self.calculate_interact_times(p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r)
    

    def get_interact_times_all_same(self):
        
        if 'vel_x' in self.population.df.names:
            p_vx = self.population.df.to_numpy(column=self.population.df.colindex('vel_x'))
            p_vy = self.population.df.to_numpy(column=self.population.df.colindex('vel_y'))
        else:
            p_vx = np.zeros(self.population.df.shape[0])
            p_vy = np.zeros(self.population.df.shape[0])
        p_x = self.population.df.to_numpy(column=self.population.df.colindex('x')) 
        p_y = self.population.df.to_numpy(column=self.population.df.colindex('y'))
        p_r = self.population.df.to_numpy(column=self.population.df.colindex(f'{str(self)}_radius'))
        
        n_vx = p_vx.copy().reshape(-1,1)
        n_vy = p_vy.copy().reshape(-1,1)
        n_x = p_x.copy().reshape(-1,1)
        n_y = p_y.copy().reshape(-1,1)
        n_r = p_r.copy().reshape(-1,1)
        
        return self.calculate_interact_times(p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r)
        

    def get_interact_times_single_same(self, actor_idx):
        
        if 'vel_x' in self.population.df.names:
            p_vx = self.population.df[actor_idx, 'vel_x']
            p_vy = self.population.df[actor_idx, 'vel_y']
        else:
            p_vx = 0
            p_vy = 0
        p_x = self.population.df[actor_idx, 'x']
        p_y = self.population.df[actor_idx, 'y']
        p_r = self.population.df[actor_idx, f'{str(self)}_radius']
                
        if 'vel_x' in self.population.df.names:
            n_vx = self.population.df.to_numpy(column=self.population.df.colindex('vel_x'))
            n_vy = self.population.df.to_numpy(column=self.population.df.colindex('vel_y'))
        else:
            n_vx = 0
            n_vy = 0
        n_x = self.population.df.to_numpy(column=self.population.df.colindex('x'))
        n_y = self.population.df.to_numpy(column=self.population.df.colindex('y'))
        n_r = self.population.df.to_numpy(column=self.population.df.colindex(f'{str(self)}_radius'))
        
        return self.calculate_interact_times(p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r)
  

    def calculate_interact_times(self, p_vx, p_vy, p_x, p_y, p_r, n_vx, n_vy, n_x, n_y, n_r):
    
        a = n_vx**2 - 2*n_vx*p_vx + n_vy**2 - 2*n_vy*p_vy + p_vx**2 + p_vy**2
        b = (2*n_vx*n_x - 2*n_vx*p_x + 2*n_vy*n_y - 2*n_vy*p_y - 2*n_x*p_vx 
             - 2*n_y*p_vy + 2*p_vx*p_x + 2*p_vy*p_y)
        c = (-n_r**2 - 2*n_r*p_r + n_x**2 - 2*n_x*p_x + n_y**2 - 2*n_y*p_y - 
             p_r**2 + p_x**2 + p_y**2)

        determinant = b**2 - 4 * a * c
        determinant[determinant<0] = np.nan
        
        # remove warnings for own collision
        if isinstance(a, (np.ndarray, np.generic)):
            a[np.isclose(a, 0)] = np.nan
        # sometimes sessile individuals interact
        #elif a == 0:
        #    a = np.nan

        t1 = (-b + np.sqrt(determinant)) / (2 * a)
        t2 = (-b - np.sqrt(determinant)) / (2 * a)
        
        t1[np.less(t1, 0, where=~np.isnan(t1))] = np.nan
        t2[np.less(t2, 0, where=~np.isnan(t2))] = np.nan

        return t1, t2
    

    def set_next(self, params):
        
        if self.is_primary:
        
            # make sure there is a row index for actor id
            actor_id = params['actor_id']
            actor_idx = self.population._get_actor_idx(actor_id)
            
            if actor_idx is not None:
                
                status = self.population.df[actor_idx, 'status']
                
                if status == 'active':
                
                    # if extra exists, make sure to ignore same to not repeat interaction
                    if 'extra' in params:
                        other_id = params['extra']
                        if ~isinstance(other_id, (int,float)):
                            other_id = None
                            other_idx = None
                    else:
                        other_id = None
                        other_idx = None

                    if self.other is not None:
                        t1, t2 = self.get_interact_times_single_main_all_other(actor_idx)

                        if other_id is not None:
                            other_idx = self.other._get_actor_idx(other_id)

                    else:
                        t1, t2 = self.get_interact_times_single_same(actor_idx)

                        if other_id is not None:
                            other_idx = self.population._get_actor_idx(other_id)

                    interact_times = np.minimum(t1, t2)

                    if other_idx is not None:
                        interact_times[other_idx] = np.nan
                
                else:
                    interact_times = [np.nan]

                min_time = None
                min_actor = None
                if np.nansum(interact_times) > 0:
                    min_time = np.nanmin(interact_times) + params['current_time']
                    min_actor = int(np.nanargmin(interact_times))
                    if self.other is not None:
                        min_actor = self.other.df[min_actor, 'id']
                    else:
                        min_actor = self.population.df[min_actor, 'id']

                self.population.df[actor_idx, f'{self}_time'] = min_time
                self.population.df[actor_idx, f'{self}_extra'] = min_actor
                
        return []

            
    def set_other_next(self, params):
        
        new_events = []
        
        if self.is_primary:
        
            other_id = int(params['actor_id']) # from birth likely
            other_idx = self.other._get_actor_idx(other_id)

            if other_idx is not None:

                t1, t2 = self.get_interact_times_all_main_single_other(other_idx)
                interact_times = np.minimum(t1, t2) 

                new_interactions = ~np.isnan(interact_times)

                if new_interactions.any():

                    positive_idxs = np.argwhere(new_interactions).reshape(1,-1)[0]
                    positive_times = interact_times[new_interactions] + params['current_time']

                    to_update = self.population.df[positive_idxs, 
                                                   ((dt.f[f'{self}_time'] > positive_times) | 
                                                    (dt.isna(dt.f[f'{self}_time'])))].to_numpy(column=0)

                    if to_update.any():

                        update_idxs = positive_idxs[to_update]
                        update_times = positive_times[to_update]

                        self.population.df[update_idxs, f'{self}_time'] = update_times
                        self.population.df[update_idxs, f'{self}_extra'] = other_id
                        
                        
                        # add new interaction if sooner than next event
                        next_times = self.population.df[update_idxs, self.population.event_list].to_numpy().min(1)
                        interact_next = next_times == update_times
                        ids = self.population.df[update_idxs[interact_next], 'id'].to_numpy().reshape(1,-1)[0]
                        for i in ids:
                            new_events += [self.population.get_next_event(i)]
                        
                        
            return new_events


    def handle(self, params, eps=0.00001):

        events = []

        actor_id = int(params['actor_id'])
        actor_idx = self.population._get_actor_idx(actor_id)
        
        if actor_idx is not None:

            status = self.population.df[actor_idx, 'status']
            
            if status == 'active':
            
                other_id = int(params['extra'])
                if self.other is not None:
                    other_idx = self.other._get_actor_idx(other_id)
                else:
                    other_idx = self.population._get_actor_idx(other_id)

                if other_idx is not None:

                    actor_x, actor_y, actor_r = self.population.df[actor_idx, ['x', 'y', f'{str(self)}_radius']].to_numpy()[0]

                    if self.other is not None:
                        other_x, other_y, other_r = self.other.df[other_idx, ['x', 'y', f'{str(self)}_radius']].to_numpy()[0]
                    else:
                        other_x, other_y, other_r = self.population.df[other_idx, ['x', 'y', f'{str(self)}_radius']].to_numpy()[0]

                    dist = np.sqrt((actor_x-other_x)**2 + (actor_y-other_y)**2)
                    r = actor_r + other_r

                    # make sure actually close, maybe previously event changed actor's direction
                    if dist <= r + eps:
                        if self.triggers:
                            events += self.triggers(params)
  
                else:
                    # no other actor, so no event actually happens
                    # sometimes triggers set next, but that won't happen
                    # so need to set next if trigger was supposed to
                    if self.trigger_set_next & self.is_primary:
                        self.set_next(params)
                        events += [self.population.get_next_event(actor_id)]
                
                # events done, set next unless done by the triggers
                if self.is_primary & (self.trigger_set_next==False):
                    self.set_next(params)
                    events += [self.population.get_next_event(actor_id)]
        
        return events
    