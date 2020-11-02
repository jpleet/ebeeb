import numpy as np
import datatable as dt
# modules to find open space
import rtree
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

from .base import Event


class BirthAsexualEvent(Event):

    def __init__(self, population, params):
        
        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None

        super().__init__(population, params['name'],
                         params['is_primary'], triggers)

        if self.is_primary:
            birth_rates = self.population.df.to_numpy(
                column=self.population.df.colindex('birth_rate'))
            birth_times = (np.random.exponential(1 / birth_rates) +
                           params['current_time'])
            self.population.df[f'{self}_time'] = birth_times

    def set_next(self, params):

        if self.is_primary:
            actor_id = params['actor_id']
            actor_idx = self.population._get_actor_idx(actor_id)
            if actor_idx is not None:
                birth_rate = self.population.df[actor_idx, 'birth_rate']
                birth_time = (np.random.exponential(1 / birth_rate) + params['current_time'])
                self.population.df[actor_idx, f'{self}_time'] = birth_time


    def handle(self, params, position_func=None):
        
        actor_id = int(params['actor_id'])
        new_events = []
        
        actor_idx = self.population._get_actor_idx(actor_id)
        
        if actor_idx:
            
            if 'number_offspring' in self.population.trait_dict:
                num_off = self.population.df[actor_idx, 'number_offspring']
            else:
                num_off = 1
                
            for _ in range(num_off):
                
                have_birth = True

                if self.population.implicit_capacity:
                    birth_prob = (1 - self.population.size / self.population.implicit_capacity)
                    if np.random.rand() > birth_prob:
                        have_birth = False
        
                if 'conversion_efficiency' in self.population.trait_dict:
                    ce = self.population.df[actor_idx, 'conversion_efficiency']
                    if np.random.rand() > ce:
                        has_birth = False
        
                if have_birth:

                    new_id = self.population.id_count
                    
                    # create new traits
                    new_traits = {}
                    for k in self.population.trait_dict:
                        val = self.population.trait_dict[k].inherit_value(actor_id)
                        
                        # some traits can be gene arrays that need to be:
                        # 1) stored in the trait gene dictionary
                        # 2) decoded into a value
                        if isinstance(val, np.ndarray):
                            self.population.trait_dict[k].gene_dict[new_id] = val.copy()
                            val = self.population.trait_dict[k].decode(val)
                            
                        new_traits[k] = val
              
                    if position_func is None:
                        x = np.random.rand() * self.population.xdim
                        y = np.random.rand() * self.population.ydim
                    
                    else:
                        new_radius = new_traits['radius']
                        px, py, pr, odm = self.population.df[actor_idx, ['x', 'y', 'radius',
                                                                         'offspring_dist_max']].to_numpy()[0]
                        x, y = position_func(px, py, pr, odm, new_radius)
                    
                    if x is not None:
                    
                        new_df = self.population.create_individual(new_id, x, y)
                        self.population.df.rbind(new_df, force=True)
                        self.population.id_count += 1
                        self.population.size += 1
                        
                        new_idx = self.population._get_actor_idx(new_id)
                        
                        for k in new_traits:
                            val = new_traits[k]
                            self.population.df[new_idx, str(k)] = val
                        
                        # update all new offspring's event times
                        new_params = dict(actor_id = new_id, 
                                          current_time = params['current_time'])

                        for k in self.population.event_dict:
                            if self.population.event_dict[k].is_primary:
                                self.population.event_dict[k].set_next(new_params)

                        new_events += [self.population.get_next_event(new_id)]

                        if self.triggers:
                            new_params = params.copy()
                            new_params['actor_id'] = new_id
                            new_events += self.triggers(new_params)             
                
        # update parent's next birth event, if there is one
        if self.is_primary:
            self.set_next(params)
            new_events += [self.population.get_next_event(actor_id)]
        
        return new_events  
        

class BirthAssxualDiffusionEvent(BirthAsexualEvent):
    
    def __init__(self, population, params):
        
        super().__init__(population, params)
        
        if 'allow_overlap' in params:
            self.allow_overlap = params['allow_overlap']
        else:
            self.allow_overlap = True
        
        self.index = rtree.index.Index()
        # add all x-y coordinates with unique ids
        for i,x,y in self.population.df[:, ['id', 'x', 'y']].to_tuples():
            self.index.insert(i, (x,y))
        
    def handle(self, params):
        new_events = super().handle(params, self.find_empty_space)
        return new_events

    def remove_rtree(self, params):
        actor_id = int(params['actor_id'])
        actor_idx = self.population._get_actor_idx(actor_id)
        x,y = self.population.df[actor_idx, ['x', 'y']].to_numpy()[0]
        self.index.delete(actor_id, (x,y))
        return []
        
    def insert_rtree(self, params):
        actor_id = int(params['actor_id'])
        actor_idx = self.population._get_actor_idx(actor_id)
        x,y = self.population.df[actor_idx, ['x', 'y']].to_numpy()[0]
        self.index.insert(actor_id, (x,y))
        return []
            
    def intersection_rtree(self, coordinate):
        return self.index.intersection(coordinate)

    def find_empty_space(self, px, py, pr, odm, new_radius, 
                         eps=0.00001):
        
        if self.allow_overlap:
            # pick random spot, allow overlap
            off_x = px + (np.random.rand()-0.5) * odm
            off_y = py + (np.random.rand()-0.5) * odm
            off_x = np.clip(off_x, new_radius+eps, 
                            self.population.xdim-new_radius-eps)
            off_y = np.clip(off_y, new_radius+eps, 
                            self.population.ydim-new_radius-eps)
            
        else:
            
            off_x, off_y = None, None
            # get what's nearby
            search_xmin = px - 2 * odm
            search_ymin = py - 2 * odm
            search_xmax = px + 2 * odm 
            search_ymax = py + 2 * odm
            
            neighs_idx = [self.population._get_actor_idx(n) for n in
                          self.intersection_rtree((search_xmin, search_ymin,
                                                   search_xmax, search_ymax))]
            neigh_stats = self.population.df[neighs_idx, ['x','y', 'radius']].to_numpy()
            neigh_stats_copy = neigh_stats.copy()
            
            hard_candidates = []
            vor_candidates = []
            fake_neighs = None
            
            if len(neigh_stats) == 1:
                
                max_od = (odm**2 / 2) ** (1/2)
                rands = (max_od - (pr+new_radius)) * np.random.rand(8) + (pr+new_radius)
                hard_candidates = np.array([[px - rands[0], py - rands[1], new_radius],
                                            [px - rands[2], py + rands[3], new_radius],
                                            [px + rands[4], py - rands[5], new_radius],
                                            [px + rands[6], py + rands[7], new_radius]])
                hard_candidates[:,0] = np.clip(hard_candidates[:,0], 
                                               new_radius + eps, 
                                               self.population.xdim - new_radius - eps)
                hard_candidates[:,1] = np.clip(hard_candidates[:,1], 
                                               new_radius + eps, 
                                               self.population.ydim - new_radius - eps)
                
            else:
                
                if len(neigh_stats) <= 3:
                    
                    max_od = odm
                    rands = (odm * 2 - (pr+new_radius)) * np.random.rand(8) + (pr+new_radius)
                    fake_neighs = np.array([[px - rands[0], py - rands[1], new_radius],
                                            [px - rands[2], py + rands[3], new_radius],
                                            [px + rands[4], py - rands[5], new_radius],
                                            [px + rands[6], py + rands[7], new_radius]])

                    neigh_stats = np.vstack([neigh_stats, fake_neighs])
                
                vor = Voronoi(neigh_stats[:, 0:2])
                vor_candidates = vor.vertices
                vor_candidates[:,0] = np.clip(vor_candidates[:,0], new_radius+eps, 
                                              self.population.xdim-new_radius-eps)
                vor_candidates[:,1] = np.clip(vor_candidates[:,1], new_radius+eps, 
                                              self.population.ydim-new_radius-eps)
                vor_candidates = np.append(vor_candidates, 
                                           np.repeat(new_radius, len(vor_candidates)).reshape(-1,1), axis=1)

                    
            candidates = []
            if len(hard_candidates) > 0:
                candidates.append(hard_candidates)
            if len(vor_candidates) > 0:
                candidates.append(vor_candidates)
            candidates = np.vstack(candidates)
                
            if len(candidates) > 0:
            
                # remove candidates further than odm (happens some times BECAUSE CANDIDATES FURTHER)
                close_idx = np.where(cdist([[px,py]], candidates[:,0:2])[0] <= odm)
                candidates = candidates[close_idx]

                # make sure enough room from others
                if len(candidates) > 0:
                    
                    if fake_neighs is not None:
                        neigh_stats = neigh_stats_copy
                    
                    cands_dist = cdist(neigh_stats[:, 0:2], candidates[:, 0:2]).min(0)
                    open_cands, = np.where(cands_dist > new_radius + candidates[:,2])
                    
                    if len(open_cands) > 0:
                        spot_idx = np.random.choice(open_cands, 1)[0]
                        off_x, off_y = candidates[spot_idx, 0:2]
        
        
        return off_x, off_y