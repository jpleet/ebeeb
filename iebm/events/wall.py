import numpy as np
import datatable as dt
from datatable import f, math

from .base import Event


class WallEvent(Event):
    """ Event to handle the wall collision of moving individuals. Moving
    individuals probably should have environmental boundaries

    """

    def __init__(self, population, params):
        """Construct Wall Event with given hard-coded name

        Parameters
        ----------

        population : class Population
            the Population class that performs the actions

        params : dict
            *must contain:
            - 'bounce' key and value either ['random', 'reflective']
            - 'current_time'
            - 'is_primary' : True|False
        """

        if 'triggers' in params:
            triggers = params['triggers']
        else:
            triggers = None
        
        # initialize base event
        super().__init__(population, params['name'], 
                         params['is_primary'], triggers)
        
        # remember bounce type
        self.bounce = params['bounce']
        # (always primary) but check
        if self.is_primary:
            # check if velocity column exists, if not create
            if 'angle' not in self.population.df.names:
                self.population.df['angle'] = np.random.rand(self.population.df.shape[0]) * 2 * np.pi
                self.population.df['vel_x'] = self.population.df[:, math.cos(f.angle) * f.velocity]
                self.population.df['vel_y'] = self.population.df[:, math.sin(f.angle) * f.velocity]
                
            # create wall dataframe based on each individual radius, position, angle, and speed
            wall_df = dt.Frame(
                wall_x0_time=self.population.df[:, ((f.radius - f.x) / f.vel_x)],
                wall_x1_time=self.population.df[:, ((self.population.xdim - f.radius - f.x) / f.vel_x)],
                wall_y0_time=self.population.df[:, (f.radius - f.y) / f.vel_y],
                wall_y1_time=self.population.df[:, ((self.population.ydim - f.radius - f.y) / f.vel_y)])
            # make sure times are forward
            wall_df[f.wall_x0_time<0, f.wall_x0_time] = np.nan
            wall_df[f.wall_x1_time<0, f.wall_x1_time] = np.nan
            wall_df[f.wall_y0_time<0, f.wall_y0_time] = np.nan
            wall_df[f.wall_y1_time<0, f.wall_y1_time] = np.nan
            # set time to most immediate wall event and adjust to simulation time
            self.population.df[:, f'{self}_time'] = (wall_df.to_numpy().min(1) +
                                                     params['current_time'])

    def set_next(self, params, actor_idx=None):
        
        if actor_idx is None:
            actor_id = params['actor_id']
            actor_idx = self.population._get_actor_idx(actor_id)

        if actor_idx is not None:

            x, y, r = self.population.df[actor_idx, ['x', 'y', 'radius']].to_numpy()[0]
            
            # change angle
            # maybe check if close to wall before changing angle
            if self.bounce == 'reflective':
                actor_x = np.array([x])
                actor_y = np.array([y])
                wall = np.argmin([np.min(np.abs(actor_x - [0, self.xdim])),
                                  np.min(np.abs(actor_y - [0, self.ydim]))])
                if wall == 0:
                    # hit vertical wall, add pi to reverse angle
                    self.population.df[actor_idx, 'angle'] = (
                        np.pi - self.population.df[actor_idx, 'angle'])
                else:
                    # hit horizontal wall, reverse angle
                    self.population.df[actor_idx, 'angle'] = (
                        -self.population.df[actor_idx, 'angle'])
            else:
                # default random angle change
                self.population.df[actor_idx, 'angle'] = (
                    np.random.rand() * 2 * np.pi)

            # check if actor on wall (need to move a bit)
            if x <= r + 0.1*r:
                self.population.df[actor_idx, 'x'] = r * 2
            if x >= self.population.xdim - (r + 0.1*r):
                self.population.df[actor_idx, 'x'] = (
                    self.population.xdim - r * 2)
            if y <= r + 0.1*r:
                self.population.df[actor_idx, 'y'] = r * 2
            if y >= self.population.ydim - (r + 0.1*r):
                self.population.df[actor_idx, 'y'] = (
                    self.population.ydim - r * 2)

            # update velocity components and new wall event times
            self.population.df[actor_idx, 'vel_x'] = self.population.df[
                actor_idx, math.cos(f.angle) * f.velocity]
            self.population.df[actor_idx, 'vel_y'] = self.population.df[
                actor_idx, math.sin(f.angle) * f.velocity]

            # update new wall times
            wall_x0 = self.population.df[
                actor_idx, (f.radius - f.x) / f.vel_x][0,0]
            wall_x1 = self.population.df[
                actor_idx, ((self.population.xdim - f.radius - f.x)
                            / f.vel_x)][0,0]
            wall_y0 = self.population.df[
                actor_idx, (f.radius - f.y) / f.vel_y][0,0]
            wall_y1 = self.population.df[
                actor_idx, ((self.population.ydim - f.radius - f.y)
                            / f.vel_y)][0,0]
            wall_times = np.array([wall_x0, wall_x1, wall_y0, wall_y1], 
                                  dtype=np.float)

            wall_times[wall_times<=0] = np.nan
            wall_time = np.nanmin(wall_times) + params['current_time']
 
            self.population.df[actor_idx, f'{self}_time'] = wall_time

    def handle(self, params):
        
        new_events = []
        
        actor_id = params['actor_id']
        actor_idx = self.population._get_actor_idx(actor_id)

        if actor_idx is not None:
            
            status = self.population.df[actor_idx, 'status']
            
            if status == 'active':
        
                self.set_next(params, actor_idx)

                if self.triggers:
                    new_events += self.triggers(params)

                if self.is_primary:
                    new_events += [self.population.get_next_event(params['actor_id'])]
        
        return new_events