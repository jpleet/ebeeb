import numpy as np
import datatable as dt
from datatable import f

from .base import Population


class Population2D(Population):
    """ Base individual-level population. Stores the size, events, traits, and
    individuals of a population.

    Example
    -------
    TO DO:
    """

    def __init__(self, name, init_size, xdim, ydim, implicit_capacity=None):
        """ Constructor for individual-level population.

        Parameters
        ----------

        name : str
            unique identifier
        init_size : int
            initial starting size of population
        xdim : int
           x-axis length of 2D environment
        ydim : int
            y-axis length of 2D environment
        implicit_capacity : default None, integer
            limit the population size to some implicit capacity to create
            logistic growth. No limit if None (default)
        """

        # set parameters
        super().__init__(name, init_size, implicit_capacity)
        # set 2D limits
        self.xdim = xdim
        self.ydim = ydim

        # create base dataframe (only id, x, y)
        self.df = self.create_population(np.arange(init_size))
        # track unique id for offspring
        self.id_count = init_size
        
        # create empty trait and event dictionarties
        self.trait_dict = {}
        self.event_dict = {}
        self.event_list = []


    def create_population(self, ids):
        """ helper function to create a population of individuals dataframe
        with unique identifiers and 2D position

        Parameters
        ----------

        ids : list unique identifiers
            unique identifiers used to create dataframe

        Returns
        -------

        df : datatable dataframe
            base population dataframe with id, x, and y columns
        """
        # create dataframe
        size = len(ids)
        df = dt.Frame(id=ids,
                      x=np.random.rand(size) * self.xdim,
                      y=np.random.rand(size) * self.ydim, 
                      status = ['active'] * size)
        return df

    def create_individual(self, new_id, x, y, status='active'):
        """ helper function to create a single individuals dataframe with a
        unique identifier and 2D position

        Parameters
        ----------

        new_id : int
            unique identifier for new individual
        x : float
            x postion to be set
        y : float
            y position to be set
        status : str
            whether individual is active or inactive

        Returns
        -------

        df : datatable dataframe
            single individual dataframe with id, x, and y columns
        """

        df = dt.Frame(id=[new_id], x=[x], y=[y], status=[status])
        return df
    
    def add_traits(self, trait_list):
        """function to add new traits to the population
        Parameters
        ----------
        trait_list : list
            list of traits with the format:
                [(Trait, param_dict), (Trait, param_dict)]
                where Trait is the class IndividualTrait and the param_dict is
                the parameters that initializes the Trait.        
        """
        
        # iteratre over trait list
        for (k, params) in trait_list:
            # initiate trait
            t = k(self, params)
            # store trait in dictionary
            self.trait_dict[f'{t}'] = t
            
    def add_events(self, event_list):
        """function to add new events to the population
        Parameters
        ----------
        event_list : list
            list of events with the format:
                [(Event1, param_dict), (Event2, param_dict)]
                where Event1 and Event 2 are different Event class and the
                param dict is the parameters to inititialize the events    
        """

        # iterate over event list
        for (k, params) in event_list:
            # initiate event
            e = k(self, params)
            # store event in dictionary
            self.event_dict[f'{e}'] = e
            
        # store primary events column
        self.event_list = [c for c in self.df.names if '_time' in c]
                    
    def update(self, lapse):
        """called to update individuals when the simulation jumps to the next
        event. currently, updates position of moving indviduals

        Parameters
        ----------

        lapse : float
            differece between previous event time and current event time
        """
        # if a population has a velocity component, means they move and need updating
        if 'vel_x' in self.df.names:
            # move
            self.df[:, 'x'] = self.df[:, f.x + f.vel_x * lapse]
            self.df[:, 'y'] = self.df[:, f.y + f.vel_y * lapse]

    def get_next_event(self, actor_id):
        """ function to get the next event for a given individual. this next
        event will be added to an event heap and performed in time

        Parameters
        ----------

        actor_id : int
            unique identifier of individual to find next event time

        Returns
        -------

        event : tuple
            follows the format (time, event, params)

        """
        
        # TODO: remove actor_idx check, get row if not empty check
        
        # get actor row number from population dataset
        actor_idx = self._get_actor_idx(actor_id)

        if actor_idx is not None:

            # get individuals event times
            row = self.df[actor_idx, self.event_list]
            # find name of column of most immediate event
            event_time_name = self.event_list[np.nanargmin(row)]
            # get event name for reference
            event_name = event_time_name.rsplit('_', maxsplit=1)[0]
            # calculate minumum event time
            event_time = np.nanmin(row)
            # create a dictionary of extra event parameters
            params = dict(current_time=event_time,
                          actor_id=actor_id)
            # check if there are extra parameters for event
            event_extra = event_time_name.replace('_time', '_extra')
            #if event_other_id_name in self.df.names:
            if event_extra in self.df.names:
                # get extra parameter info
                extra = self.df[actor_idx, event_extra]
                # add info to dictionary
                params['extra'] = extra
            # create event hash, makes comparison in heap easier
            event_hash = hash(f'{event_time}_{event_name}_' + 
                              '_'.join(str(params[k]) for k in params))
            # return next new event for this individual
            return (event_time, event_hash, self.event_dict[event_name], params)

        else:
            # otherwise, return no new events
            return []
        
    def _get_actor_idx(self, actor_id):
        actor_idx, = np.where(self.df.to_numpy(
            column=self.df.colindex('id')) == actor_id)
        if len(actor_idx) > 0:
            return int(actor_idx[0])
        else:
            return None

    def __repr__(self):
        return self.name
