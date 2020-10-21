import numpy as np
from abc import ABC, abstractmethod


class Event(ABC):
    """ Base class for individual-level events. All sublasses
    must write their own handle() and next_event() functions.

    Example
    -------
    TO DO
    """

    def __init__(self, population, name, is_primary, triggers=None):
        """ Constructor for all events.

        Parameters
        ----------

        population : class Population:
            reference to Population class that performs this event

        name : str
            unique name to identify event

        is_primary : bool
            whether a primary event and should be stored in dataframe
        """
        # reseed, in case of parallel runs
        np.random.seed()
        # set values
        self.name = name
        self.population = population
        self.is_primary = is_primary
        self.triggers = triggers

    @abstractmethod
    #def set_next(self, actor_id, current_time):
    def set_next(self, params):
        """Calculate and set the next event time"""
        pass

    @abstractmethod
    def handle(self, params):
        """How the event is performed/handled by the population"""
        pass

    def __repr__(self):
        """Returns event name for identification"""
        return self.name