import numpy as np
import heapq
from tqdm import tqdm

#import warnings
#warnings.filterwarnings("error")
#import traceback

class Simulation():
    """ Base individual-level model. Creates a simulation with a given
    dictionary of populations. The simulation is started by a run(runtime)
    function, the individuals perform their events, and the simulation
    ends once a runtime is reached. Values can be stored along the way and
    analyzed afterwards.

    Example
    -------
    """
    def __init__(self, population_dict, continue_threshold=3):
        """ Constructor for individual-level model.

        Parameters:
        -----------

        population_dict : dictionary
            Contains population string identifiers as keys and the
            individual-level population class reference as values.

        min_thresh : int
            Consider a population below this threshold to be extinct.
            Will stop the simulation run.
        """

        # set seed and initial parameters
        np.random.seed()
        self.time = 0
        self.continue_threshold = continue_threshold
        # initialize event heap
        self.event_heap = []
        heapq.heapify(self.event_heap)

        # store population dict
        self.population_dict = population_dict
        # iterate all population and add individual events to heap
        for p in population_dict:
            # get individual unique id
            col_id = population_dict[p].df.colindex('id')
            # iterate over all individuals
            for i in population_dict[p].df.to_numpy(column=col_id):
                # add individual event to event heap
                heapq.heappush(self.event_heap,
                               population_dict[p].get_next_event(i))

        # store tracked traits
        self.trait_tracks = {}
        self.trait_history = {}
        for p in population_dict:
            for k in population_dict[p].trait_dict:
                trt = population_dict[p].trait_dict[k]
                if trt.track:
                    name = str(trt)
                    self.trait_tracks[(p, name)] = trt
                    self.trait_history[(p, name)] = []
        
        self.population_history = dict([(p, [population_dict[p].size])
                                        for p in population_dict])
        self.time_history = [self.time]
        

    def run(self, runtime, progress_bar=True):
        """ Function to start (or continue) a model simulation."""
        
        prev_event_hash = None
        if progress_bar:
            pbar = tqdm(total=round(runtime, 4), 
                        bar_format=("{l_bar}{bar}| {n:.4f}/{total_fmt} " + 
                                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"))
        
        # continue running up until the runtime
        while self.time < runtime:

            # get next event from heap
            #try:
            next_event = heapq.heappop(self.event_heap)
            #except Exception as e:
            #    print(e)
            
            event_time, event_hash, event, event_params = next_event
            
            if event_hash == prev_event_hash:
                continue
            
            lapse = event_time - self.time
            
            if progress_bar:
                pbar.update(lapse)
                
            # go through each population and update
            for p in self.population_dict:
                self.population_dict[p].update(lapse)
    
            # update current time to event time
            self.time = event_time
            
            # handle event and return new events
            new_events = event.handle(event_params)

            for new_event in new_events:
                # confirm there is an event tuple
                if len(new_event) > 0:
                    # make sure all times in the future
                    if new_event[0] > self.time:
                        # add event to event heap
                        try:
                            heapq.heappush(self.event_heap, new_event)  
                        except:
                            print(f'NEW EVENT : {new_event}')
                            traceback.print_exc()
                            continue
            
            # store event_hash to make sure not repeating
            prev_event_hash = event_hash
            
            # store results, check if simulation should end
            continue_run = self.update_history()
                
            if not continue_run:
                break
            
        if progress_bar:
            pbar.close()

    def update_history(self):
        """Helper function to store population sizes and determine if a
        simulation should stop when a population is extinct"""
        continue_run = True
        
        for p in self.population_dict:
            s = self.population_dict[p].size
            if s <= self.continue_threshold:
                continue_run = False
            self.population_history[p].append(s)
            
        for (p,t) in self.trait_tracks:
            self.trait_history[(p,t)].append(self.trait_tracks[(p,t)].track_values())
            
        self.time_history.append(self.time)
        
        return continue_run

    def get_results(self):
        res = {}
        res['time'] = self.time_history
        res['size'] = {}
        for p in self.population_dict:
            res['size'][str(p)] = self.population_history[p]
        res['trait'] = {}
        for (p,t) in self.trait_tracks:
            res['trait'][(str(p), str(t))] = self.trait_history[(p,t)]
        return res
        