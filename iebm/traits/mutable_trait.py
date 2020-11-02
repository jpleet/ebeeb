import numpy as np
from .base import Trait

class MutableHaploidTrait(Trait):
    
    def __init__(self, population, params):
        """ Constructor for a mutable haploid individual-level trait.

        Parameters
        ----------

        population : Population
            reference to individual-level population class

        params : dict
            dictionary with trait parameters. The parameters determine the
            type of trait. If key is omitted, considered None
            Primary mutatable trait keys:
                name : unique trait identifier (str)
                values : list of starting initial values (list)
                frequencies : list of frequencies of each value (list)
                min_value: smallest value trait can evolve towards (float)
                max_value: largest value trait can evolve towards (float)
                array_size : length of genetic boolean array (int)
                mutate_prob : probability to reverse boolean entry (float)
                

        """
        
        super().__init__(population, params)
        
        self.min_value = params['min_value']
        self.mutate_prob = params['mutate_prob']
        self.gene_value = (params['max_value'] - params['min_value']) / params['array_size']
        
        genes_list = []
        
        # create first gene array. all subsequent gene arrays are based on this one.
        genes_base = np.zeros(params['array_size'], dtype=np.bool)
        base_count = int((params['values'][0] - self.min_value) / self.gene_value)
        base_indices = np.random.choice(np.arange(params['array_size']), base_count, replace=False)
        genes_base[base_indices] = True
        
        # store in list 
        genes_list.append(genes_base)
        
        # create similar traits for the remaining listed values
        for val in params['values'][1:]:
            genes_new = genes_base.copy()
            new_count = int((val - self.min_value) / self.gene_value)
            change = new_count - base_count
            
            if change > 0:
                # add more True to array
                indices_to_change = np.setdiff1d(np.arange(params['array_size']), base_indices)
                indices_to_change = np.random.choice(indices_to_change, change, replace=False)
                genes_new[indices_to_change] = True
            
            if change < 0:
                # remove True from array
                indices_to_change = np.intersect1d(np.arange(params['array_size']), base_indices)
                indices_to_change = np.random.choice(indices_to_change, abs(change), replace=False)
                genes_new[indices_to_change] = False
                
            genes_list.append(genes_new)
        
        # get list of population ids and assign each a trait based on frequencies
        pop_ids = self.population.df[:, 'id'].to_numpy(column=0)
        pop_size = len(pop_ids)
        
        for gene, freq in zip(genes_list, params['frequencies']):
            count = int(freq * pop_size)
            select_ids = np.random.choice(pop_ids, count, replace=False)
            pop_ids = np.setdiff1d(pop_ids, select_ids)
            
            for select_id in select_ids:
                self.gene_dict[select_id] = gene.copy()
                
        # if any remaining pop ids, set to base gene
        if len(pop_ids) > 0:
            for p in pop_ids:
                self.gene_dict[p] = genes_base.copy()
                
        # iterate over pop ids again and assign value to dataframe
        # no initial mutation to start
        values = []
        pop_ids = self.population.df[:, 'id'].to_numpy(column=0)
        for pop_id in pop_ids:
            val = self.min_value + self.gene_dict[pop_id].sum() * self.gene_value
            values.append(val)

        self.population.df[str(self)] = np.array(values)
        
    def get_value(self, actor_id):
        """function to get and return trait value for a specific actor inidividual in
        a population.

        Parameters
        ----------

        actor_id : int
            identifier interger to a specific acting individual

        """
        
        actor_idx = self.population._get_actor_idx(actor_id)
        if actor_idx is not None:
            # return value at row number and trait column name
            return self.population.df[actor_idx, str(self)]
        else:
            return None
    
    def inherit_value(self, parent_id):
        """function to pass trait values from a parent to an offspring,
        includes a mutation possibility

        Parameters
        ----------

        parent_id : int
            identifier interger for the parent
        """
        
        parent_genes = self.gene_dict[parent_id].copy()
        off_genes = self.mutate(parent_genes)

        return off_genes
        
    
    def mutate(self, genes):
        """Given a value and a mutable trait, possibly change value. Otherwise,
        the passed value is returned as is.

        Parameters
        ----------

        genes : boolean numpy array
            list of genes to possibly mutate
        """
        
        mut_spots = np.random.rand(len(genes)) < self.mutate_prob
        
        if len(mut_spots) > 0:
            genes[mut_spots] = ~genes[mut_spots]
            
        return genes
    
    def decode(self, genes):
        """ Simple function to convert the haploid gene array into
        a single value
        """
        
        return self.min_value + genes.sum() * self.gene_value
        