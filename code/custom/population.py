
#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod, abstractproperty
from functools import partial
from deap import base, gp
import copy

#------------------------------------------------------------------------------
# AbstractBaseIndividual
#------------------------------------------------------------------------------

class AbstractBaseIndividual(gp.PrimitiveTree):

    '''
    Shell for Individual class, extends gp.PrimitiveTree

    Required abstract methods:
        - mutate
        - mate
        - evaluate

    Required abstract properties:
        - fitness
    '''

    # Define this class as abstract
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):

        '''
        Calls __init__ for gp.PrimitiveTree
        '''

        self.fitness = self.fitness_class()

        super(AbstractBaseIndividual, self).__init__(*args, **kwargs)

    @abstractproperty
    def fitness_class(self):

        '''
        Class definition that subclasses deap.base.Fitness
        '''

        pass

    @abstractmethod
    def mutate(self):

        '''
        Perform a mutation operation on individual
        '''

        pass

    @abstractmethod
    def mate(self, individual):

        '''
        Mate with a second individual

        @param individual: individual to mate with
        '''

        pass

    @abstractmethod
    def evaluate(self):

        '''
        Return fitness value(s) of individual
        '''

        pass

#------------------------------------------------------------------------------
# AbstractBasePopulation
#------------------------------------------------------------------------------

class AbstractBasePopulation(list):

    '''
    Shell for Population class, extends list

    Required abstract methods:
        - select
        - evaluate
    '''

    # Define this class as abstract
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):

        '''
        Call __init__ for list 
        '''

        super(AbstractBasePopulation, self).__init__(*args, **kwargs)

    def __add__(self, rhs):

        '''
        If adding populations, return a population where the list of
        individuals is the sum of the component populations' lists of
        individuals
        '''

        if not isinstance(rhs, type(self)):

            msg = 'Can\'t add classes of type \'{}\' and \'{}\'' \
                .format(type(self).__name__, type(rhs).__name__)
            raise NotImplementedError(msg)

        return type(self)(self[:] + rhs[:])

    def __radd__(self, lhs):

        '''
        If adding populations, return a population where the list of
        individuals is the sum of the component populations' lists of
        individuals
        '''

        if not isinstance(lhs, type(self)):

            msg = 'Can\'t add classes of type \'{}\' and \'{}\'' \
                .format(type(lhs).__name__, type(self).__name__)
            raise NotImplementedError(msg)

        return type(self)(lhs[:] + self[:])

    def union(self, other):
        pop = copy.deepcopy(self)
        unique_set = set()
        for ind in pop:
            key = str(ind)
            if key not in unique_set:
                unique_set.add(key)
        for ind in other:
            key = str(ind)
            if key not in unique_set:
                unique_set.add(key)
                pop.append(ind)
        return pop

    @abstractmethod
    def select(self, n):

        '''
        Return n individuals selected from the population
        '''

        pass
    
    
    @abstractmethod
    def evaluate(self):

        '''
        Update the fitness of all individuals in the population

        Return number of .apply operations needed (params population and afs_tool are modified)
    
        '''

        pass

#------------------------------------------------------------------------------
# PopulationBuilder
#------------------------------------------------------------------------------

class PopulationBuilder(object):

    '''
    Helper class for building population objects as lists of inidividuals
    Adds properties and methods to extend abstract population and individual
    objects. Also contains properties for functions and parameters for
    creating gp.PrimitiveTree's for individual objects
    '''

    def __init__(self, population_class=AbstractBasePopulation,
        individual_class=AbstractBaseIndividual):

        '''
        @param population_class: Superclass for population object
        @param individual_class: Superclass for individual objects
        '''

        self.population_class = population_class
        self.individual_class = individual_class

        # dict's of the form {name: function} for population and individual
        # methods

        self.population_functions = {}
        self.individual_functions = {}

    @property
    def pset(self):

        '''
        gp.PrimitiveSetTyped or gp.PrimitiveSet used for creating
        gp.PrimitiveTree's for individual objects
        '''

        return self._pset

    @pset.setter
    def pset(self, value):

        self._pset = value

    @property
    def min_depth(self):

        '''
        Minimum depth for creating gp.PrimitiveTree's for individual objects
        '''

        return self._min_depth

    @min_depth.setter
    def min_depth(self, value):

        self._min_depth = value

    @property
    def max_depth(self):

        '''
        Maximum depth for creating gp.PrimitiveTree's for individual objects
        '''

        return self._max_depth

    @max_depth.setter
    def max_depth(self, value):

        self._max_depth = value

    @property
    def gen(self):

        '''
        Method for creating gp.PrimitiveTree's for individual objects
        '''

        return self._gen

    @gen.setter
    def gen(self, value):

        self._gen = value

    @property
    def weight(self):

        '''
        Weight for individual fitness: 1.0 or -1.0
        '''

        return self._weight

    @weight.setter
    def weight(self, value):

        self._weight = value

    def add_population_function(self, key, function, *args, **kwargs):

        '''
        Add instance method to class for population object

        @param name: name of function for population object
        @param function: function to be bound to population object
        @param *args: *args for function 
        @param *kwargs: *kwargs for function 
        '''

        self.population_functions[key] = \
            partial(function, *args, **kwargs)

    def add_individual_function(self, key, function, *args, **kwargs):

        '''
        Add instance method to class for individual objects

        @param name: name of function for individual object
        @param function: function to be bound to individual object
        @param *args: *args for function 
        @param *kwargs: *kwargs for function 
        '''

        self.individual_functions[key] = \
            partial(function, *args, **kwargs)

    def decorate_individual_function(self, key, decorator, *args, **kwargs):

        '''
        Replace function under key with decorated version of function with same
        args and kwargs

        @param name: name of individual object function to decorate
        @param decorator: decorator for function
        @param *args: *args for decorator 
        @param *kwargs: *kwargs for decorator 
        '''

        f = self.individual_functions[key]
        func = decorator(*args, **kwargs)(f.func)
        self.individual_functions[key] = partial(func, *f.args, **f.keywords)

    def build(self, n):

        '''
        Return population object subclassing self.population_class containing n
        individual objects subclassing self.individual_class

        @param n: number of individuals in population
        '''

        # lambda to convert partial function into bindable lambda function

        func = lambda p: lambda *args, **kwargs: p(*args, **kwargs)

        # convert dict of individual partial functions into dict of bindable
        # lambda functions to extend self.individual_class

        attr = {k: func(f) for k, f in self.individual_functions.items()}

        # type(name, (subclass, ), dict) creates dynamic subclass of 'subclass'
        # with name 'name' and members from 'dict' (obj.key = value)

        # subclass base.Fitness with correct weight values

        weights = {'weights': (self.weight,)}
        Fitness = type('Fitness', (base.Fitness, ), weights)

        # set individual.fitness_class to the subclass definition
        
        attr['fitness_class'] = Fitness

        # subclass self.individual_class with fitness property and
        # bindable versions of self.individual_functions as members

        Individual = type('Individual', (self.individual_class, ), attr)

        # set the subclass module to the current module

        Individual.__module__ = __name__

        # create list of individual objects using subclass definition,
        # tree generator function, and properties of the builder object

        gen_args = \
        {
            'pset': self.pset
            , 'min_': self.min_depth
            , 'max_': self.max_depth
        }

        individuals = [Individual(self.gen(**gen_args)) for _ in range(n)]

        # convert dict of population partial functions into dict of bindable
        # lambda functions to extend self.population_class

        attr = {k: func(f) for k, f in self.population_functions.items()}        

        # subclass self.population_class with bindable versions of
        # self.population_functions as members

        Population = type('Population', (self.population_class, ), attr)

        # set the subclass module to the current module

        Population.__module__ = __name__
        # print(__name__) #TODO: adding this line fixed the pickle issue?!?! Find out why

        # register the dynamic subclasses to the global environment, allowing
        # objects of these types to be pickled from outside the module
        # this is how deap allows for pickling of dynamic subclasses but there
        # might be a better way

        globals()[Fitness.__name__] = Fitness
        globals()[Individual.__name__] = Individual
        globals()[Population.__name__] = Population

        # return population object using subclass definition and list of
        # individuals

        return Population(individuals)