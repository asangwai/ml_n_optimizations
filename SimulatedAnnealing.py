import random
import math
import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go


# Formatting variables
HEADER = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
SPACER = '===================================================================='
FOOTER = '********************************************************************'

class SimulatedAnnealing(object):
    """ Class for performing SimulatedAnnealing optimization algorithm."""

    def __init__(self, start_temp, stop_temp, alpha, num_sims):
        """
        Constructor Method.
        """
        self.start_temp = start_temp
        self.stop_temp = stop_temp
        self.alpha = alpha
        self.num_sims = num_sims

    def _random_int(self, lower_bound, upper_bound, step):
        return random.randrange(lower_bound, upper_bound, step)

    def _random_float(self, lower_bound, upper_bound, sig_dig):
        return round(random.uniform(lower_bound, upper_bound), sig_dig)

    def _get_neighbor_int(self, current_value, distance, step, lower_bound, upper_bound):
        """ Return a neighboring random integer value from some current value.
        :param distance: defines neighborhood as current value +- distance
        :param step: step size range within neighborhood
        :param lower_bound: hard cutoff lower bound that overrides distances lower than this threshold
        :param upper_bound: hard cutoff upper bound that overrides distances higher than this threshold
        :return: random value within neighborhood of current input value
        
        Method generates a random integer value within a specified neighborhood 
        around a given current value. It ensures that the generated value respects
        the defined lower and upper bounds and increments by the specified step size. 
        This method is useful for generating random integer values within a controlled range, 
        which can be applied in various scenarios such as optimization algorithms or simulations.       
        """
        upper_neighbor = current_value + distance
        if upper_neighbor > upper_bound:
            upper_neighbor = upper_bound
        else:
            upper_neighbor = upper_neighbor

        lower_neighbor = current_value - distance
        if lower_neighbor < lower_bound:
            lower_neighbor = lower_bound
        else:
            lower_neighbor = lower_neighbor
        neighbor = random.randrange(lower_neighbor, upper_neighbor, step)
        return neighbor

    def _get_neighbor_float(self, current_value, distance, lower_bound, upper_bound, sig_dig):
        """ Return a neighboring random float from some current value.
        :param distance: defines neighborhood as current value +- distance
        :param sig_dig: number of significant digits in result
        :param lower_bound: hard cutoff lower bound that overrides distances lower than this threshold
        :param upper_bound: hard cutoff upper bound that overrides distances higher than this threshold
        :return: random value within neighborhood of current input value
        
        Method generates a random float value within a specified neighborhood 
        around a given current value. It ensures that the generated value respects 
        the defined lower and upper bounds and has the specified number of significant digits. 
        This method is useful for generating random values within a controlled range, 
        which can be applied in various scenarios such as optimization algorithms or simulations.
        """
        upper_neighbor = current_value + distance
        if upper_neighbor > upper_bound:
            upper_neighbor = upper_bound
        else:
            upper_neighbor = upper_neighbor

        lower_neighbor = current_value - distance
        if lower_neighbor < lower_bound:
            lower_neighbor = lower_bound
        else:
            lower_neighbor = lower_neighbor
        neighbor = random.uniform(lower_neighbor, upper_neighbor)
        neighbor = round(neighbor, sig_dig)
        return neighbor

    def _roll_dice(self, delta, t_new):
        """
        Method to calculate the probability of acceptance
        :param delta: difference between the new cost and the old cost
        :param t_new: the current temperature of the annealing algorithm
        :return: Accept the new design? (True/False)

        Method calculates the probability of accepting a new solution in a 
        simulated annealing algorithm based on the difference in cost and 
        the current temperature. It generates a random number to simulate a 
        "dice roll" and compares it with the calculated probability to make 
        the acceptance decision. The method returns a boolean value indicating 
        whether the new solution is accepted or rejected. This probabilistic 
        acceptance mechanism allows the algorithm to explore the solution space 
        more effectively and avoid getting stuck in local optima.
        """
        pa = math.exp(delta / t_new)
        print('Probability of Acceptance = ' + str(pa) + ' %')
        r = random.random()
        print('Rolling the dice... = ' + str(r) + '%')
        decision = pa - r
        if decision >= 0:
            d = True
        else:
            d = False
        return d

    def run_annealing(self, xy_function, x_ub, x_lb, y_ub, y_lb, x_neighbor_distance, y_neighbor_distance):
        """ Run the annealing algorithm.
        INPUTS:
            xy_function ::      (function) 2-D function of x and y
            x_ub        ::      (float) Upper X-boundary of search space
            x_lb        ::      (float) Lower X-boundary of search space
            y_ub        ::      (float) Upper Y-boundary of search space
            y_lb        ::      (float) Lower Y-boundary of search space
            x_neighbor_distance ::  (float) Distance between neighbors for X parameter
            y_neighbor_distance ::  (float) Distance between neighbors for Y parameter
        RETURNS:
            df          ::      (pandas dataframe) The search path of the algorithm in x, y, z (cost) coordinates
        
        Method implements the simulated annealing algorithm to find an approximate 
        solution to an optimization problem. It initializes a random solution, 
        iteratively generates new solutions, and decides whether to accept them 
        based on a probabilistic acceptance criterion. The method returns a DataFrame 
        containing the search path of the algorithm in x, y, and cost coordinates.
        """
        print('\n' + HEADER)
        print("Testing SimulatedAnnealing Module")
        print(HEADER + '\n')

        # 1 - 2. Initialize random solution and calculate cost
        x_list = []
        y_list = []
        c_list = []
        x_old = self._random_float(x_lb, x_ub, 12)
        y_old = self._random_float(y_lb, y_ub, 12)
        c_old = xy_function(x_old, y_old)
        x_list.append(x_old)
        y_list.append(y_old)
        c_list.append(c_old)
        print("Initial Cost = " + str(c_old))

        # 3. Generate a new neighboring solution
        t_new = self.start_temp
        while t_new > self.stop_temp:
            print(SPACER)
            print('\n' + 'Running ' + str(self.num_sims) + ' simulations @ Temperature = ' + str(t_new) + '\n')
            print(SPACER)
            for i in range(self.num_sims):  # TODO: Not very Pythonic...
                print('\n' + 'Simulation Run #:   ' + str(i + 1))
                x_new = self._get_neighbor_float(current_value=x_old, distance=x_neighbor_distance, lower_bound=x_lb,
                                                 upper_bound=x_ub,
                                                 sig_dig=12)
                y_new = self._get_neighbor_float(current_value=y_old, distance=y_neighbor_distance, lower_bound=y_lb,
                                                 upper_bound=y_ub,
                                                 sig_dig=12)
                c_new = xy_function(x_new, y_new)

                # Probability of acceptance
                delta = c_new - c_old
                print('delta = ' + str(delta))
                if delta >= 0:
                    print('The new solution is better - moving to it!')
                    x_old = x_new
                    y_old = y_new
                    c_old = c_new
                else:
                    print('The new solution is worse - rolling the dice to decide whether to accept it anyway...')
                    decision = self._roll_dice(delta=delta, t_new=t_new)
                    if decision == True:
                        print('The gods have spoken: selecting the new solution even though it is worse!')
                        x_old = x_new
                        y_old = y_new
                        c_old = c_new
                    else:
                        print(
                            'The gods have smiled on us: rejecting this worse solution and sticking with the old one!')
                        x_old = x_old
                        y_old = y_old
                        c_old = c_old

                print('x_old = ' + str(x_old))
                print('y_old = ' + str(y_old))
                print('c_old = ' + str(c_old))
                x_list.append(x_old)
                y_list.append(y_old)
                c_list.append(c_old)
                print("New Cost = " + str(c_new))
            t_new = t_new * self.alpha
            print(FOOTER)
            print('New Temperature = ' + str(t_new))
            print(FOOTER)

        dfx = pd.DataFrame(x_list)
        dfx.columns = ['x']
        dfy = pd.DataFrame(y_list)
        dfy.columns = ['y']
        dfc = pd.DataFrame(c_list)
        dfc.columns = ['c']
        results = pd.concat([dfx, dfy, dfc], axis=1)
        return results


def goldstein_price_function(x, y):
    """ Method for testing algorithm using the Goldstein-Price Function as the problem to optimize.
    See wiki page for optimization functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization

    A mathematical function used to test optimization algorithms. 
    It calculates the cost based on the Goldstein-Price function formula, 
    which involves complex polynomial expressions of the input variables x and y. 
    The function returns the calculated cost, which can be used to evaluate the 
    performance of optimization algorithms in finding the global minimum of the function.
    """
    cost = (1 + ((x + y + 1) ** 2) * (19 - 14 * x + 3 * (x ** 2) - 14 * y + 6 * x * y + 3 * (y ** 2))) * (
            30 + ((2 * x - 3 * y) ** 2) * (18 - 32 * x + 12 * (x ** 2) + 48 * y - 36 * x * y + 27 * (y ** 2)))
    return cost

def ackleys_function(x, y):
    """ Method for testing algorithm using the Goldstein-Price Function as the problem to optimize.
    See wiki page for optimization functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    
    A mathematical function used to test optimization algorithms. It calculates 
    the cost based on Ackley's function formula, which involves exponential and 
    trigonometric expressions of the input variables x and y. The function 
    returns the calculated cost, which can be used to evaluate the performance 
    of optimization algorithms in finding the global minimum of the function. 
    Note that the docstring should be corrected to reference Ackley's function 
    instead of the Goldstein-Price function.
    """
    cost = -20 * math.exp((-0.2 * (math.sqrt(0.5 * (x ** 2 + y ** 2))))) - math.exp(
        (0.5 * ((math.cos(2 * math.pi * x)) + math.cos(2 * math.pi * y)))) + math.exp(1) + 20
    return cost


def plot_function_points(xy_function):
    """
    Generates a DataFrame of x, y, and z (cost) coordinates by evaluating a given 
    2-D function over a specified range of x and y values. The resulting DataFrame 
    can be used to visualize the function's behavior and analyze its properties. 
    The code could be made more Pythonic by using more efficient looping or vectorized operations.
    """
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    x_list = []
    y_list = []
    cost_list = []
    for j in range(len(x)):  # TODO: Again, lots of non-pythonic stuff
        x1 = x[j]
        for p in range(len(y)):  # TODO: Again, lots of non-pythonic stuff
            y1 = y[p]
            cost = xy_function(x1, y1)
            x_list.append(x1)
            y_list.append(y1)
            cost_list.append(cost)
    dfx = pd.DataFrame(x_list)
    dfx.columns = ['x']
    dfy = pd.DataFrame(y_list)
    dfy.columns = ['y']
    dfz = pd.DataFrame(cost_list)
    dfz.columns = ['z']
    function_data = pd.concat([dfx, dfy, dfz], axis=1)
    return function_data


if __name__ == "__main__":
    sa = SimulatedAnnealing(start_temp=1000, stop_temp=1, alpha=0.95, num_sims=250)
    results = sa.run_annealing(xy_function=goldstein_price_function, x_ub=2.0001, x_lb=-2.0001, y_ub=2.0001,
                               y_lb=-2.0001, x_neighbor_distance=0.1000000, y_neighbor_distance=0.1000000)
    function_data = plot_function_points(goldstein_price_function)
    trace1 = go.Scatter3d(
        x=results['x'],
        y=results['y'],
        z=results['c'],
        mode='markers',
        name='Optimization Path',
        marker=dict(
            size=9,
            color=results['c'],
            colorscale='Rainbow',
            opacity=0.95
        )
    )
    trace2 = go.Scatter3d(
        x=function_data['x'],
        y=function_data['y'],
        z=function_data['z'],
        mode='markers',
        name='Function Surface',
        marker=dict(
            size=6,
            color=function_data['z'],
            colorscale='Blues',
            opacity=0.65
        )
    )
    data = [trace2, trace1]
    layout = go.Layout(
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True,
        title=' Simulated Annealing Algorithm ',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        font=dict(size=15)
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3d-scatter-colorscale')
