import numpy as np
from perlin_noise import PerlinNoise


############################################################################################
############################################################################################


'''
# GLOBAL PARAMETERS

y_screen -> dimensions of simulation window
x_screen -> dimensions of simulation window
                
dt -> Value of one time step
init_guess -> Initial Guess for PSO
max_iter -> Max no. of iterations for simulation
range_var -> Range of variables randomly chosen around init_guess
Q_ETC -> diff_input.T @ Q_ETC @ diff_input: Weights for determining ETC triggers
Q_err -> err.T @ Q_err @ err: Weights for determining current linear & angular deviations

num_bots -> Number of particles in PSO
learning_rate -> Cognitive & Social Learning Rates
num_steps -> Number of max iterations for PSO training

'''

y_screen = 800
x_screen = 1300
                  
etc_pso_params = {"dt": 0.01,
                  "max_iter": 1e3,
                  "Q_err": np.diag([0.01, 50.]),
                  "Q_ETC": np.diag([0.01, 5.]),
                  "init_guess": np.array([10., -50., 1., 10., 1., 25., 300.]),
                  "range_var": np.array([50., 50., 25., 25., 25., 25., 250.])
                  }


game_params = {"x_screen": x_screen,
              "y_screen": y_screen,
              "bot_radius": (x_screen + y_screen)//70,
              "Q": etc_pso_params["Q_ETC"],
              "dt": etc_pso_params["dt"]
              }


pso_params = {"num_bots": 50,
              "num_steps": 50,
              "learning_rate": np.array([0.05, 0.1])}


############################################################################################
############################################################################################


def get_coord(octaves=5, seed=5):
    '''
    For generating closely related yet
    random coordinates using Perlin Noise
    
    '''
    density= octaves*100
    noise = PerlinNoise(octaves=octaves, seed=seed)
    coords = np.array([noise(i/density) for i in range(density)])
    coords += np.abs(np.min(coords))
    coords /= np.max(coords)
    return coords


def get_path(x_screen=x_screen, y_screen=y_screen, octaves=5, seeds=[3, 17], random_flag=1):
    '''
    For generating paths using coordinates
    obtained from get_coord function
    
    '''
    if random_flag:
        octaves = np.random.randint(3, 5)
        seeds = np.random.random((1, 2))[0]*1000
    
    x_coord = get_coord(octaves=octaves, seed=seeds[0])*x_screen
    y_coord = get_coord(octaves=octaves, seed=seeds[1])*y_screen
    path = np.dstack((x_coord, y_coord))[0]
    return path


############################################################################################
############################################################################################

'''
Functions for plotting graphs
and generating animations
'''

def plot_curve(ax, x_label, y_label, title, vals, legends):
    line_style_arr = ["solid", "dashed", "dotted", "dashdot"]
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    line_style_index = 0
    ax.set_title(title)
    for val in vals:
        if len(val.shape) == 1:
            ax.plot(val)
        elif len(val.shape) == 2:
            ax.plot(val[:,0], val[:,1], linestyle=line_style_arr[line_style_index])
            line_style_index = (line_style_index + 1)%len(line_style_arr)
    ax.legend(legends)


def bar_graph(ax, y_label, title, categories, vals):
    ax.set_title(title)
    rects = ax.bar(categories, vals)
    ax.set_ylabel(y_label)
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., height,
                f"{height}", ha='center', va='bottom')
        

def plot_vlines(ax, instants):
    for pt in instants:
        ax.axvline(x=pt, linewidth=1, alpha=0.3, color="white")


class animate():
    def __init__(self, ax, x_label, y_label, title, data, global_best, init_guess):
        self.load(ax, x_label, y_label, title, data, global_best, init_guess)
        self.line_style_arr = ["solid", "dashed", "dotted", "dashdot"]
        
    def load(self, ax, x_label, y_label, title, data, global_best, init_guess):
        self.ax = ax
        self.data = data
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        self.g_best = global_best
        self.init_guess = init_guess
        self.num_lines = self.data.shape[0]
        
        self.oneD = True
        if self.data.shape[-1] == 2:
            self.oneD = False
    
    
    def update(self, frame):
        line_style_index = 0
        for index in range(self.num_lines):
            curr_pt = self.data[index, :frame]
            if self.oneD:
                self.ax.plot(curr_pt, linestyle=self.line_style_arr[line_style_index])
            elif not self.oneD:
                self.ax.plot(curr_pt[:, 0], curr_pt[:, 1], linestyle=self.line_style_arr[line_style_index])
            line_style_index = (line_style_index + 1)%len(self.line_style_arr)
        
        if self.oneD:
            self.ax.axhline(y=self.g_best, alpha=0.5, color="green", linestyle="dashed")
            self.ax.axhline(y=self.init_guess, alpha=0.5, color="red", linestyle="dashed")
        
        elif not self.oneD:
            self.ax.scatter(self.g_best[0], self.g_best[1], color="green")
            self.ax.scatter(self.init_guess[0], self.init_guess[1], color="red")
        
    
############################################################################################
############################################################################################


