import os
import numpy as np
from sys import path
from datetime import datetime    

from SOURCE.main import *


if __name__ == "__main__": 
    file_path = os.path.join(path[0], "OUTPUTS", datetime.now().strftime(rf"%d_%m_%Y %H_%M_%S"))
    print(f"\nCheck {file_path} for output plots...")
    os.mkdir(file_path)
    
    '''
    CHANGE THE PARAMS IF NEEDED BY
    UN-COMMENTING THE DESIRED PARAM AND
    ASSIGNING NEW VALUES, DEFAULT VALUES
    ARE SHOWN HERE...
    
    ORIGINAL PARAMS HAVE BEEN DEFINED IN
    SOURCE\\utilities.py...
    
    REVERT TO DEFAULT IF PERFORMANCE
    DETERIORATES!
    
    '''
    
    # y_screen = 800    # dimensions of simulation window
    # x_screen = 1300   # dimensions of simulation window
                  
    # etc_pso_params["dt"] = 0.01                       # Value of one time step
    # etc_pso_params["max_iter"] = 1e3                  # Max no. of iterations for simulation
    # etc_pso_params["Q_err"] = np.diag([0.01, 50.])    # err.T @ Q_err @ err: Weights for determining current linear & angular deviations
    # etc_pso_params["Q_ETC"] = np.diag([0.01, 5.])     # diff_input.T @ Q_ETC @ diff_input: Weights for determining ETC triggers
    # etc_pso_params["init_guess"] = np.array([10., -50., 1., 10., 1., 25., 300.])      # Initial Guess for PSO
    # etc_pso_params["range_var"] = np.array([50., 50., 25., 25., 25., 25., 250.])      # Range of variables randomly chosen around init_guess

    # game_params["x_screen"] = x_screen
    # game_params["y_screen"] = y_screen
    # game_params["bot_radius"] = (x_screen + y_screen)//70
    # game_params["Q"] = etc_pso_params["Q_ETC"]
    # game_params["dt"] = etc_pso_params["dt"]

    # pso_params["num_bots"] = 50       # Number of particles in PSO
    # pso_params["num_steps"] = 50      # Number of max iterations for PSO training
    # pso_params["learning_rate"] = np.array([0.05, 0.1])       # Cognitive & Social Learning Rates
    
    
    '''
    FINAL RUN FUNTION FOR:
    TRAINING, SIMULATING, PLOTTING
    AND SAVING RESULTS
    
    '''
    
    RUN(file_path)