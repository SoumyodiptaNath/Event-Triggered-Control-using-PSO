from .utilities import *
from .pso_algo import PSO
from .bot_sim import ETC_PSO_Member
from .pygame_handler import GameHandler

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


############################################################################################
############################################################################################


def RUN(file_path):
    '''
    FINAL RUN FUNTION FOR
    GENERATING PATHS,
    TRAINING PSO,
    SIMULATING BOT,
    PLOTTING AND SAVING RESULTS
    ALL-IN-ONE!
    
    '''
    
    plt.style.use("dark_background")

    path = get_path(random_flag=1)
    etc_pso_params["path"] = path
    print("\nNew Path Assigned Successfully...")
    print(f"Number of Waypoints: {path.shape[0]}")
    
    
############################################################################################
############################################################################################

    '''
    Simulation with initial Guess
    
    '''
    
    bot = ETC_PSO_Member(**etc_pso_params)
    bot.vars = etc_pso_params["init_guess"]
    score = bot.eval()
    
    print("\nInitial Guess...")
    print(f"P: {np.round(bot.P, 3)}")
    print(f"I: {np.round(bot.I, 3)}")
    print(f"D: {np.round(bot.D, 3)}")
    print(f"ETC Thresh: {bot.thresh}")
    print(f"Score on current path with initial Guess: {np.round(score, 3)}")
    

############################################################################################
############################################################################################

    '''
    PSO Training
    
    '''
    
    print("\nInitialising PSO training...\n")
    pso = PSO(pso_params["num_bots"], **etc_pso_params)
    output = pso.train(pso_params["num_steps"], pso_params["learning_rate"])
    
    print("\nTrained Weights...")
    print(f"P: {np.round(output['P'], 3)}")
    print(f"I: {np.round(output['I'], 3)}")
    print(f"D: {np.round(output['D'], 3)}")
    print(f"ETC Thresh: {np.round(output['thresh'], 3)}")
    print(f"Max Score on current path after training: {np.round(output['max_score'], 3)}")
    print("Generating Plots on training results... Please Wait")

    
############################################################################################
############################################################################################

    '''
    Ploting Evolution of scores
    
    '''
    
    i = 0
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    ax[i].imshow(output["scores_history"], cmap='cividis', alpha=0.9)
    ax[i].set_title("EVOLUTION OF INDIVIDUAL SCORES")
    ax[i].set_ylabel("Number of Iteratiosn")
    ax[i].set_xlabel("Members")
    i += 1
    
    params = {"ax": ax[i],
              "x_label": "Number of Iterations",
              "y_label": "Best Global Scores",
              "vals": [output["best_scores_history"]],
              "title": f"EVOLUTION OF BEST GLOBAL SCORE (MAX: {np.round(output['max_score'], 3)})",
              "legends": []
              }
    plot_curve(**params)
    i += 1
    
    plt.savefig(f"{file_path}\EVOLUTION OF SCORES.png", dpi=300)
    plt.show()
    

############################################################################################
############################################################################################

    '''
    Plotting & Saving animation for 
    Evolution of Trained Parameters
    
    '''
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    params = {"ax": ax,
              "init_guess": bot.P,
              "global_best": output["P"],
              "x_label": "P (for Linear Velocity)",
              "y_label": "P (for Angular Velocity)",
              "title": f"EVOLUTION OF WEIGHTS (P); \nGreen Dot: Final Soln; Red Dot: Initial Guess",
              "data": np.array([output["vars_history"][:, k, 0:2] for k in range(pso.num_bots)])
              }
    anim = animate(**params)
    final = FuncAnimation(fig=fig, func=anim.update, frames = range(1, pso_params["num_steps"]), interval=100)
    final.save(f"{file_path}\EVOLUTION OF WEIGHTS (P).gif")
    print("Saving Animation for P"); plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    params = {"ax": ax,
              "init_guess": bot.I,
              "global_best": output["I"],
              "x_label": "I (for Linear Velocity)",
              "y_label": "I (for Angular Velocity)",
              "title": f"EVOLUTION OF WEIGHTS (I); \nGreen Dot: Final Soln; Red Dot: Initial Guess",
              "data": np.array([output["vars_history"][:, k, 2:4] for k in range(pso.num_bots)])
              }
    anim.load(**params)
    final = FuncAnimation(fig=fig, func=anim.update, frames = range(1, pso_params["num_steps"]), interval=100)
    final.save(f"{file_path}\EVOLUTION OF WEIGHTS (I).gif")
    print("Saving Animation for I"); plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    params = {"ax": ax,
              "init_guess": bot.D,
              "global_best": output["D"],
              "x_label": "D (for Linear Velocity)",
              "y_label": "D (for Angular Velocity)",
              "title": f"EVOLUTION OF WEIGHTS (D); \nGreen Dot: Final Soln; Red Dot: Initial Guess",
              "data": np.array([output["vars_history"][:, k, 4:6] for k in range(pso.num_bots)])
              }
    anim.load(**params)
    final = FuncAnimation(fig=fig, func=anim.update, frames = range(1, pso_params["num_steps"]), interval=100)
    final.save(f"{file_path}\EVOLUTION OF WEIGHTS (D).gif")
    print("Saving Animation for D"); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    params = {"ax": ax,
              "init_guess": bot.thresh,
              "global_best": output["thresh"],
              "x_label": "Training Iteration",
              "y_label": "Event Triggering Threshold",
              "title": f"EVOLUTION OF WEIGHTS (thresh); \nGreen Line: Final Soln; Red Line: Initial Guess",
              "data": np.array([output["vars_history"][:, k, -1] for k in range(pso.num_bots)])
              }
    anim.load(**params)
    final = FuncAnimation(fig=fig, func=anim.update, frames = range(1, pso_params["num_steps"]), interval=100)
    final.save(f"{file_path}\EVOLUTION OF WEIGHTS (thresh).gif")
    print("Saving Animation for Thresh"); plt.close()
    

############################################################################################
############################################################################################

    '''
    Plotting & Saving Final Plot for 
    Evolution of Trained Parameters
    
    '''
    
    i, j = 0, 0
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    
    params = {"ax": ax[i][j],
              "x_label": "P (for Linear Velocity)",
              "y_label": "P (for Angular Velocity)",
              "title": f"Proportional Coeff (P)",
              "vals": [output["vars_history"][:, k, 0:2] for k in range(pso.num_bots)],
              "legends": []                          
            }
    ax[i][j].scatter(output['P'][0], output['P'][1], color="green")
    ax[i][j].scatter(bot.P[0], bot.P[1], color="red")
    plot_curve(**params)
    j += 1
    
    params = {"ax": ax[i][j],
              "x_label": "I (for Linear Velocity)",
              "y_label": "I (for Angular Velocity)",
              "title": f"Integral Coeff (I)",
              "vals": [output["vars_history"][:, k, 2:4] for k in range(pso.num_bots)],
              "legends": []                          
            }
    ax[i][j].scatter(output['I'][0], output['I'][1], color="green")
    ax[i][j].scatter(bot.I[0], bot.I[1], color="red")
    plot_curve(**params)
    i += 1
    
    params = {"ax": ax[i][j],
              "x_label": "D (for Linear Velocity)",
              "y_label": "D (for Angular Velocity)",
              "title": f"Differential Coeff (D)",
              "vals": [output["vars_history"][:, k, 4:6] for k in range(pso.num_bots)],
              "legends": []
            }
    ax[i][j].scatter(output['D'][0], output['D'][1], color="green")
    ax[i][j].scatter(bot.D[0], bot.D[1], color="red")
    plot_curve(**params)
    j -= 1

    params = {"ax": ax[i][j],
              "x_label": "Training Iteration",
              "y_label": "Event Triggering Threshold",
              "title": f"Event Triggering Threshold (thresh)",
              "vals": [output["vars_history"][:, k, -1] for k in range(pso.num_bots)],
              "legends": []
            }
    ax[i][j].axhline(y=output['thresh'], alpha=0.5, color="green", linestyle="dashed")
    ax[i][j].axhline(y=bot.thresh, alpha=0.5, color="red", linestyle="dashed")
    plot_curve(**params)
    i += 1
    
    fig.suptitle("EVOLUTION OF WEIGHTS; Green Dot: Final Soln; Red Dot: Initial Guess")
    plt.savefig(f"{file_path}\EVOLUTION OF WEIGHTS.png", dpi=300)
    plt.show()
    print("Plots generated successfully...")
    del pso, bot, anim


############################################################################################
############################################################################################

    '''
    Initiating PyGame simulation
    with Trained Parameters &
    Plotting the observations
    
    '''
    
    print("\nGenerating PyGame simulation...")
    game = GameHandler(**game_params)
    game.load_bot(path, output["P"], output["I"], output["D"])
    output = game.simulate(output["thresh"], plot_flag=1)
    game.quit()
    
    i = 0
    fig, ax = plt.subplots(3, 1, figsize=(15, 7))
    
    params = {"ax": ax[i],
              "x_label": "",
              "y_label": "Pixels per dt",
              "title": f"BOT PARAMETERS (dt: {game.dt})",
              "vals": [output["v_left"], output["v_right"]],
              "legends": ["Right Wheel Velocity Magnitudes",
                          "Left Wheel Velocity Magnitudes"]
              }
    plot_curve(**params)
    plot_vlines(ax[i], output["etc_instants"])
    i += 1
    
    params = {"ax": ax[i],
              "x_label": "",
              "y_label": "Pixels",
              "title": "",
              "vals": [output["pos_error"]],
              "legends": ["Positional Error"]
              }
    plot_vlines(ax[i], output["etc_instants"])
    plot_curve(**params)
    ax[i].set_ylim(np.min(output["pos_error"][5:]))
    i += 1
    
    params = {"ax": ax[i],
            "x_label": "Time Instants (Event Triggers marked as vertical lines)",
            "y_label": "Radians",
            "title": "",
            "vals": [output["angle_error"]],
            "legends": ["Angular Error"]
            }
    plot_vlines(ax[i], output["etc_instants"])
    plot_curve(**params)
    i += 1
    
    plt.savefig(f"{file_path}\BOT PARAMS dt_{game.dt}.png", dpi=300)
    plt.show()


############################################################################################
############################################################################################

    '''
    Comparing ETC and TTC
    in terms of accuracy &
    efficiency
    
    '''
    
    i = 0
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), width_ratios=[0.75, 0.25])
    
    game.path_coord[:, 1] *= -1
    output["bot_traj"][:, 1] *= -1
    params = {"ax": ax[i],
        "x_label": "X coordinate",
        "y_label": "Y coordinate",
        "title": "COMPARISON",
        "vals": [game.path_coord, output["bot_traj"]],
        "legends": ["Actual Path", "Bot's Path"]
        }
    plot_curve(**params)
    i += 1
    
    efficiency = np.round((1 - output["count_with_etc"]/output["count_with_ttc"])*100, 2)
    params = {"ax": ax[i],
              "y_label": "Interaction Count",
              "title": f"INTERACTIONS (ETC: {efficiency}% lesser)",
              "categories": ["With ETC", "With TTC"],
              "vals": [output["count_with_etc"], output["count_with_ttc"]]
              }
    bar_graph(**params)
    i += 1

    plt.savefig(f"{file_path}\COMPARISONS.png", dpi=300)
    plt.show()
    print(f"Number of Interactions with ETC: {output['count_with_etc']}")
    print(f"Number of Interactions with TTC: {output['count_with_ttc']}")
    print(f"ETC is better than TTC by {efficiency}%")
    print("Simulation completed successfully...\n")
    
    
############################################################################################
############################################################################################