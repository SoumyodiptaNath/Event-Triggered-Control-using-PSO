
# Event Triggered Control with Particle Swarm Optimization

A Python interface for training optimal control parameters using Particle Swarm Optimization for Event Triggered Control of a Path Following Bot Simulation.

The PSO algorithm is used to find the optimal P, I, D and Event Triggering Threshold (Thresh) values for controlling the bot, where the control input is fedback only when the weighted sum of error in current input and last input provided exceeds the Thresh. The path is randomly generated using Perlin Noise.


## Usage/Examples

- Download the project repository to your local machine
- Navigate to the project directory using `cd`
- Install dependencies using `pip install -r requirements.txt`
- Run `python3 run.py` if on Linux or Mac
- Run `python run.py` if on Windows

Watch the algorithm learn the weights...<br/><br/>
<img src="https://github.com/SoumyodiptaNath/Event_Triggered_Control_using_PSO/assets/122808862/883c3bf5-5b35-4a74-9183-24bc4d922abf)", align="center"> 
<br/><br/>
Comparisons between Event Triggered Control and Time Triggered Control...<br/><br/>
<img src="https://github.com/SoumyodiptaNath/Event_Triggered_Control_using_PSO/assets/122808862/eb61fa5a-38ec-42cb-b524-773da172099d", align="center">
<br/><br/>
</center>

## Configurations

All the parameters for Bot, Event Triggered Control PSO Member and PSO training itself have been defined in `\SOURCE\utilities.py` as follows:

- Pygame Simulation Parameters:

```
game_params = {
    x_screen: Width of simulation Wndow
    y_screen: Height of simulation Window
    bot_radius: Radius of the Bot 
    Q: Weights for determining ETC triggers
    dt: Value of one time step
}
```

- ETC PSO Member Parameters:

```
etc_pso_params = {
    dt: Value of one time step
    init_guess: Initial Guess for PSO
    Q_ETC: Weights for determining ETC triggers
    max_iter: Max no. of iterations for simulation
    range_var: Range of variables randomly chosen around init_guess
    Q_err: Weights for determining current linear & angular deviations
}
```

- PSO Training Parameters:

```
pso_params = {
    num_bots: Number of particles in PSO
    learning_rate: Cognitive & Social Learning Rates
    num_steps: Number of max iterations for PSO training
}
```
All of these can be accessed in `run.py` for modifications as per requirement.


## Contributing

Contributions are always welcome!

This project is open to contributions, bug reports, and suggestions. If you've found a bug or have a suggestion, please open an issue.


## 🛠 Skills
Particle Swarm Optimisation, Event Trigger Control, Linear Control Systems, Numerical Methods, Physics Simulation, Python, PyGame

