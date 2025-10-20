# Smart Bot Control: Event-Triggered PID Tuning with PSO

This project demonstrates an intelligent control strategy where a path-following robot is optimized using **Particle Swarm Optimization (PSO)**.

The simulation, built with PyGame, showcases how PSO can discover the optimal PID gains and, more importantly, an **event-triggering threshold**. This allows the robot to operate efficiently, updating its control signals only when necessary, rather than at every fixed time interval. The result is a robust and resource-conscious control system.



## 🌟 Key Features

-   **Particle Swarm Optimization**: Intelligently searches for the best PID gains (`Kp`, `Ki`, `Kd`) and the event-triggering threshold.
-   **Event-Triggered Control (ETC)**: Reduces unnecessary control updates, saving computational resources compared to traditional time-triggered systems.
-   **Dynamic Path Generation**: The robot follows a complex, unpredictable path generated using Perlin Noise.
-   **PyGame Simulation**: Provides a clear and interactive visualization of the bot's behavior and the learning process.

## 🤖 See it in Action

Watch as the PSO algorithm refines the control parameters over generations, teaching the bot to follow the path with increasing precision.

**PSO Training Process**
<kbd><img src="https://github.com/SoumyodiptaNath/Event_Triggered_Control_using_PSO/assets/122808862/883c3bf5-5b35-4a74-9183-24bc4d922abf" alt="PSO Training GIF"> </kbd>
<br/><br/>

**ETC vs. Traditional Control**
Notice how the Event-Triggered bot (right) achieves the same performance as the Time-Triggered bot (left) but with significantly fewer control updates (indicated by the red flashes).

<kbd><img src="https://github.com/SoumyodiptaNath/Event_Triggered_Control_using_PSO/assets/122808862/eb61fa5a-38ec-42cb-b524-773da172099d" alt="ETC vs TTC Comparison GIF"></kbd>
<br/><br/>

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.x
-   Git

### 2. Installation & Setup

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/your-username/Event_Triggered_Control_using_PSO.git](https://github.com/your-username/Event_Triggered_Control_using_PSO.git)

# Navigate to the project directory
cd Event_Triggered_Control_using_PSO

# Install dependencies
pip install -r requirements.txt
````

### 3\. Run the Simulation

Execute the main script to start the simulation and training process.

```bash
# On Linux or macOS
python3 run.py

# On Windows
python run.py
```

## 🔧 Configuration

All key parameters for the simulation, the bot, and the PSO algorithm are centralized in `SOURCE/utilities.py`. You can modify them directly in the file to experiment with different behaviors.

#### Pygame Simulation Parameters (`game_params`)

  - `x_screen`: Width of the simulation window.
  - `y_screen`: Height of the simulation window.
  - `bot_radius`: Visual size of the bot.
  - `Q`: Weighting matrix for determining ETC triggers.
  - `dt`: Simulation time step duration.

#### PSO Member Parameters (`etc_pso_params`)

  - `dt`: Time step for the individual bot simulation.
  - `init_guess`: Central values for the initial random swarm positions (PID gains, Threshold).
  - `max_iter`: Maximum simulation steps for a single bot's evaluation.
  - `range_var`: The spread around `init_guess` for initial particle positions.
  - `Q_err`: Weighting matrix for calculating the bot's path deviation error.

#### PSO Algorithm Parameters (`pso_params`)

  - `num_bots`: The number of particles (bots) in the swarm.
  - `learning_rate`: Cognitive and social learning rates (`c1`, `c2`).
  - `num_steps`: The maximum number of PSO iterations (generations).

## 🛠️ Tech Stack & Concepts

  - **Algorithm**: Particle Swarm Optimization (PSO)
  - **Control Theory**: Event-Triggered Control (ETC), PID Control
  - **Simulation**: PyGame
  - **Core Language**: Python (NumPy)
  - **Concepts**: Numerical Methods, Physics Simulation, Optimization

## 🤝 Contributing

Contributions are always welcome\!

This project is open to contributions, bug reports, and suggestions. If you've found a bug, have an idea for an improvement, or want to add a new feature, please feel free to open an issue or submit a pull request.

