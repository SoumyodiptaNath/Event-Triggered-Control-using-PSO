import numpy as np
from functools import partial


############################################################################################
############################################################################################


class Bot():
    '''
    Parent Class for BOT
    
    '''

    def __init__(self, dt):
        self.dt = dt
        self.num_states = 3
        self.num_inputs = 2
    
    
    def get_dir(self):
        return np.array([np.cos(self.curr_state[-1]), 
                         np.sin(self.curr_state[-1])])
    
    
    def load(self, path, P, I, D):
        '''
        For refreshing the states
        and loading new parameters:
        path coordinates, P, I, D
        
        '''
        
        self.P = P
        self.I = I
        self.D = D
        self.curr_index = 0
        self.path_coord = path
        self.num_way_points = path.shape[0]
        self.err = np.zeros(self.num_inputs)
        self.prev_err = np.zeros_like(self.err)
        self.curr_state = np.zeros(self.num_states)
        
        init_pos = path[0]
        init_dir = path[1] - path[0]
        init_angle = np.arctan2(init_dir[1], init_dir[0])
        self.curr_state[-1] = init_angle
        self.curr_state[:2] = init_pos
        self.dir = self.get_dir()


    def get_vels(self):
        '''
        For getting the current deviation of the Bot
        with respect to the desired trajectory and
        generating control input in terms of linear
        & angular velocities with the help of a 
        PID controller using the errors calculated
        
        '''
        
        del_pos = self.curr_state[:2] - self.path_coord[self.curr_index]
        self.err[0] = np.linalg.norm(del_pos)
        self.err[1] = np.cross(self.dir, del_pos)/(self.err[0]+1e-3)
        
        velocities = (self.err*self.P
                      + (self.err+self.prev_err)*self.I*0.5
                      + (self.err-self.prev_err)*self.D)
        self.prev_err = np.copy(self.err)
        
        stop_flag = False
        if self.err[0] <= 50.: 
            self.curr_index += 1
            if self.curr_index == self.num_way_points:
                stop_flag = True
        return stop_flag, velocities
    
    
    def step_sim(self, vels=np.zeros(2)):
        '''
        Using the control inputs linear &
        angular velocities to advance the Bot
        by a time step equivalent to dt
        
        '''
        
        self.curr_state[:2] += vels[0]*self.dt*self.dir
        self.curr_state[-1] += vels[1]*self.dt
        self.dir = self.get_dir()
        if np.abs(self.curr_state[-1]) >= np.pi:
            self.curr_state[-1] *= -1


############################################################################################
############################################################################################


class ETC_PSO_Member(Bot):
    '''
    Child Class for BOT simulation in PSO
    
    '''
    
    def __init__(self, path, dt, max_iter, Q_err, Q_ETC, init_guess, range_var):
        self.Q = Q_ETC
        self.Q_ = Q_err
        self.best_score = 0.
        super().__init__(dt)
        self.max_iter = max_iter
        self.load = partial(self.load, path)
        
        # Generating design varaibles randomly in the vicinity of an initial guess
        self.vars = init_guess + (np.random.random((3*self.num_inputs+1))-0.5)*range_var
        self.personal_best = np.copy(self.vars)


    def get_dist_covered(self, traj):
        del_traj = traj - np.roll(traj, shift = -1, axis=0)
        tot_dist = np.sum(np.linalg.norm(del_traj[:-1,:], axis=1))
        return tot_dist


    def eval(self):
        '''
        For simulating the bot's trajectory and 
        calculating the score based on its 
        performance where the parameters:
        P, I, D and Event Triggering Threshold
        are learnt from the PSO algorithm
        
        ''' 
        
        bot_traj = []
        avg_error = 0.
        running = True
        count_with_ttc = 0
        count_with_etc = 0
        prev_vels = np.zeros(self.num_inputs)
        
        '''
        Deciphering P, I, D and Event Triggering thresh
        from the design variables learnt using PSO
        
        '''
        
        self.thresh = self.vars[-1]
        self.load(P = self.vars[:self.num_inputs],
                  I = self.vars[self.num_inputs:2*self.num_inputs],
                  D = self.vars[2*self.num_inputs:3*self.num_inputs])

        
        while running:
            stop_flag, vels = self.get_vels()
            if stop_flag: running = False; break
            
            '''
            The New Control Input is fed to the bot
            only when the error in Current Input and
            Newly Generated Input is greater than a
            threshold (Type of an Event Triggered Control),
            where the error in inputs are weighed by a
            Positive Deifnite Matrix Q_ETC
            
            '''
            
            del_vels = vels - prev_vels
            if del_vels.T @ self.Q @ del_vels >= self.thresh:
                self.step_sim(vels)
                count_with_etc += 1
                prev_vels = np.copy(vels)
            else:
                self.step_sim(prev_vels)
            
            '''
            Simulation Stops if Current Deviation of the bot
            with respect to the Trajectory is significantly
            High or the simulation excceds the Maximum Number
            of Iterations; again the error is weighed by another
            Positive Definite Matrix Q_err
            
            '''
            
            curr_error = self.err.T @ self.Q_ @ self.err
            if curr_error > 1e2 or count_with_ttc > self.max_iter:
                running = False
                
            count_with_ttc += 1
            avg_error += curr_error*self.dt
            bot_traj.append(np.copy(self.curr_state[:2]))
        
        '''
        Final Score is calculated based on bot's performance
        and the personal best design vairbales along with the
        personal best score for the current member are updated
        if the score is higher than the previous personal best
        
        '''
        
        traj_diff = (self.get_dist_covered(np.array(bot_traj))
                     - self.get_dist_covered(self.path_coord))
        
        score = (1e3 + (count_with_ttc - count_with_etc)*0.35
                 - (np.abs(traj_diff))*0.65 - avg_error)
        
        if score > self.best_score:
            self.personal_best = np.copy(self.vars)
            self.best_score = score
        return score
        
        
############################################################################################
############################################################################################