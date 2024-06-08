import numpy as np
from tqdm import tqdm
from .bot_sim import ETC_PSO_Member


############################################################################################
############################################################################################


class PSO():
    '''
    Class for implementing
    Particle Swarm Optimization
    Design Variable contains:
    P, I, D (each for linear and angluar errors) 
    and Event Triggering Threshold value
    Thus, 7 variables in total
    
    '''
    
    def __init__(self, num_bots, **sys_params) -> None:
        self.best_score = 0.
        self.num_bots = num_bots
        self.global_best = sys_params["init_guess"]
        self.bots = [ETC_PSO_Member(**sys_params) for _ in range(num_bots)]
    

    def eval_all(self):
        '''
        All the members are simulated
        serially and global best design
        variables and scores are updated;
        but parallel execution maybe
        introduced if needed
        
        '''
        
        scores = []
        curr_vars = []
        
        for bot in self.bots:
            score = bot.eval()
            scores.append(score)
            curr_vars.append(np.copy(bot.vars))
            
            if score > self.best_score:
                self.global_best = np.copy(bot.vars)
                self.best_score = score
        return scores, curr_vars
                
    
    def update_vars(self):
        '''
        Design Variables are updated as:
        new_var = (old_var + 
                   c1*lr1*(personal_best - old_var) + 
                   c2*lr2*(global_best - old_var)), where,
                   
        lr1: Cognitive Learning Rate
        lr2: Social Learning Rate
        c1 and c2 are randomly generated numbers
        distributed uniformly between 0 to 1
        
        '''
        
        for bot in self.bots:
            random_coeffs = np.random.random((2))*self.lrs
            bot.vars += (random_coeffs[0]*(bot.personal_best-bot.vars) +
                         random_coeffs[1]*(self.global_best-bot.vars))


    def train(self, num_iter, learning_rates):
        '''
        Training PSO
        
        '''
        
        vars_history = []
        scores_history = []
        best_scores_history = []
        self.lrs = learning_rates
        
        for _ in tqdm(range(num_iter)):
            scores, curr_vars = self.eval_all()
            self.update_vars()
            
            scores_history.append(scores)
            vars_history.append(curr_vars)
            best_scores_history.append(self.best_score)
        
        bot = self.bots[0]
        bot.vars = self.global_best
        score = bot.eval()
        
        output_dict = {"P": bot.P,
                       "I": bot.I,
                       "D": bot.D,
                       "max_score": score,
                       "thresh": bot.thresh,
                       "vars_history": np.array(vars_history),
                       "scores_history": np.array(scores_history),
                       "best_scores_history": np.array(best_scores_history)}
        return output_dict
        

############################################################################################
############################################################################################

