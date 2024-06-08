import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
from .bot_sim import Bot


############################################################################################
############################################################################################


class GameHandler(Bot):
    '''
    Similar to ETC_PSO_Member, except
    this is used to show the simulation
    using PyGame Window
    
    '''
    
    def __init__(self, x_screen, y_screen, bot_radius, Q, dt) -> None:
        pygame.init()
        super().__init__(dt)
        
        self.Q = Q
        self.r = bot_radius
        self.clock = pygame.time.Clock()
        self.x_screen = x_screen; self.y_screen = y_screen
        self.screen = pygame.display.set_mode((x_screen, y_screen))
        self.path_screen = pygame.Surface((x_screen, y_screen))
        
        self.bg_1 = (16, 20, 31)
        self.bg_2 = (37, 58, 94)
        self.dot_1 = (60, 94, 139)
        self.dot_2 = (190, 119, 43)
        self.dot_3 = (164, 221, 219)
        self.red = (239, 71, 111)
        self.green = (6, 214, 160)
        self.body = (232, 193, 112)
        self.border = (207, 87, 60)
        self.path_screen.fill(self.bg_1)
        
    
    def quit(self):
        pygame.quit()
    
    
    def load_bot(self, path, P, I, D):
        self.load(path, P, I, D)
        self.path_screen.fill(self.bg_1)
        pygame.draw.lines(self.path_screen, self.bg_2, False, path, 2)
        for pt in path:
            pygame.draw.circle(self.path_screen, self.dot_1, pt, 2)
            
        font = pygame.font.SysFont('Comic Sans MS', 20)
        text = font.render(f"Control Input is sent to the Bot only when the LED at the center turns Green", 0, "white")
        text_rect = text.get_rect()
        text_rect.center = (self.x_screen//2, 10)
        self.path_screen.blit(text, text_rect)
        
    
    
    def draw(self):
        curr_pos = self.curr_state[:2]
        pygame.draw.circle(self.path_screen, self.dot_2, curr_pos, 1)
        self.screen.blit(self.path_screen, (0,0))
        pygame.draw.circle(self.screen, self.body, curr_pos, self.r)
        pygame.draw.line(self.screen, self.border, curr_pos, curr_pos+(self.r*self.dir), self.r//5)
        pygame.draw.circle(self.screen, self.border, curr_pos, 1.1*self.r, self.r//5)
        pygame.draw.circle(self.screen, self.color, curr_pos, self.r//5)
        pygame.draw.circle(self.screen, "white", self.path_coord[self.curr_index], 4)
        pygame.display.update()
        
    
    def simulate(self, thresh, plot_flag=0):
        if plot_flag:
            v_left = []
            v_right = []
            bot_traj = []
            pos_error = []
            angle_error = []
            etc_instants = []
        
        running = True
        count_with_etc = 0
        count_with_ttc = 0
        prev_err_vel = np.zeros(self.num_inputs)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; break
            stop_flag, err_vel = self.get_vels()
            if stop_flag: running = False; break
            
            del_err = err_vel-prev_err_vel
            if del_err.T @ self.Q @ del_err >= thresh:
                count_with_etc += 1
                self.step_sim(err_vel)
                self.color = self.green
                prev_err_vel = np.copy(err_vel)
                if plot_flag: etc_instants.append(count_with_ttc)
            else:
                self.color = self.red
                self.step_sim(prev_err_vel)
            count_with_ttc += 1
            
            if plot_flag:
                pos_error.append(self.err[0])
                angle_error.append(self.err[1])
                bot_traj.append(np.copy(self.curr_state[:2]))
                v_mag = np.linalg.norm(self.curr_state[:2])
                v_left.append(v_mag + self.r*self.curr_state[-1])
                v_right.append(v_mag - self.r*self.curr_state[-1])
            
            self.draw()
            self.clock.tick(50)
        
        if plot_flag:
            output_dict = {"v_left": np.array(v_left),
                           "v_right": np.array(v_right),
                           "bot_traj": np.array(bot_traj),
                           "pos_error": np.array(pos_error),
                           "count_with_etc": count_with_etc,
                           "count_with_ttc": count_with_ttc,
                           "angle_error": np.array(angle_error),
                           "etc_instants": np.array(etc_instants),
                           }
            return output_dict
        

############################################################################################
############################################################################################