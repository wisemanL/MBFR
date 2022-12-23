from Src.Utils import utils , Basis
from Src.Algorithms import NS_utils
import numpy as np
import torch
import matplotlib.pyplot as plt
torch.set_printoptions(threshold=10_000)


class model :
    def __init__(self,config):
        self.state_features = Basis.get_Basis(config=config)
        self.state_dim = config.env.observation_space.shape[0]
        self.n_action = config.env.n_actions
        _ , self.atype, self.action_size = NS_utils.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        self.realTrajectory_memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        self.counter = 0
        self.config = config

        self.grid_size  = self.config.grid_size
        self.reward_function_reference_lag = self.config.reward_function_reference_lag
        self.reward_function_predict_lag = self.config.reward_function_predict_lag

        ## divide the 2D reacher into grid_size tables.
        self.transition_visit = torch.zeros((self.grid_size,self.grid_size,self.n_action,self.grid_size,self.grid_size),requires_grad=False,device=self.config.device)
        self.transition_prob = torch.zeros((self.grid_size,self.grid_size,self.n_action,self.grid_size,self.grid_size),requires_grad=False,device=self.config.device)
        # transition_prob: (s_x,s_y,a,s_next_x,s_next_y)
        self.reward_table = torch.zeros((self.grid_size,self.grid_size,self.n_action,self.reward_function_reference_lag+self.reward_function_predict_lag),requires_grad=False,device=self.config.device)

        ## compute pseudo inverse of naive least square method
        self.pseudo_inv_X = self.compute_pseudo_inverse_X()

        self.current_G = None


        self.self_counter_for_drawing = 0

    def reset(self):
        self.realTrajectory_memory.next()
        self.counter += 1
        self.gamma_t = 1

    def modify_matrix_to_visualize_r(self,matrix):
        ##[1] take transpose
        matrix = torch.transpose(matrix, 0, 1)
        ##[2] flip along grid_size/2
        matrix2 = torch.zeros_like(matrix)
        for i in range(self.grid_size) :
            matrix2[i,:] = matrix[self.grid_size-1-i,:]
        return matrix2

    def visualize_r(self):
        plt.close("all")
        for i in range(self.n_action) :
            q_map = plt.figure(i)
            ax = plt.gca()
            q_modified = self.modify_matrix_to_visualize_q(self.q[:,:,i])
            im = ax.imshow(q_modified, cmap='cividis')
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Q value', rotation=-90, va='bottom')
            ax.set_xticklabels([p for p in range(self.grid_size)])
            ax.set_yticklabels([p for p in range(self.grid_size)])
            ax.set_title("Q value function : action " + str(i))
            plt.tight_layout()
        # plt.show()




    def get_future_reward_from_model(self,state,action,p):
        sx_d, sy_d = self.convert_cState_to_dState(state)
        return self.reward_table[sx_d,sy_d,action,p]

    def get_next_state_from_model(self,state,action):
        sx_d , sy_d = self.convert_cState_to_dState(state)
        before_reshape_next_state_prob = self.transition_prob[sx_d,sy_d,action,:,:].cpu().data.numpy()
        after_reshape_next_state_prob = self.transition_prob[sx_d,sy_d,action,:,:].cpu().reshape(-1).data.numpy()
        if abs(np.sum(after_reshape_next_state_prob)-1) < 1e-3 :
            new_state_index = np.random.choice(np.arange(len(after_reshape_next_state_prob)), p=after_reshape_next_state_prob)
            wherecomesfrom = "NotRandom"
        elif abs(np.sum(after_reshape_next_state_prob) - 0) < 1e-3 :
            new_state_index = np.random.choice(np.arange(len(after_reshape_next_state_prob)))
            wherecomesfrom = "Random"

        sx_next_d_manuel = new_state_index // before_reshape_next_state_prob.shape[1]
        sy_next_d_manuel = new_state_index % before_reshape_next_state_prob.shape[1]
        (sx_next_d, sy_next_d) = np.unravel_index(np.ravel_multi_index((new_state_index,), after_reshape_next_state_prob.shape), before_reshape_next_state_prob.shape)
        assert sx_next_d_manuel == sx_next_d and sy_next_d_manuel == sy_next_d

        return (sx_next_d , sy_next_d) , wherecomesfrom

    def compute_pseudo_inverse_X(self):
        X = torch.zeros((self.reward_function_reference_lag, 2),requires_grad=False,device=self.config.device)
        for i in range(self.reward_function_reference_lag) :
            X[i,0] = i+1
            X[i,1] = 1

        X_transpose = torch.transpose(X,0,1)
        pseudo_inverse = torch.matmul(torch.linalg.inv(torch.matmul(X_transpose,X)),X_transpose)
        return pseudo_inverse

    def add_visit_count(self,s,a,s_next):
        # convert the continuous s,s_next to discrete value
        (sx_d, sy_d),(sx_next_d, sy_next_d) = self.convert_cState_to_dState(s), self.convert_cState_to_dState(s_next)
        self.transition_visit[sx_d, sy_d, a, sx_next_d, sy_next_d] += 1

    def add_reward_table(self,s,a,r,episode):
        (sx_d, sy_d) = self.convert_cState_to_dState(s)
        if episode < self.reward_function_reference_lag :
            self.reward_table[sx_d,sy_d,a,episode] = r
            # print("save reward table index")
            # print(episode)
            # print("=================")
        else :
            self.reward_table[sx_d, sy_d, a, self.reward_function_reference_lag-1] = r
            # print("save the latest reward table index")
            # print(self.reward_function_reference_lag-1)
            # print("=================")

    def move_forward_reward_table(self,episode1):
        # move i+1 reward table to i
        if episode1 <= self.reward_function_reference_lag :
            return
        else :
            # print("========move reward table =========")
            for i in range(1,self.reward_function_reference_lag) :
                # print("move "+str(i)+" to "+str(i-1))
                self.reward_table[:, :, :, i-1] = self.reward_table[:, :, :, i]
            # print("====================================")
            # print("======= reset the index : "+str(self.reward_function_reference_lag-1) + " reward table =======")
            self.reward_table[:,:,:,self.reward_function_reference_lag-1] = self.config.env.step_reward



    def update_transition_probability(self):
        total_visit = torch.sum(self.transition_visit, dim=[3, 4]).clone().detach()
        for idx_sx in range(self.transition_visit.size()[0]):
            for idx_sy in range(self.transition_visit.size()[1]):
                for idx_a in range(self.transition_visit.size()[2]):
                    if total_visit[idx_sx, idx_sy, idx_a] == 0:
                        self.transition_prob[idx_sx, idx_sy, idx_a, :, :] = 0
                    else:
                        self.transition_prob[idx_sx, idx_sy, idx_a, :, :] = torch.div(
                            self.transition_visit[idx_sx, idx_sy, idx_a, :, :], total_visit[idx_sx, idx_sy, idx_a])

    def update(self, s1, a1, prob, r1, s2, done,episode1):
        # save the trajectory to the D_{env}
        self.realTrajectory_memory.add(s1, a1, prob, self.gamma_t * r1)
        self.gamma_t *= self.config.gamma

        # update the visit count table
        self.add_visit_count(s1, a1, s2)

        # update the reward table
        self.add_reward_table(s1, a1, r1,episode1)


        if done and episode1 >= self.reward_function_reference_lag:
            # print("==== predict future reward ====")
            # print(self.reward_function_reference_lag)
            self.optimize()
            # print("===============================")
    def convert_cState_to_dState(self,state):
        # x_batch,y_batch = state[:,[0]],state[:,[1]]
        x , y = state[0], state[1]
        upper_right , lower_left = \
            torch.tensor(self.config.env.observation_space.high,requires_grad=False,device=self.config.device) , \
            torch.tensor(self.config.env.observation_space.low,requires_grad=False,device=self.config.device)
        x_left , x_right , y_down , y_up = lower_left[0],upper_right[0] , lower_left[1] , upper_right[1]

        i,j = torch.floor((x-x_left) / (x_right - x_left) * self.grid_size) ,  torch.floor((y-y_down) / (y_up - y_down) * self.grid_size)
        return i.type(torch.long),j.type(torch.long)

    @classmethod
    def convert_dState_to_random_cState(cls,state,config) :
        B = state.shape[0]
        i,j = state[:,0],state[:,1]
        upper_right , lower_left = \
            torch.tensor(config.env.observation_space.high,requires_grad=False,device=config.device) , \
            torch.tensor(config.env.observation_space.low,requires_grad=False,device=config.device)
        x_left , x_right , y_down , y_up = lower_left[0],upper_right[0] , lower_left[1] , upper_right[1]

        x_lower = i * (x_right-x_left) / config.grid_size
        x_upper = (i+1) * (x_right-x_left) / config.grid_size
        y_lower = j * (y_up-y_down) / config.grid_size
        y_upper = (j+1) * (y_up-y_down) / config.grid_size
        x_random = torch.rand(B) * (x_upper-x_lower) + x_lower
        y_random = torch.rand(B) * (y_upper-y_lower) + y_lower
        return torch.cat((torch.unsqueeze(x_random,dim=-1),torch.unsqueeze(y_random,dim=-1)),dim=1)

    @classmethod
    def convert_dState_to_specific_cState(cls,state,config) :
        B = state.shape[0]
        i,j = state[:,0],state[:,1]
        upper_right , lower_left = \
            torch.tensor(config.env.observation_space.high,requires_grad=False,device=config.device) , \
            torch.tensor(config.env.observation_space.low,requires_grad=False,device=config.device)
        x_left , x_right , y_down , y_up = lower_left[0],upper_right[0] , lower_left[1] , upper_right[1]

        x_lower = i * (x_right-x_left) / config.grid_size
        x_upper = (i+1) * (x_right-x_left) / config.grid_size
        y_lower = j * (y_up-y_down) / config.grid_size
        y_upper = (j+1) * (y_up-y_down) / config.grid_size
        x_random = x_lower
        y_random = y_lower
        return torch.cat((torch.unsqueeze(x_random,dim=-1),torch.unsqueeze(y_random,dim=-1)),dim=1)


    def divide_into_sars_list(self,s,a,r):
        assert s.size()[:-1] == a.size()[:-1]
        assert s.size()[:-1] == r.size()

        total_step = s.size()[1]
        sars_list = []
        for i in range (total_step-1) :
            sars_list.append([s[:,i],a[:,i],r[:,i],s[:,i+1]])
        return sars_list

    def predict_future_reward_perfect(self):
        ## update the future reward perfectly ##
        ## -> see whether this makes the policy network converges ##
        x_left,y_down,x_right,y_up  = self.config.env.G1
        success_reward = self.config.env.G1_reward
        p1,p2,p3,p4 = [x_left,y_up], [x_right,y_up], [x_right,y_down], [x_left,y_down]
        d_p1 , d_p2 , d_p3 , d_p4 = self.convert_cState_to_dState(p1), self.convert_cState_to_dState(p2), \
                self.convert_cState_to_dState(p3) , self.convert_cState_to_dState(p4)
        G_i_left, G_j_up, G_i_right, G_j_down = d_p1[0], d_p1[1], d_p4[0] , d_p4[1]

        self.current_G = [G_i_left, G_i_right, G_j_down, G_j_up]

        ## let's restrict to four action ##
        if self.config.env.n_actions != 4 :
            raise NotImplementedError

        ## recall motions ##
        up = self.config.env.motions[0]
        right = self.config.env.motions[1]
        down = self.config.env.motions[2]
        left = self.config.env.motions[3]

        up_index , right_index , down_index, left_index = 0,1,2,3

        ## assign reward maunaully ##
        for i_idx in range(G_i_left,G_i_right+1) :
            if G_j_up != self.grid_size -1 :
                self.reward_table[i_idx, G_j_up + 1, down_index,-1] = success_reward
            if G_j_down != 0 :
                self.reward_table[i_idx, G_j_down - 1, up_index, -1] = success_reward

        for j_idx in range(G_j_down,G_j_up+1) :
            if G_i_right != self.grid_size -1 :
                self.reward_table[G_i_right+1,j_idx,left_index,-1] = success_reward
            if G_i_left != 0 :
                self.reward_table[G_i_left-1,j_idx,right_index,-1] = success_reward

        # last_reward = self.reward_table[:, :, :, -1]
        #
        # ## Goal point mark ##
        # if G_i_left == G_i_right :
        #     if G_j_down == G_j_up :
        #         last_reward[G_i_left,G_j_down,:] = success_reward + 2
        #     else :
        #         last_reward[G_i_left, G_j_down:G_j_up+1,:] = success_reward + 2
        # else :
        #     if G_j_down == G_j_up:
        #         last_reward[G_i_left:G_i_right+1, G_j_down,:] = success_reward + 2
        #     else:
        #         last_reward[G_i_left:G_i_right+1, G_j_down:G_j_up + 1,:] = success_reward + 2
        #
        #
        #
        # up_reward = self.modify_matrix_to_visualize_r(last_reward[:, :, 0])
        # right_reward = self.modify_matrix_to_visualize_r(last_reward[:, :, 1])
        # down_reward = self.modify_matrix_to_visualize_r(last_reward[:, :, 2])
        # left_reward = self.modify_matrix_to_visualize_r(last_reward[:, :, 3])
        #
        #
        #
        #
        # ## show the reward ##
        # if self.self_counter_for_drawing % 100 == 0 :
        #     plt.imshow(up_reward)
        #     plt.show()
        #     plt.imshow(right_reward)
        #     plt.show()
        #     plt.imshow(down_reward)
        #     plt.show()
        #     plt.imshow(left_reward)
        #     plt.show()
        #
        # self.self_counter_for_drawing +=1



    # def draw_r(self,reward_matrix,G_i_left, G_j_up, G_i_right, G_j_down):
        # if G_i_left == G_i_right :
        #     goal_i_range = [G_i_left]
        # else :
        #     goal_i_range = []
        # if G_j_up == G_j_down :
        #
        # reward_matrix[[G_i_left:G_i_right+1]]
        # plt.imshow(reward_matrix)


    def predict_future_reward_LS(self):
        ## least square model to forecast ##
        for i_sx in range(self.grid_size) :
            for i_sy in range(self.grid_size) :
                for i_a in range(self.n_action) :
                    for p in range(self.reward_function_reference_lag) :
                        Y = torch.unsqueeze(torch.tensor([self.reward_table[i_sx, i_sy, i_a,p] for p in range(self.reward_function_reference_lag)],requires_grad=False,device=self.config.device),dim=-1)
                        coff = torch.matmul(self.pseudo_inv_X, Y)
                        self.reward_table[i_sx, i_sy, i_a,-1] = coff[0]*(self.reward_function_reference_lag+1)+coff[1]

    def predict_future_reward_average(self):
        self.reward_table[:,:,:,-1] = torch.mean(self.reward_table[:,:,:,:-1],axis=-1)

    def predict_future_reward_AR(self):
        self.reward_table

    def optimize(self):
        self.update_transition_probability()
        ## should empty the future reward table ##
        self.reward_table[:, :, :, -1] = self.config.env.step_reward
        ###########################################
        self.predict_future_reward_perfect()

