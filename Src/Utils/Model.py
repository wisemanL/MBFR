from Src.Utils import utils , Basis
from Src.Algorithms import NS_utils
import numpy as np
import torch
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

    def reset(self):
        self.realTrajectory_memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_future_reward_from_model(self,state,action,p):
        sx_d, sy_d = self.convert_cState_to_dState(state)
        return self.reward_table[sx_d,sy_d,action,p]

    def get_next_state_from_model(self,state,action):
        sx_d , sy_d = self.convert_cState_to_dState(state)
        before_reshape_next_state_prob = self.transition_prob[sx_d,sy_d,action,:,:].cpu().data.numpy()
        after_reshape_next_state_prob = self.transition_prob[sx_d,sy_d,action,:,:].cpu().reshape(-1).data.numpy()
        if np.sum(after_reshape_next_state_prob) == 1 :
            action_index = np.random.choice(np.arange(len(after_reshape_next_state_prob)), p=after_reshape_next_state_prob)
        elif np.sum(after_reshape_next_state_prob) == 0 :
            action_index = np.random.choice(self.n_action)
        (sx_next_d, sy_next_d) = np.unravel_index(np.ravel_multi_index((action_index,), after_reshape_next_state_prob.shape), before_reshape_next_state_prob.shape)
        return sx_next_d , sy_next_d


    def compute_pseudo_inverse_X(self):
        X = torch.zeros((self.reward_function_reference_lag, 2),requires_grad=False,device=self.config.device)
        for i in range(self.reward_function_reference_lag) :
            X[i,0] = i+1
            X[i,1] = 1

        X_transpose = torch.transpose(X,0,1)
        pseudo_inverse = torch.matmul(torch.linalg.inv(torch.matmul(X_transpose,X)),X_transpose)
        return pseudo_inverse

    def update_visit_count(self,s,a,s_next):
        # convert the continuous s,s_next to discrete value
        (sx_d, sy_d),(sx_next_d, sy_next_d) = self.convert_cState_to_dState(s), self.convert_cState_to_dState(s_next)
        self.transition_visit[sx_d, sy_d, a, sx_next_d, sy_next_d] += 1

    def update_reward_table(self,s,a,r):
        (sx_d, sy_d) = self.convert_cState_to_dState(s)
        if self.counter < self.reward_function_reference_lag :
            self.reward_table[sx_d,sy_d,a,self.counter] = r
        else :
            for i in range(self.reward_function_reference_lag) :
                self.reward_table[sx_d, sy_d, a, i-1] = self.reward_table[sx_d, sy_d, a, i]
            self.reward_table[sx_d, sy_d, a, self.reward_function_reference_lag] = r

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

    def update(self, s1, a1, prob, r1, s2, done):
        # save the trajectory to the D_{env}
        self.realTrajectory_memory.add(s1, a1, prob, self.gamma_t * r1)
        self.gamma_t *= self.config.gamma

        # update the visit count table
        self.update_visit_count(s1, a1, s2)

        # update the reward table
        self.update_reward_table(s1, a1, r1)

        if done and self.realTrajectory_memory.size >= self.reward_function_reference_lag:
            self.optimize()

    def convert_cState_to_dState(self,state):
        # x_batch,y_batch = state[:,[0]],state[:,[1]]
        x , y = state[0], state[1]
        upper_right , lower_left = \
            torch.tensor(self.config.env.observation_space.high,requires_grad=False,device=self.config.device) , \
            torch.tensor(self.config.env.observation_space.low,requires_grad=False,device=self.config.device)
        x_left , x_right , y_down , y_up = lower_left[0],upper_right[0] , lower_left[1] , upper_right[1]

        i,j = torch.floor((x-x_left) / (x_right - x_left) * self.grid_size) ,  torch.floor((y-y_down) / (y_up - y_down) * self.grid_size)
        return i.type(torch.long),j.type(torch.long)

    def divide_into_sars_list(self,s,a,r):
        assert s.size()[:-1] == a.size()[:-1]
        assert s.size()[:-1] == r.size()

        total_step = s.size()[1]
        sars_list = []
        for i in range (total_step-1) :
            sars_list.append([s[:,i],a[:,i],r[:,i],s[:,i+1]])
        return sars_list

    def predict_future_reward(self):
        ## least square model to forecast ##
        for i_sx in range(self.grid_size) :
            for i_sy in range(self.grid_size) :
                for i_a in range(self.n_action) :
                    for p in range(self.reward_function_reference_lag) :
                        Y = torch.transpose(torch.tensor([self.reward_buffer[i_sx, i_sy, i_a,p] for p in range(self.reward_function_reference_lag)],requries_gard=False,device=self.config.device),0,1)
                        self.reward_buffer[i_sx, i_sy, i_a,-1] = torch.matmul(self.pseudo_inv_X, Y)

    def optimize(self):
        # if self.realTrajectory_memory.size <= self.config.fourier_k:
        #     # If number of rows is less than number of features (columns), it wont have full column rank.
        #     return
        #batch_size = self.realTrajectory_memory.size if self.realTrajectory_memory.size < self.config.batch_size else self.config.batch_size
        self.update_transition_probability()
        self.predict_future_reward()



        # for iter in range(3) : #self.config.model_optimize):
            # id, s, a, beta, r, mask = self.realTrajectory_memory.sample(batch_size)
            # ## [1] divide the samples into the multiple (s,a,r,s')
            # sars_list = self.divide_into_sars_list(s,a,r)
            # ## [2] use sars_list to update the model and reward function
            # for sars_item in sars_list :
            #     (sx_d,sy_d) , a, r , (sx_next_d,sy_next_d)  = self.convert_cState_to_dState(sars_item[0]), sars_item[1], sars_item[2], self.convert_cState_to_dState(sars_item[3])
            #     for i in range(batch_size) :
            #         self.transition_visit[sx_d[i],sy_d[i],a[i],sx_next_d[i],sy_next_d[i]] +=1
            #         self.reward_table[sx_d[i],sy_d[i],a[i],id[i]] = r

            ## [3] predict the reward function


