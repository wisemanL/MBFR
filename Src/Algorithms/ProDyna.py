import numpy as np
import torch
from torch import tensor, float32
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, utils, Model
from Src.Algorithms import NS_utils
from Src.Algorithms.Extrapolator import OLS
import torch.nn.functional as F


"""

"""
class ProDyna(Agent):
    def __init__(self, config):
        super(ProDyna, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = NS_utils.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        # self.syntheticTrajectory_memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
        #                                      action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.grid_size = config.grid_size
        self.state_matrix = self._make_state_matrix()

        self.KLloss = 0
        self.state_dim = config.env.observation_space.shape[0]
        self.n_action = config.env.n_actions

        self.SARS_discrete_memory = utils.SARS_Buffer(buffer_size=config.buffer_size, state_dim = self.state_dim, action_dim = self.action_size, atype=self.atype, config=config)

        self.Q_discrete = utils.Q_table(config=self.config,n_action=self.config.env.n_actions)
        self.V_discrete = utils.V_table(self.config)

        self.modules = [('actor', self.actor), ('state_features', self.state_features)]
        self.counter = 0
        self.init()

    def reset(self):
        super(ProDyna, self).reset()
        # self.SARS_discrete_memory.next()
        self.counter += 1
        self.gamma_t = 1

    def convert_dState_to_random_cState(self,state):
        return Model.model.convert_dState_to_random_cState(state,self.config)

    def convert_dState_to_specific_cState(self,state):
        return Model.model.convert_dState_to_specific_cState(state,self.config)

    def get_action(self, state):
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.actor.get_action_w_prob_dist(state)

        return action, prob, dist

    def _make_state_matrix(self):
        state_matrix = torch.zeros((self.grid_size, self.grid_size, self.state_dim),requires_grad=False, device = self.config.device)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                continuous_state = self.convert_dState_to_specific_cState(torch.tensor([[i, j]]))
                continuous_state = continuous_state.squeeze()
                state_matrix[i, j, :] = continuous_state

        return state_matrix

    def get_action_prob_of_all_discrete_states(self):
        B, H, D = self.state_matrix.shape
        s_feature = self.state_features.forward(self.state_matrix.view(B*H,-1))
        dist_all = self.actor.get_onlyProb(s_feature)
        dist_all = dist_all.view(B,H,-1)

        ###### detach the gradient flow ######
        dist_all = dist_all.detach()
        ######################################

        ### check whether view works well ###
        # action_prob_matrix = torch.zeros((self.grid_size,self.grid_size,self.n_action),requires_grad=False, device = self.config.device)
        # for i in range(self.grid_size):
        #     for j in range(self.grid_size):
        #         discrete_s = torch.tensor([[i, j]])
        #         continuous_s = self.convert_dState_to_specific_cState(discrete_s)
        #         s_feature = self.state_features.forward(continuous_s)
        #         action_prob = torch.squeeze(self.actor.get_onlyProb(s_feature))
        #         action_prob_matrix[i,j,:] = action_prob
        #
        # assert dist_all.all() == action_prob_matrix.all()

        return dist_all

    def update(self, s1_discrete, a1, f_r1, s2_discrete, action_prob,q_table_minus_mean = False):
        # Batch episode history
        self.SARS_discrete_memory.add(s1_discrete, a1, f_r1, s2_discrete)
        howmuchqupdate = self.update_Qtable(s1_discrete,a1,f_r1,s2_discrete)
        if q_table_minus_mean :
            self.Q_discrete.q_table_minus_mean()
        self.update_Vtable(action_prob)
        # self.gamma_t *= self.config.gamma
        return howmuchqupdate
        # if done and self.counter % self.config.delta == 0:
        #     self.optimize()

    def reset_q_table(self):
        self.Q_discrete.reset_table()

    def update_Qtable(self,s,a,r,s_next):
        howmuchqupdate = self.Q_discrete.update_q(s,a,r,s_next)
        return howmuchqupdate

    def update_Vtable(self,action_prob):
        self.V_discrete.update_v_from_model_and_q(self.Q_discrete.q,action_prob)

    def change_inf_to_1(self,prob):
        indices_list = torch.isinf(prob).nonzero()
        ## change that speicific inf value to the 1 ##
        for index in indices_list:
            # prob[index[0], index[1]] = 1
            for i in range(self.n_action) :
                if i == index[1] :
                    prob[index[0], i] = 1
                else :
                    prob[index[0],i] = 0
            try:
                assert abs(torch.sum(prob[index[0], :]) - 1) < 1e-3
            except:
                raise NotImplementedError

        return prob

    def update_policy_MBPOstyle(self):
        if self.SARS_discrete_memory.size <= self.config.reward_function_reference_lag:
            return
        ## sample from the sythetic trajectory to update the policy
        sample_s,sample_a = self.SARS_discrete_memory.sample(self.config.sars_batchSize_for_policyUpdate)
        sample_s_continuous = self.convert_dState_to_specific_cState(sample_s)
        s_feature = self.state_features.forward(sample_s_continuous)
        _, pi_all = self.actor.get_prob (s_feature, sample_a) # pi_all : pi(*|s_t) \in R^{4*1} where s_t ~ sample s
        ## compute exp(Q(s_t,*) - V(s_t)) / Z(s_t)
        q_v = self.Q_discrete.q[sample_s[:,0],sample_s[:,1],:] - torch.unsqueeze(self.V_discrete.v[sample_s[:,0],sample_s[:,1]],dim=-1)*torch.ones(self.n_action)
        ## minus mean to avoid overflow
        # q_v_minusMean = q_v - torch.mean(q_v,dim=1,keepdim=True) * torch.ones(self.n_action)
        # exp_q_v_minusMean = torch.exp(q_v_minusMean)

        # exp_q_v = torch.exp(q_v)
        #
        # if torch.sum(torch.isinf(exp_q_v)) > 0 :
        #     exp_q_v = self.change_inf_to_1(exp_q_v)
        #
        # exp_q_v = torch.div(exp_q_v, torch.unsqueeze(torch.sum(exp_q_v, dim=1), dim=-1) * torch.ones(self.n_action))


        # exp_q_v_minusMean = torch.div(exp_q_v_minusMean , torch.unsqueeze(torch.sum(exp_q_v_minusMean,dim=1),dim=-1)*torch.ones(self.n_action))
        # exp_q_v = torch.div(exp_q_v, torch.unsqueeze(torch.sum(exp_q_v, dim=1), dim=-1) * torch.ones(self.n_action))

        exp_q_v_fromsoftmaxpytorch = F.softmax(q_v,dim=1)

        ## compute KL divergence between pi(*|s_t) and exp(Q(s_t,*) - V(s_t)) / Z(s_t)
        loss = utils.kl_divergence(pi_all,exp_q_v_fromsoftmaxpytorch)
        # loss_kl_pytorch = F.kl_div(torch.log(exp_q_v_fromsoftmaxpytorch),pi_all,reduction="batchmean")

        ## add entorpy to make the policy more explore ##
        loss = loss - self.config.entropy_alpha* utils.entropy(pi_all)


        self.KLloss = loss

        self.step(loss)

    def optimize(self):
        if self.syntheticTrajectory_memory.size <= self.config.fourier_k:
            # If number of rows is less than number of features (columns), it wont have full column rank.
            return

        batch_size = self.syntheticTrajectory_memory.size if self.syntheticTrajectory_memory.size < self.config.batch_size else self.config.batch_size

        # Compute and cache the partial derivatives w.r.t to each of the episodes
        self.extrapolator.update(self.syntheticTrajectory_memory.size, self.config.delta)

        # Inner optimization loop
        # Note: Works best with large number of iterations with small step-sizes.
        for iter in range(self.config.max_inner):
            id, s, a, beta, r, mask = self.syntheticTrajectory_memory.sample(batch_size)            # B, BxHxD, BxHxA, BxH, BxH, BxH

            ## dyna-style ## -> levine lecture12 page18
            #[1] observe next state and r to get the transition (s,a,r
            # tuple_sar = self.change_continuous_to_discrete(s,a,r)

            B, H, D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature = self.state_features.forward(s.view(B * H, D))           # BxHxD -> (BxH)xd

            # Get action probabilities
            log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a.view(B * H, -1))     # (BxH)xd, (BxH)xA
            log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH
            pi_a = torch.exp(log_pi)                                                         # (BxH)x1 -> BxH

            # Get importance ratios and log probabilities
            rho = (pi_a / beta).detach()                                        # BxH / BxH -> BxH

            # Forward multiply all the rho to get probability of trajectory
            for i in range(1, H):
                rho[:, i] *= rho[:, i-1]

            rho = torch.clamp(rho, 0, self.config.importance_clip)              # Clipped Importance sampling (Biased)
            rho = rho * mask                                                    # BxH * BxH -> BxH

            # Create importance sampled rewards
            returns = rho * r                                                   # BxH * BxH -> BxH

            # Reverse sum all the returns to get actual returns
            for i in range(H-2, -1, -1):
                returns[:, i] += returns[:, i+1]

            loss = 0
            log_pi_return = torch.sum(log_pi * returns, dim=-1, keepdim=True)   # sum(BxH * BxH) -> Bx1

            # Get the Extrapolator gradients w.r.t Off-policy terms
            # Using the formula for the full derivative, we can compute this first part directly
            # to save compute time.
            del_extrapolator = torch.tensor(self.extrapolator.derivatives(id), dtype=float32)  # Bx1

            # Compute the final loss
            loss += - 1.0 * torch.sum(del_extrapolator * log_pi_return)              # sum(Bx1 * Bx1) -> 1

            # Discourage very deterministic policies.
            if self.config.entropy_lambda > 0:
                if self.config.cont_actions:
                    entropy = torch.sum(dist_all.entropy().view(B, H, -1).sum(dim=-1) * mask) / torch.sum(mask)  # (BxH)xA -> BxH
                else:
                    log_pi_all = dist_all.view(B, H, -1)
                    pi_all = torch.exp(log_pi_all)                                      # (BxH)xA -> BxHxA
                    entropy = torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)

                loss = loss + self.config.entropy_lambda * entropy

            # Compute the total derivative and update the parameters.
            self.step(loss)

