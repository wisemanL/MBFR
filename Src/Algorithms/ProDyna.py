import numpy as np
import torch
from torch import tensor, float32
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, utils
from Src.Algorithms import NS_utils
from Src.Algorithms.Extrapolator import OLS

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

        self.state_dim = config.env.observation_space.shape[0]

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


    def get_action(self, state):
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.actor.get_action_w_prob_dist(state)

        return action, prob, dist


    def update(self, s1_discrete, a1, f_r1, s2_discrete,model_transition_prob):
        # Batch episode history
        self.SARS_discrete_memory.add(s1_discrete, a1, f_r1, s2_discrete)
        self.update_Qtable(s1_discrete,a1,f_r1,s2_discrete)
        self.update_Vtable(model_transition_prob)
        # self.gamma_t *= self.config.gamma

        # if done and self.counter % self.config.delta == 0:
        #     self.optimize()

    def update_Qtable(self,s,a,r,s_next):
        self.Q_discrete.update_q(s,a,r,s_next)

    def update_Vtable(self,model_transition_prob):
        self.V_discrete.update_v_from_model_and_q(self.Q_discrete.q,model_transition_prob)

    def change_continuous_to_discrete(self,s,a,r):
        ## divide trajectory (s^{i}_1,...s^{i}_H} into (s,a,r,s') where {i} \in {1,...,B} and B is number of trajectories sampled from the Batch. ##
        return 0,0,0


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
            tuple_sar = self.change_continuous_to_discrete(s,a,r)



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

