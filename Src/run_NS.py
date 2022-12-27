#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
import sys
sys.path.insert(0, '../')

import numpy as np
import Src.Utils.utils as utils
from Src.NS_parser import Parser
from Src.config import Config
from time import time
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from Src.Utils.Model import  model

import random



class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]
        self.epsilon_exploration = 0.1
        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.agent = config.algo(config=config)
        self.model = model(config=config)
        self.draw_trajectory = False

        self.action_prob_history = torch.zeros((self.config.grid_size, self.config.grid_size, self.config.env.n_actions,self.config.max_episodes), requires_grad=False, device=config.device)

    # def train_ProOLS_ProWLS(self):
    #     return_history = []
    #     true_rewards = []
    #     action_prob = []
    #
    #     ckpt = self.config.save_after
    #     rm_history, regret, rm, start_ep = [], 0, 0, 0
    #     gradient_norm_list = []
    #
    #     self.config.paths['results'] = self.config.paths['results'] + "algo_" + str(
    #         self.config.algo_name) + "_ep_" + str(self.config.max_episodes) + "_speed_" + str(self.config.speed) +"_gridsize_" + str(self.config.grid_size) \
    #                                    +"_rolloutEp_" + str(self.config.max_episodes_syntheticTrajectory) + "_rolloutStep_"+str(self.config.max_step_syntheticTrajectory) \
    #                                    +"_gradientStep_"+str(self.config.gradient_step)+"/"
    #     if not os.path.exists(self.config.paths['results']):
    #         os.mkdir(self.config.paths['results'])
    #
    #     self.writer = SummaryWriter(self.config.paths['results']+'tensorboard_logdir/')
    #
    #     steps = 0
    #     t0 = time()
    #     for episode1 in range(start_ep, self.config.max_episodes):
    #         # Reset both environment and model before a new episode
    #         state = self.env.reset()
    #         state_list = [state]
    #         self.agent.reset()
    #
    #         step, total_r = 0, 0
    #         done = False
    #         while not done:
    #             # self.env.render(mode='human')
    #             action, extra_info, dist = self.agent.get_action(state)
    #             new_state, reward, done, info = self.env.step(action=action)
    #             state_list.append(new_state)
    #             self.agent.update(state, action, extra_info, reward, new_state, done)
    #             state = new_state
    #
    #             # Tracking intra-episode progress
    #             total_r += reward
    #             # regret += (reward - info['Max'])
    #             step += 1
    #             if step >= self.config.max_steps:
    #                 break
    #
    #         # track inter-episode progress
    #         # returns.append(total_r)
    #         steps += step
    #         # rm = 0.9*rm + 0.1*total_r
    #         rm += total_r
    #         self.writer.add_scalar("reward_episode/train", total_r, episode1)
    #         self.writer.add_scalar("cumulative_reward/train", rm, episode1)
    #
    #         print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
    #               format(episode1, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.agent.entropy, self.agent.get_grads()))
    #
    #         if self.draw_trajectory :
    #             ## draw heatmap
    #             xs = [x[0] for x in state_list]
    #             ys = [x[1] for x in state_list]
    #             fig, ax = plt.subplots()
    #             ax.plot(xs,ys,"*--")
    #             ax.set_xlim(0,1)
    #             ax.set_ylim(0, 1)
    #             goal_coords = self.env.get_goal_area()
    #             x1, y1, x2, y2 = goal_coords
    #             w, h = x2-x1, y2-y1
    #             ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=True, color='gray'))
    #             if self.env.success :
    #                 fig.suptitle("success")
    #             fig.savefig(self.config.paths['results']+"trajectory_episode" + str(episode1).zfill(4) + ".png")
    #             plt.close()


    def train(self):
        # Learn the agent on the environment
        return_history = []
        true_rewards = []
        action_prob = []

        sars_dic = {}

        # ckpt = self.config.save_after
        self.config.paths['results'] = self.config.paths['results'] + "algo_" + str(
            self.config.algo_name) + "_ep_" + str(self.config.max_episodes) + "_speed_" + str(self.config.speed) +"_gridsize_" + str(self.config.grid_size) \
                                       +"_rolloutEp_" + str(self.config.max_episodes_syntheticTrajectory) + "_rolloutStep_"+str(self.config.max_step_syntheticTrajectory) \
                                       +"_gradientStep_"+str(self.config.gradient_step)+ "_entropyAlpha"+str(self.config.entropy_alpha)+ "_sarsBatchSize_"+str(self.config.sars_batchSize_for_policyUpdate)+"/"
        if not os.path.exists(self.config.paths['results']):
            os.mkdir(self.config.paths['results'])

        self.writer = SummaryWriter(self.config.paths['results']+'tensorboard_logdir/')

        ckpt = 1
        rm_history, regret, rm, start_ep, G1_history,gradient_norm_history = [], 0, 0, 0 , [],[]
        steps = 0
        t0 = time()
        for episode1 in range(start_ep, self.config.max_episodes):
            # Reset both environment and agent before a new episode
            state = self.env.reset()
            state_list = [state]
            sars_list = []
            self.agent.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False

            #### move reward table forward ####
            self.model.move_forward_reward_table(episode1)
            ###################################


            ### rollout real trajectory and save it ####
            while not done:
                action, extra_info, _ = self.agent.get_action(state) # get_action is stochastic policy. pick random action based on action_prob
                new_state, reward, done, info = self.env.step(action=action)
                sars_list.append([state,action,reward,new_state])
                state_list.append(new_state)
                self.model.update(state,action,extra_info,reward,new_state,done,episode1) # add reward table, update transition prob update , predict future reward
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break
            steps += step
            rm += total_r
            self.writer.add_scalar("reward_episode/train", total_r, episode1)
            self.writer.add_scalar("cumulative_reward/train", rm, episode1)
            ############################################

            ## update if the agent succss ##
            if self.env.success:
                self.model.realTrajectory_memory.add_trajectory_success()

            # if episode1 == 3000 :
            #     # reset the Q table
            #     self.agent.reset_q_table()
            #     # reset the policy network
            #     self.agent.actor.fc1.weight.data =torch.tensor([[-0.1683,  0.1939],
            #                                         [-0.0361,  0.3021],
            #                                         [ 0.1683, -0.0813],
            #                                         [-0.5717,  0.1614]], requires_grad=True)
            #     self.agent.actor.fc1.bias.data = torch.tensor([-0.6260,  0.0929,  0.0470, -0.1555], requires_grad=True)
            #     print("init!!")

            if episode1 >= self.config.reward_function_reference_lag :
                # print("rollout start")
                ## rollout based on the model
                sars_pair = []
                sars_pair.append(self.model.current_G)
                for episode2 in range(start_ep,self.config.max_episodes_syntheticTrajectory) :
                    # state2 = self.model.realTrajectory_memory.get_random_state_in_list(within_lag=True)
                    state2 ,p,s= self.model.realTrajectory_memory.get_success_random_state_in_list(within_lag_episode = True, within_few_latest_step = False)
                    for h in range(self.config.max_step_syntheticTrajectory) :
                        action2,_,_ = self.agent.get_action(state2)
                        state2_discrete = self.model.convert_cState_to_dState(state2)
                        new_state2_discrete, wherecomesfrom = self.model.get_next_state_from_model(state2,action2)
                        future_reward = self.model.get_future_reward_from_model(state2,action2,-1)
                        action_prob_allstates = self.agent.get_action_prob_of_all_discrete_states()
                        self.action_prob_history[:,:,:,episode1] = action_prob_allstates
                        # if future_reward == 6 :
                        #     print("========= success in rollout =========")
                        #     print(episode1)
                        #     print([state2_discrete,action2,future_reward,new_state2_discrete,wherecomesfrom])
                        #     print("======================================")
                        howmuchqupdate = self.agent.update(state2_discrete,action2,future_reward,new_state2_discrete,action_prob_allstates,q_table_minus_mean=False) # upddate Q,V
                        state2 = self.model.convert_dState_to_specific_cState(torch.unsqueeze(torch.tensor(new_state2_discrete),dim=0),self.config)
                        state2 = torch.squeeze(state2).tolist()

                        sars_pair.append([state2_discrete,action2,future_reward,new_state2_discrete,wherecomesfrom,howmuchqupdate])
                sars_dic[str(episode1)] = sars_pair

                # if episode1 > 500 and episode1 < 550:
                #     for l in sars_pair :
                #         print(l,sep='')


                ## update the policy ##
                for g in range(self.config.gradient_step) :
                    self.agent.update_policy_MBPOstyle()
                self.writer.add_scalar("KL loss/train",self.agent.KLloss,episode1)


                ## stack v and q ##
                self.agent.V_discrete.stack_v(episode1)
                self.agent.Q_discrete.stack_q(episode1)
                ####################

                if False :
                    ## visualize v and q ##
                    self.agent.V_discrete.visualize_v(self.config,episode1,save_fig=True)
                    self.agent.V_discrete.visualize_v_3d(self.config,episode1,save_fig=True)
                    self.agent.Q_discrete.visualize_q(episode1,save_fig=True)
                    #######################
            if episode1%ckpt == 0 or episode1 == self.config.max_episodes-1:
                rm_history.append(rm)
                G1_history.append(self.env.G1)
                return_history.append(total_r)
                if len(self.agent.get_grads()[0]) == 0:
                    gradient_norm = 0
                else:
                    gradient_norm = np.linalg.norm(self.agent.get_grads()[0])
                gradient_norm_history.append(gradient_norm)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode1, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.agent.entropy, self.agent.get_grads()))

                t0 = time()
                steps = 0

            if self.draw_trajectory :
                ## draw heatmap
                xs = [x[0] for x in state_list]
                ys = [x[1] for x in state_list]
                fig, ax = plt.subplots()
                ax.plot(xs,ys,"*--")
                ax.set_xlim(0,1)
                ax.set_ylim(0, 1)
                goal_coords = self.env.get_goal_area()
                x1, y1, x2, y2 = goal_coords
                w, h = x2-x1, y2-y1
                ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=True, color='gray'))
                if self.env.success :
                    fig.suptitle("success")
                fig.savefig(self.config.paths['results']+"trajectory_episode" + str(episode1).zfill(4) + ".png")
                plt.close()



            if episode1 == self.config.max_episodes-1 :
                utils.save_plots(return_history, config=self.config, name='{seed}_return_history_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                            seed=self.config.seed,
                            algo=self.config.algo_name,
                            ep=self.config.max_episodes,
                            step=self.config.max_steps,
                            speed=self.config.speed,
                            actor_lr="{:e}".format(self.config.actor_lr),
                            entropy=self.config.entropy_lambda,
                            delta=self.config.delta
                        ))
                utils.save_plots(rm_history, config=self.config, name='{seed}_cum_return_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                            seed=self.config.seed,
                            algo=self.config.algo_name,
                            ep=self.config.max_episodes,
                            step=self.config.max_steps,
                            speed=self.config.speed,
                            actor_lr="{:e}".format(self.config.actor_lr),
                            entropy=self.config.entropy_lambda,
                            delta=self.config.delta
                        ))
                utils.save_plots(G1_history, config=self.config, name='{seed}_G1_history_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                            seed=self.config.seed,
                            algo=self.config.algo_name,
                            ep=self.config.max_episodes,
                            step=self.config.max_steps,
                            speed=self.config.speed,
                            actor_lr="{:e}".format(self.config.actor_lr),
                            entropy=self.config.entropy_lambda,
                            delta=self.config.delta
                        ))
                utils.save_plots(gradient_norm_history, config=self.config, name='{seed}_gradeint_norm_algo_{algo}_ep{ep}_step{step}_speed{speed}_actorlr{actor_lr}_entropylambda{entropy}_delta{delta}'.format(
                            seed=self.config.seed,
                            algo=self.config.algo_name,
                            ep=self.config.max_episodes,
                            step=self.config.max_steps,
                            speed=self.config.speed,
                            actor_lr="{:e}".format(self.config.actor_lr),
                            entropy=self.config.entropy_lambda,
                            delta=self.config.delta
                        ))
        if False :
            ## save v_history and q_history ##
            self.agent.V_discrete.save_v_history()
            self.agent.Q_discrete.save_q_history()
            ##################################


            ## save action prob ##
            action_prob_history = self.action_prob_history.numpy()
            np.save(self.config.paths['results'] + "2_action_prob_history.npy", action_prob_history)

            ## save Q update value ##
            import pickle
            with open(self.config.paths['results']+"2_sars_Qupdate.pkl", "wb") as fp:  # Pickling
                pickle.dump(sars_dic, fp)

        self.writer.flush()
        self.writer.close()

# @profile
def main(train=True, inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    # Use only on-policy method for oracle
    if args.oracle >= 0:
            args.algo_name = 'ONPG'

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)
    solver = Solver(config=config)

    # Training mode
    if train:
        if args.algo_name in ["ProOLS", "ProWLS"] :
            solver.train_ProOLS_ProWLS()
        elif args.algo_name == "ProDyna" :
            solver.train()

    print("Total time taken: {}".format(time()-t))

if __name__ == "__main__":
        main(train=True)

