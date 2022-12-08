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



class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]
        self.epsilon_exploration = 0.3
        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.agent = config.algo(config=config)
        self.model = model(config=config)
        self.draw_trajectory = True


    def train(self):
        # Learn the agent on the environment
        return_history = []
        true_rewards = []
        action_prob = []

        # ckpt = self.config.save_after
        self.config.paths['results'] = self.config.paths['results'] + "/algo_" + str(
            self.config.algo_name) + "_ep_" + str(self.config.max_episodes) + "_speed_" + str(self.config.speed) +"_howManyChange_" + str(self.config.howmanychange) + "/"
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

            if episode1 >= self.config.reward_function_reference_lag :
                print("rollout start")
                ## rollout based on the model
                for episode2 in range(start_ep,self.config.max_episodes_syntheticTrajectory) :
                    state2 = self.model.realTrajectory_memory.get_random_state_in_list(within_lag=True)
                    for h in range(self.config.max_step_syntheticTrajectory) :
                        action2,_,_ = self.agent.get_action(state2)
                        state2_discrete = self.model.convert_cState_to_dState(state2)
                        new_state2_discrete, future_reward = self.model.get_next_state_from_model(state2,action2), self.model.get_future_reward_from_model(state2,action2,-1)
                        action_prob_allstates = self.agent.get_action_prob_of_all_discrete_states()
                        self.agent.update(state2_discrete,action2,future_reward,new_state2_discrete,action_prob_allstates,q_table_minus_mean=False) # upddate Q,V
                ## update the policy ##
                for g in range(self.config.gradient_step) :
                    self.agent.update_policy_MBPOstyle()
                self.writer.add_scalar("KL loss/train",self.agent.KLloss,episode1)


                ## stack v and q ##
                self.agent.V_discrete.stack_v()
                self.agent.Q_discrete.stack_q()
                ####################


                ## visualize v and q ##
                self.agent.V_discrete.visualize_v(self.config,episode1,save_fig=True)
                # self.agent.V_discrete.visualize_v_with_action(self.config, episode1,self.agent, save_fig=True)
                self.agent.V_discrete.visualize_v_3d(self.config,episode1,save_fig=True)
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

        ## save v_history and q_history ##
        self.agent.V_discrete.save_v_history()
        self.agent.Q_discrete.save_q_history()
        ##################################

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
        solver.train()

    print("Total time taken: {}".format(time()-t))

if __name__ == "__main__":
        main(train=True)

