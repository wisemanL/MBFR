import matplotlib.pyplot as plt
import numpy as np

change_0_reward = np.load("./algo_ProOLS_ep_1500_speed_0/2_cum_return_algo_ProOLS_ep1500_step500_speed0_actorlr5.000000e-03_entropylambda0.1_delta1.npy")
change_3_reward = np.load("./algo_ProOLS_ep_1500_speed_4_howManyChange_3/2_cum_return_algo_ProOLS_ep1500_step500_speed4_actorlr5.000000e-03_entropylambda0.1_delta1.npy")
change_5_reward = np.load("./algo_ProOLS_ep_1500_speed_4_howManyChange_5/2_cum_return_algo_ProOLS_ep1500_step500_speed4_actorlr5.000000e-03_entropylambda0.1_delta1.npy")
change_7_reward = np.load("./algo_ProOLS_ep_1500_speed_4_howManyChange_7/2_cum_return_algo_ProOLS_ep1500_step500_speed4_actorlr5.000000e-03_entropylambda0.1_delta1.npy")




def plot_target_change_point(how_many_changes , reward_file,color):
    total_episode = len(reward_file)
    for i in range(how_many_changes):
        if i == 0:
            pass
        else:
            plt.plot(i * int(total_episode / how_many_changes), reward_file[i * int(total_episode / how_many_changes)], color=color, marker="v")



plt.figure(1)
plt.plot(change_0_reward,label="non-starionary")
plt.legend()
plt.title("cumulative reward")
plt.xlabel("episode")
plt.ylabel("cumulative reward")
plt.figure(2)
p = plt.plot(change_3_reward,color='r',label="rotate 3 times")
plot_target_change_point(3,change_3_reward,p[0].get_color())
p = plt.plot(change_5_reward,label="rotate 5 times")
plot_target_change_point(5,change_5_reward,p[0].get_color())
p = plt.plot(change_7_reward,label="rotate 8 times")
plot_target_change_point(7,change_7_reward,p[0].get_color())
plt.legend()
plt.title("cumulative reward")
plt.xlabel("episode")
plt.ylabel("cumulative reward")

plt.show()


