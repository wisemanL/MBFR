import numpy as np
import matplotlib.pyplot as plt

folder = "algo_ProDyna_ep_5000_speed_0_howManyChange_0"

v_history = np.load(folder+"/2_v_history.npy")
q_history = np.load(folder+"/2_q_history.npy")

## the grid is divided into two parts...
# let's take a look at V_history [4:5,4:5] and v_history[4,5]
one = v_history[5,5,:]
two = v_history[4,5,:]
three = v_history[4,4,:]
four = v_history[5,4,:]

Q_value_initialstate = q_history[5,5,:,:]
up_q = Q_value_initialstate[0,:]
right_q = Q_value_initialstate[1,:]
down_q = Q_value_initialstate[2,:]
left_q = Q_value_initialstate[3,:]

plt.figure(1)
plt.plot(one,label="1")
plt.plot(two,label="2")
plt.plot(three,label="3")
plt.plot(four,label="4")
plt.legend()

plt.figure(2)
plt.plot(up_q,label="up")
plt.plot(right_q,label="right")
plt.plot(down_q,label="down")
plt.plot(left_q,label="left")
plt.legend()

plt.show()



print(1)