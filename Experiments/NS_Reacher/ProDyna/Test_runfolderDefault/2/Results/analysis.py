import numpy as np
import matplotlib.pyplot as plt
import pickle
folder = "algo_ProDyna_ep_1500_speed_1_howManyChange_0"

v_history = np.load(folder+"/2_v_history.npy")
q_history = np.load(folder+"/2_q_history.npy")

with open(folder+"/2_sars_Qupdate.pkl", "rb") as fp:   # Unpickling
	b = pickle.load(fp)
print("bring data")

for key,value in b.items() :
	print(key)
	for l in value :
		print(l, sep='')

for key in [str(x) for x in range(1000,1100)] :
	print(key)
	for l in b[key] :
		print(l, sep='')



exit()

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
