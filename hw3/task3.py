import numpy as np
import matplotlib.pyplot as plt
import pdb

def find_S_t(data):
	N = data.shape[0]
	x_i = data[:,[0]]
	y_i = data[:,[1]]
	x_i_comma = data[:,[2]]
	y_i_comma = data[:,[3]]
	#breakpoint()
	A = np.zeros((2*N,6))
	A[0:N,[0]] = x_i
	A[0:N,[1]] = y_i
	A[0:N, [4]] = np.ones((N,1))
	A[N:,[2]] = x_i
	A[N:,[3]] = y_i
	A[N:, [5]] = np.ones((N,1))
	#breakpoint()
	b = np.vstack((x_i_comma,y_i_comma))

	v = np.linalg.lstsq(A,b, rcond=None)[0]
	v = v.reshape(v.size)

	S = np.array([[v[0], v[1]], [v[2],v[3]]])
	#breakpoint()
	t = np.array([ [v[4]], [v[5]] ])
	#breakpoint()
	print("S = \n", S)
	print("t = \n", t)

	return S, t



data_case1 = np.load('./task3/points_case_1.npy')
data_case2 = np.load('./task3/points_case_2.npy')
#part(a)
S, t = find_S_t(data_case1)

#part(b)
x_i = data_case1[:,[0]]
y_i = data_case1[:,[1]]
x_i_comma = data_case1[:,[2]]
y_i_comma = data_case1[:,[3]]
xy = np.hstack((x_i,y_i))
optimal = S.dot(xy.T) + t

plt.scatter(x_i, y_i, 1, c="red")
plt.scatter(x_i_comma, y_i_comma,1, c="green")
plt.scatter(optimal[0,:], optimal[1,:], 0.01, c="blue")
plt.savefig("3_b_case1.png")
plt.close()

S, t = find_S_t(data_case2)
x_i = data_case2[:,[0]]
y_i = data_case2[:,[1]]
x_i_comma = data_case2[:,[2]]
y_i_comma = data_case2[:,[3]]
xy = np.hstack((x_i,y_i))
optimal = S.dot(xy.T) + t
plt.scatter(x_i, y_i, 1, c="red")
plt.scatter(x_i_comma, y_i_comma,1, c="green")
plt.scatter(optimal[0,:], optimal[1,:], 0.01, c="blue")
plt.savefig("3_b_case2.png")
#plt.show()








