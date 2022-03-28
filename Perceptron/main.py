import perceptron
from matplotlib import pyplot as plt
from matplotlib import animation

data = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
per = perceptron.Perceptron(data)
per.sgd(data)

# w = [[1, 2], [2, 2], [1, 1]]
# b = [3, 4, 5]
# x = [-1, 1]


# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.xlim((-1, 5))
# plt.ylim((-1, 5))
# line, = ax.plot([], [], color='green', lw=2)
# label = ax.text([], [], '')


# def init():
#     line.set_data([], [])
#     x, y, x_, y_ = [], [], [], []
#     x = [data[i][0][0] for i in range(len(data)) if data[i][1] == 1]
#     y = [data[i][0][1] for i in range(len(data)) if data[i][1] == 1]
#     x_ = [data[i][0][0] for i in range(len(data)) if data[i][1] == -1]
#     y_ = [data[i][0][1] for i in range(len(data)) if data[i][1] == -1]
#     plt.plot(x, y, 'bo', x_, y_, 'rx')
#     plt.grid(True)
#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.title('hhh')
#     return line, label
#
#
# def animate(i):
#     w_ = w[i]
#     b_ = b[i]
#     x0 = 7
#     y0 = -(w_[0]*x0 + b_)/w_[1]
#     x1 = -7
#     y1 = -(w_[0]*x1 + b_) / w_[1]
#     line.set_data([x0, y0], [x1, y1])
#     x1 = 0
#     y1 = -(w_[0]*x1 + b_) / w_[1]
#     label.set_text(w[i])
#     label.set_position([x1, y1])
#     return line, label
#
# ani = animation.FuncAnimation(fig,animate,frames =3,init_func=init,interval=1000,repeat=True,blit=True)
# plt.show()




# def init():
#     x_p = [data[i][0][0] for i in range(len(data)) if data[i][1] == 1]
#     y_p = [data[i][0][1] for i in range(len(data)) if data[i][1] == 1]
#     x_n = [data[i][0][0] for i in range(len(data)) if data[i][1] == -1]
#     y_n = [data[i][0][1] for i in range(len(data)) if data[i][1] == -1]
#     plt.scatter(x_p, y_p, color='red', marker='+')
#     plt.scatter(x_n, y_n, color='blue', marker='+')
#     line.set_data([], [])
#     return line,
#
#
# def animate(i):
#     x0 = 7
#     y0 = -(w[i][0] * x0 + b[i])/w[i][1]
#     x1 = -7
#     y1 = -(w[i][0] * x1 + b[i]) / w[i][1]
#     line.set_data([x0, y0], [x1, y1])
#     return line,
#
#
# anim = animation.FuncAnimation(fig=fig, func=animate, frames=3, init_func=init, interval=1000, repeat=True, blit=False)
# plt.show()



