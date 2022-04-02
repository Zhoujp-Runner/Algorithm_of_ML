import copy
from matplotlib import pyplot as plt
from matplotlib import animation


class Perceptron(object):
    def __init__(self, data):
        self.weights = [0, 0]
        self.bias = 0
        self.beta = 1
        self.recording = []
        self.data_s = copy.deepcopy(data)
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-1, 5), ylim=(-1, 5))
        self.line, = self.ax.plot([], [], color='green', lw=2)

    # update w and b
    def update(self, data):
        self.weights[0] += self.beta * data[1] * data[0][0]
        self.weights[1] += self.beta * data[1] * data[0][1]
        self.bias += self.beta * data[1]
        self.recording.append([copy.copy(self.weights), self.bias])

    # cost derivative
    def cost_cal(self, data):
        distance = 0
        for i in range(len(data[0])):
            distance += data[0][i] * self.weights[i]
        distance += self.bias
        distance *= data[1]
        return distance
        # return((data[0][i]*self.weights[i] for i in range(len(data[0])))+self.bias)

    # Stochastic Gradient Descent
    def sgd(self, data):
        flag = 1
        epoch = 0
        while flag > 0:
            flag = 0
            for data_i in data:
                if self.cost_cal(data_i) <= 0:
                    self.update(data_i)
                    print("realtime weights:", self.weights)
                    print("realtime bias:", self.bias)
                    flag += 1
            epoch += 1
        print("All points are correctly classified!")
        print("weights:", self.weights)
        print("bias:", self.bias)
        print("epoch:", epoch)
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.recording), init_func=self.init, interval=1000, repeat=True, blit=True)
        plt.show()

    #show the picture
    #############################
    def init(self):
        x_p = [self.data_s[i][0][0] for i in range(len(self.data_s)) if self.data_s[i][1] == 1]
        y_p = [self.data_s[i][0][1] for i in range(len(self.data_s)) if self.data_s[i][1] == 1]
        x_n = [self.data_s[i][0][0] for i in range(len(self.data_s)) if self.data_s[i][1] == -1]
        y_n = [self.data_s[i][0][1] for i in range(len(self.data_s)) if self.data_s[i][1] == -1]
        plt.scatter(x_p, y_p, color='red', marker='+')
        plt.scatter(x_n, y_n, color='blue', marker='+')
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        w = self.recording[i][0]
        b = self.recording[i][1]
        if w[1]*w[0] != 0:
            x0 = 7
            y0 = -(w[0]*x0 + b) / w[1]
            x1 = -7
            y1 = -(w[0]*x1 + b) / w[1]
            self.line.set_data([x0, x1], [y0, y1])
        elif w[1] == 0 and w[0] != 0:
            self.line.set_data([-b/w[0], -b/w[0]], [0, 1])
        elif w[0] == 0 and w[1] != 0:
            self.line.set_data([0, 1], [-b/w[1], -b/w[1]])
        # x1 = 0
        # y1 = -(w_[0]*x1 + b_) / w_[1]
        # label.set_text(w[i])
        # label.set_position([x1, y1])
        return self.line,
