/***********************************************************************
 * Copyright 2023 by Zhou Junping
 *
 * @file     BP_nn.hpp
 * @brief    BP神经网络头文件
 *
 * @details
 * 全连接神经网络的BP算法实现。
 * 包括前向传播、反向传播、梯度更新。
 * 其中，梯度更新算法包括SGD, BGD, Adam, RMSprop
 * 最近修改日期：2023-11-13
 *
 * @author   Zhou Junping
 * @email    zhoujunping0413@qq.com
 * @version  1.0
 * @data     2023-11-07
 *
 */
# include <iostream>
# include <math.h>
# include <vector>
# include <fstream>
# include <stdlib.h>
# include <string>

using namespace std;

#define PI acos(-1)

double sigmoid(double x) {  // sigmoid激活函数
    return (1 / (1 + exp(-x)));
}

double sigmoid_det(double x) {  // sigmoid函数的导数
    return sigmoid(x) * (1 - sigmoid(x));
}

double relu_activate(double x) {  // relu激活函数
    if (x <= 0) return 0;
    return x;
}

double relu_activate_det(double x) {  // relu激活函数的导数
    if (x <= 0) return 0;
    return 1;
}

double tanh_active(double x) {  // tanh激活函数
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double tanh_active_det(double x) {  // tanh激活函数的导数
    return 1 - tanh(x) * tanh(x);
}

double activate(double x, string type) {
    if (type == "sigmoid") return sigmoid(x);
    if (type == "relu") return relu_activate(x);
    if (type == "tanh") return tanh_active(x);
    throw "该激活函数不是可选的激活函数之一，请从sigmoid, relu, tanh中选择";
}

double activate_det(double x, string type) {
    if (type == "sigmoid") return sigmoid_det(x);
    if (type == "relu") return relu_activate_det(x);
    if (type == "tanh") return tanh_active_det(x);
    throw "该激活函数不是可选的激活函数之一，请从sigmoid, relu, tanh中选择";
}

struct Layer {
    vector<vector<double>> weights;  // 权重（上一层连到当前层）
    vector<double> bias;  // 偏置
    vector<double> input;  // 上一层的输出乘上对应权重求和后的结果（未经过激活函数）
    vector<double> out;  // 经过激活函数后当前层的输出
    vector<double> error;  // 当前层的误差
    vector<vector<double>> weights_grad;  // 权重的梯度
    vector<double> bias_grad;  // 偏置的梯度
    // Adam所需的缓存值
    vector<vector<double>> weights_mt;
    vector<vector<double>> weights_vt;
    vector<double> bias_mt;
    vector<double> bias_vt;
    // RMSProp所需的缓存值
    vector<vector<double>> weights_r;
    vector<double> bias_r;
    int step;
    Layer(int in_dim, int out_dim, bool random_init_parameters = false) {
        if (random_init_parameters) {  // 随机初始化，参数值的范围是-1到1
            weights.resize(in_dim);
            for (int i = 0; i < in_dim; i++) {
                weights[i].resize(out_dim);
                for (int j = 0; j < out_dim; j++) {
                    weights[i][j] = -1.0 + (double(rand()) / double(RAND_MAX)) * 2.0;
                }
            }
            bias.resize(out_dim);
            for (int i = 0; i < out_dim; i++) {
                bias[i] = -1.0 + (double(rand()) / double(RAND_MAX)) * 2.0;
            }
        } else {  // 将参数初始化为1
            weights = vector<vector<double>>(in_dim, vector<double>(out_dim, 1));
            bias = vector<double>(out_dim, 1);
        }
        input = vector<double>(out_dim, 0);
        out = vector<double>(out_dim, 0);
        error = vector<double>(out_dim, 0);
        weights_grad = vector<vector<double>>(in_dim, vector<double>(out_dim, 0));
        bias_grad = vector<double>(out_dim, 0);
        weights_mt = vector<vector<double>>(in_dim, vector<double>(out_dim, 0));
        weights_vt = vector<vector<double>>(in_dim, vector<double>(out_dim, 0));
        bias_mt = vector<double>(out_dim, 0);
        bias_vt = vector<double>(out_dim, 0);
        weights_r = vector<vector<double>>(in_dim, vector<double>(out_dim, 0));
        bias_r = vector<double>(out_dim, 0);
        step = 0;
    }
};

class Network {
private:
    int layers_num;  // 隐藏层层数+1（隐藏层+输出层）
    vector<Layer> layers;  // 存储隐藏层和输出层
    vector<int> nodes_num_per_layer;  // 网络每一层的节点数，包括输入层
    // TODO 这里规定了网络中所有的激活函数都是相同的，可以考虑修改成自定义激活函数的位置
    string active_type;  // 网络使用的激活函数
    bool final_activate;  // 网络的输出层是否需要激活函数
    vector<double> data;  // 训练数据（在构造函数中构建）
    vector<double> label;  // 训练标签（在构造函数中构建）
    int data_size = 1000;  // 训练集的大小
    double lr;  // 学习率
    int epochs;  // 训练周期
    // Adam算法的参数
    double beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8;
    // RMSProp算法的参数
    double rou = 0.9, eps_rms = 1e-6;
public:
    /**
     * 构造函数
     * @param nodes 网络中每一层的节点数（包括输入层）
     * @param activate 网络使用的激活函数类型
     * @param final_layer_active 最后一层是否使用激活函数
     * @param learning_rate 学习率
     * @param e 训练周期
     * @param random_init_parameters 是否随机初始化网络参数
     */
    explicit Network(vector<int>& nodes,
                     string& activate,
                     bool final_layer_active = false,
                     double learning_rate = 0.03,
                     int e = 20000,
                     bool random_init_parameters = true) : active_type(activate), final_activate(final_layer_active),
                                                           lr(learning_rate), epochs(e) {
        // 网络的构建及参数的初始化
        nodes_num_per_layer = nodes;
        layers_num = int(nodes.size()) - 1;
        int in_dim = nodes_num_per_layer[0], out_dim = 0;
        for (int i = 1; i < nodes_num_per_layer.size(); i++) {
            out_dim = nodes_num_per_layer[i];
            layers.emplace_back(in_dim, out_dim, random_init_parameters);
            in_dim = out_dim;
        }
        // 创建训练数据以及标签，这里目的是在-2*PI到2*PI之间生成|sin(x)|的数据
        double range = 4.0 * PI;
        data = vector<double>(data_size);
        label = vector<double>(data_size);
        for (int i = 0; i < data_size; i++) {
            data[i] = -2.0 * PI + (range / double(data_size)) * double(i);
            label[i] = fabs(sin(data[i]));
        }
    }

    /**
     * 单样本前向计算过程
     * @param index 训练数据的索引
     */
    void forward(int index) {
        // 从输入层到第一层
        for (int i = 0; i < nodes_num_per_layer[1]; i++) {
            layers[0].input[i] = data[index] * get_weight(1, 0, i) + get_bias(1, i);
            layers[0].out[i] = activate(get_input(1, i), active_type);
        }

        // 各隐藏层和输出层
        for (int n = 2; n <= layers_num; n++) {
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {  // 当前层
                layers[n - 1].input[i] = 0;
                for (int j = 0; j < nodes_num_per_layer[n - 1]; j++) {  // 上一层
                    layers[n - 1].input[i] += get_output(n - 1, j) * get_weight(n, j, i);
                }
                layers[n - 1].input[i] += get_bias(n, i);
                // 激活函数
                if (final_activate) {
                    layers[n - 1].out[i] = activate(get_input(n, i), active_type);
                } else {
                    if (n == layers_num) {
                        layers[n - 1].out[i] = get_input(n, i);
                    } else {
                        layers[n - 1].out[i] = activate(get_input(n, i), active_type);
                    }
                }
            }
        }
    }
    /**
     * 单样本反向传播过程
     * @param index 训练数据的索引
     */
    void backward(int index) {
        // 计算输出层的误差
        if (final_activate) {
            layers[layers_num - 1].error[0] = (get_output(layers_num, 0) - label[index]) * activate_det(get_input(layers_num, 0), active_type);
        } else {
            layers[layers_num - 1].error[0] = (get_output(layers_num, 0) - label[index]);  // 假设最后一层没有损失函数
        }

        // 计算隐藏层的误差
        for (int n = layers_num - 1; n >= 1; n--) {
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {  // 当前层
                layers[n - 1].error[i] = 0;
                for (int j = 0; j < nodes_num_per_layer[n + 1]; j++) {  // 下一层
                    layers[n - 1].error[i] += get_weight(n + 1, i, j) * get_error(n + 1, j) * activate_det(get_input(n, i), active_type);
                }
            }
        }

    }

    /**
     * 批量样本的前向过程与反向传播
     * @param batch_id 当前批次的序号
     * @param batch_size 批次的样本量
     */
    void batch_forward_backward(int batch_id, int batch_size) {
        // 将梯度清0
        for (int n = 1; n <= layers_num; n++) {
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                layers[n - 1].bias_grad[j] = 0;
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    layers[n - 1].weights_grad[i][j] = 0;
                }
            }
        }
        for (int index = batch_id * batch_size; index < (batch_id + 1) * batch_size; index++) {

            forward(index);
            backward(index);

            // 对第一层的权重梯度求和
            for (int i = 0; i < nodes_num_per_layer[1]; i++) {
                layers[0].weights_grad[0][i] += get_error(1, i) * data[index];
            }
            // 对隐藏层的权重梯度求和
            for (int n = 2; n <= layers_num; n++) {
                for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                    for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                        layers[n - 1].weights_grad[i][j] += get_error(n, j) * get_output(n - 1, i);
                    }
                }
            }
            // 对全部层的偏置梯度求和
            for (int n = 1; n <= layers_num; n++) {
                for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                    layers[n - 1].bias_grad[i] += get_error(n, i) ;
                }
            }
        }

        // 将梯度除以batch_size
        for (int n = 1; n <= layers_num; n++) {
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                layers[n - 1].bias_grad[j] /= double(batch_size);
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    layers[n - 1].weights_grad[i][j] /= double(batch_size);
                }
            }
        }
    }

    /**
     * 单个样本的随机梯度下降
     * @param index 训练样本的索引
     */
    void update_sgd(int index) {
        // 更新第一层的权重
        for (int j = 0; j < nodes_num_per_layer[1]; j++) {
            layers[0].weights[0][j] -= lr * get_error(1, j) * data[index];
        }
        // 更新隐藏层的权重
        for (int n = 2; n <= layers_num; n++) {
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    layers[n - 1].weights[i][j] -= lr * get_error(n, j) * get_output(n - 1, i);
                }
            }
        }
        // 更新所有层的偏置
        for(int n = 1; n <= layers_num; n++) {
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                layers[n - 1].bias[i] -= lr * get_error(n, i);
            }
        }
    }

    /**
     * 单个样本点的Adam算法
     * @param index 训练样本的索引
     */
    void update_adam(int index) {
        // 更新第一层的权重
        layers[0].step += 1;  // 更新当前时刻
        for (int j = 0; j < nodes_num_per_layer[1]; j++) {
            double beta_1_t = pow(beta_1, layers[0].step);
            double beta_2_t = pow(beta_2, layers[0].step);
            double grad = get_error(1, j) * data[index];  // 获取当前参数的梯度
            layers[0].weights_mt[0][j] = get_weight_mt(1, 0, j) * beta_1 + (1 - beta_1) * grad;  // m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
            layers[0].weights_vt[0][j] = get_weight_vt(1, 0, j) * beta_2 + (1 - beta_2) * grad * grad;  // v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
            double mt_hat = get_weight_mt(1, 0, j) / (1 - beta_1_t);  // mt_hat = m_t / (1 - beta_1_t)
            double vt_hat = get_weight_vt(1, 0, j) / (1 - beta_2_t);  // vt_hat = v_t / (1 - beta_2_t)
            layers[0].weights[0][j] -= lr * mt_hat / (sqrt(vt_hat) + eps);  // sita_t = sita_{t-1} - alpha * mt_hat / (sqrt(vt_hat) + eps)
        }
        // 更新隐藏层的权重
        for (int n = 2; n <= layers_num; n++) {
            layers[n - 1].step += 1;
            double beta_1_t = pow(beta_1, layers[n - 1].step);
            double beta_2_t = pow(beta_2, layers[n - 1].step);
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    double grad = get_error(n, j) * get_output(n - 1, i);
                    layers[n - 1].weights_mt[i][j] = get_weight_mt(n, i, j) * beta_1 + (1 - beta_1) * grad;
                    layers[n - 1].weights_vt[i][j] = get_weight_vt(n, i, j) * beta_2 + (1 - beta_2) * grad * grad;
                    double mt_hat = get_weight_mt(n, i, j) / (1 - beta_1_t);
                    double vt_hat = get_weight_vt(n, i, j) / (1 - beta_2_t);
                    layers[n - 1].weights[i][j] -= lr * mt_hat / (sqrt(vt_hat) + eps);
                }
            }
        }
        // 更新所有层的偏置，由于step在更新权重时已经自增过了，所以这里不用再对step自增
        for (int n = 1; n <= layers_num; n++) {
            double beta_1_t = pow(beta_1, layers[n - 1].step);
            double beta_2_t = pow(beta_2, layers[n - 1].step);
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                double grad = get_error(n, i);
                layers[n - 1].bias_mt[i] = get_bias_mt(n, i) * beta_1 + (1 - beta_1) * grad;
                layers[n - 1].bias_vt[i] = get_bias_vt(n, i) * beta_2 + (1 - beta_2) * grad * grad;
                double mt_hat = get_bias_mt(n, i) / (1 - beta_1_t);
                double vt_hat = get_bias_vt(n, i) / (1 - beta_2_t);
                layers[n - 1].bias[i] -= lr * mt_hat / (sqrt(vt_hat) + eps);
            }
        }
    }

    /**
     * 批量梯度下降算法
     */
    void batch_sgd(){
        // 更新所有层的权重
        for (int n = 1; n <= layers_num; n++) {
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    layers[n - 1].weights[i][j] -= lr * get_weight_grad(n, i, j);
                }
            }
        }
        // 更新所有层的偏置
        for(int n = 1; n <= layers_num; n++) {
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                layers[n - 1].bias[i] -= lr * get_bias_grad(n, i);
            }
        }
    }

    /**
     * 批量Adam算法 \n
     * **注意：这里参照pytorch的Adam源码的计算方式，而不是直接使用论文中的更新公式**
     */
    void batch_adam() {
        // 更新所有层的权重
        for (int n = 1; n <= layers_num; n++) {
            layers[n - 1].step += 1;
            double beta_1_t = 1 - pow(beta_1, layers[n - 1].step);
            double beta_2_t = 1 - pow(beta_2, layers[n - 1].step);
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    double grad = get_weight_grad(n, i, j);
                    layers[n - 1].weights_mt[i][j] = get_weight_mt(n, i, j) * beta_1 + (1 - beta_1) * grad;
                    layers[n - 1].weights_vt[i][j] = get_weight_vt(n, i, j) * beta_2 + (1 - beta_2) * grad * grad;
                    double denom = sqrt(get_weight_vt(n, i, j)) / sqrt(beta_2_t) + eps;
                    double stepsize = lr / beta_1_t;
                    layers[n - 1].weights[i][j] -= stepsize * get_weight_mt(n, i, j) / denom;
                }
            }
        }
        // 更新所有层的偏置，由于step在更新权重时已经自增过了，所以这里不用再对step自增
        for (int n = 1; n <= layers_num; n++) {
            double beta_1_t = 1 - pow(beta_1, layers[n - 1].step);
            double beta_2_t = 1 - pow(beta_2, layers[n - 1].step);
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                double grad = get_bias_grad(n, i);
                layers[n - 1].bias_mt[i] = get_bias_mt(n, i) * beta_1 + (1 - beta_1) * grad;
                layers[n - 1].bias_vt[i] = get_bias_vt(n, i) * beta_2 + (1 - beta_2) * grad * grad;
                double denom = sqrt(get_bias_vt(n, i)) / sqrt(beta_2_t) + eps;
                double stepsize = lr / beta_1_t;
                layers[n - 1].bias[i] -= stepsize * get_bias_mt(n, i) / denom;
            }
        }
    }

    /**
     * 批量RMSprop算法
     */
    void batch_rmsprop() {
        // 更新所有层的权重
        for (int n = 1; n <= layers_num; n++) {
            for (int j = 0; j < nodes_num_per_layer[n]; j++) {  // 当前层
                for (int i = 0; i < nodes_num_per_layer[n - 1]; i++) {  // 上一层
                    double grad = get_weight_grad(n, i, j);
                    layers[n - 1].weights_r[i][j] = rou * layers[n - 1].weights_r[i][j] + (1 - rou) * grad * grad;
                    layers[n - 1].weights[i][j] -= lr * grad / sqrt(layers[n - 1].weights_r[i][j] + eps_rms);
                }
            }
        }
        // 更新所有层的偏置
        for (int n = 1; n <= layers_num; n++) {
            for (int i = 0; i < nodes_num_per_layer[n]; i++) {
                double grad = get_bias_grad(n, i);
                layers[n - 1].bias_r[i] = rou * layers[n - 1].bias_r[i] + (1 - rou) * grad * grad;
                layers[n - 1].bias[i] -= lr * grad / sqrt(layers[n - 1].bias_r[i] + eps_rms);
            }
        }
    }

    void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            cout << epoch << endl;
            shuffle_random();
            // 单样本进行训练
            for (int index = 0; index < data_size; index++) {
                forward(index);
                backward(index);
                update_sgd(index);
            }
            // 批量样本训练
            for (int id = 0; id < 10; id++) {
                batch_forward_backward(id, 100);
                batch_rmsprop();
            }
        }
    }

    /**
     * 再次使用-2*PI到2PI区间，将拟合结果保存为csv文件，查看拟合效果
     */
    void save_prediction() {
        ofstream prediction_file;
        prediction_file.open("D:\\Data Structures And Algorithms\\test.csv", ios::out | ios::trunc);
        for (int index = 0; index < data_size; index++) {
            forward(index);
            prediction_file << data[index] << ',' << get_output(layers_num, 0) << endl;
        }
        prediction_file.close();
    }

    /**
     * 保存原始的训练样本和标签
     */
    void save_source() {
        ofstream prediction_file;
        prediction_file.open("D:\\Data Structures And Algorithms\\test3.csv", ios::out | ios::trunc);
        for (int index = 0; index < data.size(); index++) {
            prediction_file << data[index] << ',' << label[index] << endl;
        }
        prediction_file.close();
    }

    /**
     * 打乱数据集
     */
    void shuffle_random() {
        for (int i = 0; i < 500; i++) {
            int j = (float(rand()) / float(RAND_MAX + 1)) * float(data_size);
            int k = (float(rand()) / float(RAND_MAX + 1)) * float(data_size);
            double temp = data[j];
            data[j] = data[k];
            data[k] = temp;
            temp = label[j];
            label[j] = label[k];
            label[k] = temp;
        }
    }

    inline double get_weight(int layer, int i, int j) {  // 第layer层中，上一层第i个节点连接到当前层第j个节点的权重值
        return layers[layer - 1].weights[i][j];
    }

    inline double get_bias(int layer, int j) {  // 第layer层中，第j个节点的偏置
        return layers[layer - 1].bias[j];
    }

    inline double get_input(int layer, int j) {  // 第layer层中，第j个节点的输入
        return layers[layer - 1].input[j];
    }

    inline double get_output(int layer, int j) {  // 第layer层中，第j个节点的输出
        return layers[layer - 1].out[j];
    }

    inline double get_error(int layer, int j) {  // 第layer层中，第j个节点的误差
        return layers[layer - 1].error[j];
    }

    inline double get_weight_mt(int layer, int i, int j) {  // 第layer层中，上一层第i个节点连接到当前层第j个节点的权重的mt
        return layers[layer - 1].weights_mt[i][j];
    }

    inline double get_weight_vt(int layer, int i, int j) {  // 第layer层中，上一层第i个节点连接到当前层第j个节点的权重的vt
        return layers[layer - 1].weights_vt[i][j];
    }

    inline double get_bias_mt(int layer, int j) {  // 第layer层中，第j个节点的偏置的mt
        return layers[layer - 1].bias_mt[j];
    }

    inline double get_bias_vt(int layer, int j) {  // 第layer层中，第j个节点的偏置的vt
        return layers[layer - 1].bias_vt[j];
    }

    inline double get_weight_grad(int layer, int i, int j) {  // 第layer层中，上一层第i个节点连接到当前层第j个节点的权重的梯度
        return layers[layer - 1].weights_grad[i][j];
    }

    inline double get_bias_grad(int layer, int j) {  // 第layer层中，第j个节点的偏置的梯度
        return layers[layer - 1].bias_grad[j];
    }

};