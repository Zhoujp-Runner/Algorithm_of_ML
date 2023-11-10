/***********************************************************************
 * Copyright 2023 by Zhou Junping
 *
 * @file     main.cpp
 * @brief    BP算法主函数
 *
 * @details
 * 主函数文件
 * 最近修改日期：2023-11-10
 *
 * @author   Zhou Junping
 * @email    zhoujunpingnn@gmail.com
 * @version  1.0
 * @data     2023-11-06
 *
 */
#include <iostream>
#include <math.h>
#include "BP_network.hpp"
#include <fstream>
#include <string>
#include "BP_nn.hpp"

using namespace std;

int main() {
    srand(unsigned(time(NULL)));
    vector<int> nodes = {1,3,4,5,1};
    Network model(nodes);
    model.train();
    model.save_prediction();
    return 0;
}