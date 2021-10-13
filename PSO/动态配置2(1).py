import numpy as np
import copy
from matplotlib import pyplot as plt
import time


M = 999
class Individual:
    def __init__(self, excessPointNum, lackPointNum, kindsNum, excessAll):
        # 变量, 运输量占比 (kindsNum, excessPointNum, lackPointNum)
        self.x = np.random.rand(kindsNum, excessPointNum, lackPointNum)\
                 * excessAll.reshape(kindsNum, excessPointNum, 1).repeat(lackPointNum, 2)
        # 变量, 点开放变量
        self.y = np.random.rand(excessPointNum)
        for i in range(excessPointNum):
            if self.y[i] < 0.5:
                self.x[:, i, :] = 0

        # 速度
        self.vx = np.random.rand(kindsNum, excessPointNum, lackPointNum)
        self.vy = np.random.rand(excessPointNum)


class DynamicConfig:
    def __init__(self,
                 kindsNum,
                 excessPointNum,
                 lackPointNum,
                 timeMatrix,
                 excessConfig,
                 lackConfig,
                 excessDemand,
                 lackDemand,
                 storageCost,
                 transCost,
                 openCost,
                 timeCost,
                 popNum,
                 iterMax,
                 omega,
                 c1,
                 c2,
                 w1,
                 w2,
                 p1,
                 p2):
        """
        :param kindsNum:        货物种数
        :param excessPointNum:  配置过量点数
        :param lackPointNum:    配置不足点数
        :param timeMatrix:      行驶时间矩阵      (excessPointNum, lackPointNum)
        :param excessConfig:    配置过量点储备量   (kindsNum, excessPointNum)
        :param lackConfig:      配置不足点储备量   (kindsNum, lackPointNum)
        :param excessDemand:    配置过量点需求量   (kindsNum, excessPointNum)
        :param lackDemand:      配置不足点需求量   (kindsNum, lackPointNum)
        :param storageCost:     储备成本          (kindsNum, excessPointNum + lackPointNum)
        :param transCost:       运输成本          (kindsNum, excessPointNum, lackPointNum)
        :param openCost:        开放成本          (excessPointNum, )
        :param timeCost:        单位装货时间成本   (kindsNum, excessPointNum)

        :param popNum:          种群规模
        :param iterMax:         最大迭代次数
        :param omega:           惯性因子
        :param c1:              个体学习因子
        :param c2:              社会学习因子

        :param w1:              目标函数权重1
        :param w2:              目标函数权重2

        :param p1:              惩罚项权重1
        :param p2:              惩罚项权重2
        """
        assert excessConfig.shape == excessDemand.shape == (kindsNum, excessPointNum)\
               and lackConfig.shape == lackDemand.shape == (kindsNum, lackPointNum)\
               and timeMatrix.shape == (excessPointNum, lackPointNum) \
               and storageCost.shape == (kindsNum, excessPointNum + lackPointNum) \
               and transCost.shape == (kindsNum, excessPointNum, lackPointNum)\
               and openCost.shape == (excessPointNum, )\
               and timeCost.shape == (kindsNum, excessPointNum), \
            "Input Data Error"
        self.kindsNum = kindsNum
        self.excessPointNum = excessPointNum
        self.lackPointNum = lackPointNum
        self.timeMatrix = timeMatrix
        self.excessConfig = excessConfig
        self.lackConfig = lackConfig
        self.excessDemand = excessDemand
        self.lackDemand = lackDemand
        self.storageCost = storageCost
        self.transCost = transCost
        self.openCost = openCost
        self.timeCost = timeCost

        # 过量物资量
        self.excessAll = excessConfig - excessDemand
        # 不足物资量
        self.lackAll = lackDemand - lackConfig

        self.popNum = popNum
        self.iterMax = iterMax
        self.omega = omega
        self.c1 = c1
        self.c2 = c2

        # 目标函数权重
        self.w1 = w1
        self.w2 = w2

        # 惩罚项权重
        self.p1 = p1
        self.p2 = p2
        
#个体初始化
    def evaluateInd(self, individual):
        Z1 = np.sum(self.openCost * individual.y) + np.sum(individual.x * self.transCost * self.timeMatrix)
        Z2 = np.max(individual.x *\
                    self.timeCost.reshape(self.kindsNum, self.excessPointNum, 1).repeat(self.lackPointNum, 2) +\
                    self.timeMatrix)

        penalty1 = np.sum((self.lackAll - individual.x.sum(1)) > 0)
        penalty2 = np.sum((individual.x.sum(2) - self.excessAll) > 0)
        cost = self.w1 * Z1 + self.w2 * Z2 + self.p1 * penalty1 * M + self.p2 * penalty2 * M
        # cost = penalty1 * M + penalty2 * M
        return cost

#群体初始化
    def evaluatePop(self, pop):
        fits = np.zeros(self.popNum)
        for p in range(self.popNum):
            fits[p] = self.evaluateInd(pop[p])
        return fits

#个体适应度
    def initIndividual(self):
        individual = Individual(self.excessPointNum, self.lackPointNum, self.kindsNum, self.excessAll)

        return individual

#群体适应度
    def initPop(self):
        pop = []
        for i in range(self.popNum):
            pop.append(self.initIndividual())
        return pop

#粒子速度、位置更新
    def updatePop(self, initPop):
        pop = initPop
        fits = self.evaluatePop(pop)
        pBestFits = fits.copy()
        pBest = pop.copy()
        gBestIdx = fits.argmin()
        gBestFit = fits[gBestIdx]
        gBestFits = [gBestFit]
        gBest = copy.deepcopy(pop[gBestIdx])
        start = time.time()
        for iter in range(self.iterMax):
            tempPop = []
            for p in range(self.popNum):
                tempInd = Individual(self.excessPointNum, self.lackPointNum, self.kindsNum, self.excessAll)

                tempInd.vx = self.omega * pop[p].vx\
                             + self.c1 * np.random.rand() * (pBest[p].x - pop[p].x) \
                             + self.c2 * np.random.rand() * (gBest.x - pop[p].x)
                tempInd.vy = self.omega * pop[p].vy \
                             + self.c1 * np.random.rand() * (pBest[p].y - pop[p].y) \
                             + self.c2 * np.random.rand() * (gBest.y - pop[p].y)

                tempInd.x = pop[p].x + tempInd.vx
                tempInd.y = pop[p].y + tempInd.vy

                for k in range(self.kindsNum):
                    for i in range(self.excessPointNum):
                        for j in range(self.lackPointNum):
                            if tempInd.x[k, i, j] < 0:
                                tempInd.x[k, i, j] = 0

                for i in range(self.excessPointNum):
                    if tempInd.y[i] > 1:
                        tempInd.y[i] = 1
                    if tempInd.y[i] < 0:
                        tempInd.y[i] = 0
                    if tempInd.y[i] < 0.5:
                        tempInd.x[:, i, :] = 0

                tempPop.append(copy.deepcopy(tempInd))

                fit = self.evaluateInd(tempInd)
                if fit < fits[p]:
                    pBestFits[p] = fit
                    pBest[p] = copy.deepcopy(tempInd)
                if fit < gBestFit:
                    gBestFit = fit
                    gBest = copy.deepcopy(tempInd)
                fits[p] = fit
            gBestFits.append(gBestFit)

        for k in range(self.kindsNum):
            print("第" + str(k+1) + "种物资:")
            for i in range(self.excessPointNum):
                for j in range(self.lackPointNum):
                    # if gBest.x[k, i, j] > 0:
                    print("从" + str(i+1) + "到" + str(j+1) + "运输量为：" + str(gBest.x[k, i, j]))

        stop = time.time()
        print('总时间(s)：',stop - start)
        plt.figure()
        plt.plot(gBestFits)
        plt.show()
        # return gBest


#  if __name__ == '__main__':
#     kindsNum = 3
#     excessPointNum = 2
#     lackPointNum = 3
#     timeMatrix = np.array([[0.8, 1, 1.8],
#                             [1.3, 1.1, 1.5]])
#     excessConfig = np.array([[4, 9],
#                               [4, 9],
#                               [4, 9]])
#     lackConfig = np.array([[1, 1.3, 1.6],
#                             [1, 1.3, 1.6],
#                             [1, 1.3, 1.6]])
#     excessDemand = np.array([[1.3, 2],
#                               [1.3, 2],
#                               [1.3, 2]])
#     lackDemand = np.array([[1.3, 2.6, 3],
#                             [1.3, 2.6, 3],
#                             [1.3, 2.6, 3]])
#     storageCost = np.array([[40, 40, 40, 40, 40],
#                             [40, 40, 40, 40, 40],
#                             [40, 40, 40, 40, 40]])
#     transCost = np.array([[[20, 20, 20],
#                           [20, 20, 20]],

#                           [[20, 20, 20],
#                             [20, 20, 20]],

#                           [[20, 20, 20],
#                             [20, 20, 20]]
#                           ])
#     openCost = np.array([20, 20])
#     timeCost = np.array([[20, 20],
#                           [20, 20],
#                           [20, 20]])


if __name__ == '__main__':
    kindsNum = 3
    excessPointNum = 9
    lackPointNum = 3
    #用过了
    timeMatrix = np.array([[15, 10, 28],
                            [33, 16, 24],
                            [40, 24, 29],
                            [35, 30, 25],
                            [21, 15, 12],
                            [16, 21, 19],
                            [13, 23, 24],
                            [18, 29, 22],
                            [24, 35, 29]])
    
    #用过了
    excessConfig = np.array([[81, 51, 81, 61, 91, 46, 66, 71, 61],
                              [5, 4, 9, 7, 6, 6, 9, 7, 4],
                              [7, 3, 4, 3, 3, 3, 4, 5, 4]])
    lackConfig = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
    excessDemand =np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    
    lackDemand = np.array([[101, 121, 91],
                            [13, 15, 19],
                            [8, 5, 6]])
    
    storageCost = np.array([[8, 12, 14, 11, 14, 13, 15, 16, 14, 17, 9, 11],
                            [26, 23, 30, 28, 35, 27, 28, 24, 22, 34, 28, 29],
                            [40, 38, 31, 28, 29, 36, 38, 27, 28, 34, 29, 35]])
    
    transCost = np.array([[[5, 5, 5],
                            [6, 6, 6],
                            [7, 7, 7],
                            [7, 7, 7],
                            [4, 4, 4],
                            [5, 5, 5],
                            [5, 5, 5],
                            [7, 7, 7],
                            [6, 6, 6]],

                          [[5, 5, 5],
                            [6, 6, 6],
                            [7, 7, 7],
                            [7, 7, 7],
                            [4, 4, 4],
                            [5, 5, 5],
                            [5, 5, 5],
                            [7, 7, 7],
                            [6, 6, 6]],

                          [[5, 5, 5],
                            [6, 6, 6],
                            [7, 7, 7],
                            [7, 7, 7],
                            [4, 4, 4],
                            [5, 5, 5],
                            [5, 5, 5],
                            [7, 7, 7],
                            [6, 6, 6]]
                          ])
    
    openCost = np.array([250, 300, 280, 420, 400, 380, 300, 440, 320])
    timeCost = np.array([[0.3, 0.5, 0.5, 0.6, 0.4, 0.5, 0.6, 0.3, 0.5],
                          [4, 4, 3, 5, 3, 4, 4, 6, 5],
                          [7, 8, 6, 6, 8, 9, 5, 5, 5]])
    popNum = 20
    iterMax = 500
    omega = 0.5
    c1 = 1.5
    c2 = 1.5

    w1 = 0.5
    w2 = 0.5
    p1 = 1
    p2 = 1

    """
        :param kindsNum:        货物种数
        :param excessPointNum:  配置过量点数
        :param lackPointNum:    配置不足点数
        :param timeMatrix:      行驶时间矩阵      (excessPointNum, lackPointNum)
        :param excessConfig:    配置过量点储备量   (kindsNum, excessPointNum)
        :param lackConfig:      配置不足点储备量   (kindsNum, lackPointNum)
        :param excessDemand:    配置过量点需求量   (kindsNum, excessPointNum)
        :param lackDemand:      配置不足点需求量   (kindsNum, lackPointNum)
        :param storageCost:     储备成本          (kindsNum, excessPointNum + lackPointNum)
        :param transCost:       运输成本          (kindsNum, excessPointNum, lackPointNum)
        :param openCost:        开放成本          (excessPointNum, )
        :param timeCost:        单位装货时间成本   (kindsNum, excessPointNum)

        :param popNum:          种群规模
        :param iterMax:         最大迭代次数
        :param omega:           惯性因子
        :param c1:              个体学习因子
        :param c2:              社会学习因子

        :param w1:              目标函数权重1
        :param w2:              目标函数权重2

        :param p1:              惩罚项权重1
        :param p2:              惩罚项权重2
    """

    dc = DynamicConfig(kindsNum,
                 excessPointNum,
                 lackPointNum,
                 timeMatrix,
                 excessConfig,
                 lackConfig,
                 excessDemand,
                 lackDemand,
                 storageCost,
                 transCost,
                 openCost,
                 timeCost,
                 popNum,
                 iterMax,
                 omega,
                 c1,
                 c2,
                 w1,
                 w2,
                 p1,
                 p2)
    initPop = dc.initPop()
    dc.updatePop(initPop)







