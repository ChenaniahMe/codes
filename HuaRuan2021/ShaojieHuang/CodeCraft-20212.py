# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:13:12 2021

@author: 少杰
"""

import copy
import sys
#import numpy as np

serverInfos = dict() #(string, list) 服务器信息
serverInfosList = list() #获取性价比的帮助变量

vmInfos = dict() # (string, list(int)) 虚拟机信息

requestInfos = list() # list[list<string>] 请求信息

# 购买的服务器信息
serverNumber = 0
sysServerResource = dict() #(int, list(int)) (服务器id， 【cpuA，cpuA, coreA, coreB, cost, loss】)

# 当前开机服务器
serverRunVms = list() #int 索引为服务器id，值为该服务器上虚拟机的数量，初始为0
capacity = list()
# 记录虚拟机运行在那个服务器上
vmOnServer = dict() # (string, list(int))

serverToBuy = dict() #serverTpye -> num
purchasestr = ''
opstr = list()

res = list() # string

SERVERCOST = 0
POWERCOST=0
TOTALCOST =0

is_test = True


requestdays = -1
serverTypeChosenOne = ''


def bugServer():
    serverType = "hostUY41I"
    n = 2500
    global serverRunVms
    global serverNumber
    global SERVERCOST
    global serverInfos
    global sysServerResource
    serverRunVms = [0 for _ in range(n)]
    initBuy = "(purchase, "
    initBuy += str(2)+")\n"

    res.append(initBuy)

    pauseInfo ="("+serverType+", "
    pauseInfo += str(n//2)+")\n"

    res.append(pauseInfo)

    for i in range(n//2):
        sysServerResource[serverNumber] = copy.deepcopy(serverInfos[serverType])
        serverNumber += 1
        SERVERCOST += serverInfos[serverType][4]

    serverType = "host78BMY"
    pauseInfo ="("+serverType+", "
    pauseInfo = pauseInfo + str(serverNumber)+")\n"

    res.append(pauseInfo)

    for i in range(n//2):
        sysServerResource[serverNumber] = copy.deepcopy(serverInfos[serverType])
        serverNumber += 1
        SERVERCOST += serverInfos[serverType][4]
        
def serverXingjiabi(coreNum,Memory,cost,loss,currentday,totalday):
    xingjiabi = (cost + loss * (totalday-currentday)) / (coreNum + Memory)
    return xingjiabi

def updateXingjiabi(day):
    global serverInfosList
    global requestdays
    for lis in serverInfosList:
        lis.append(serverXingjiabi(lis[1],lis[2],lis[3],lis[4],day,requestdays))
    serverInfosList = sorted(serverInfosList, key=lambda x:x[5])
    
    
        
def purchase(reVmType,day):
    global serverInfos
    global serverNumber
    global serverToBuy
    global sysServerResource
    global SERVERCOST
    global serverInfosList
    global vmInfos
    global requestdays
    global serverTypeChosenOne
    global capacity
    
    #买最贵的
    vmCore = vmInfos[reVmType][0]
    vmMemory = vmInfos[reVmType][1]
    twoOrOne = vmInfos[reVmType][2]
    serverTypeChosenOne = ''
    if twoOrOne == 1:
        for server in serverInfosList:
            if server[1] >= vmCore and server[2] >= vmMemory:
                serverTypeChosenOne = server[0]
                break
    else:
        for server in serverInfosList:
            if (server[1] // 2) >= vmCore and (server[2] // 2) >= vmMemory:
                serverTypeChosenOne = server[0]
                break
            
    
    #serverType = 'hostVIHCW'
    #serverType = 'hostV05X4'
    #serverType = serverInfosList[0][0]
    sysServerResource[serverNumber] = copy.deepcopy(serverInfos[serverTypeChosenOne])
    #sysServerResource[serverNumber].append(0) #利用率 【cpuA，cpuA, coreA, coreB, cost, loss，liyonglv】
    serverNumber += 1
    serverRunVms.append(0)
    capacity.append(0)
    SERVERCOST += serverInfos[serverTypeChosenOne][4]
    
    if serverTypeChosenOne not in serverToBuy.keys():    
        serverToBuy[serverTypeChosenOne] = 1
    else:
        serverToBuy[serverTypeChosenOne] += 1
        
    

def expansion():
    global res
    s = '(purchase, 0)\n'
    res.append(s)

def migrate():
    s = '(migration, 0)\n'
    res.append(s)    

def match(day):
    global requestInfos
    global opstr
    global purchasestr
    global serverToBuy
    global res
    global serverTypeChosenOne
    
    """
    if day != 0:
        expansion()
    migrate()
    """
    #updateXingjiabi(day)
    for req in requestInfos:
        opType = 1 if len(req) == 3 else 0
        if opType == 1:
            reVmType = req[1]
            reVmId = req[2]
            while(True):
                resourceEnough = createVM(reVmType,reVmId,day)
                if resourceEnough == -1:
                    purchase(reVmType,day)
                else:
                    break
            assert resourceEnough != -1
        else:
            reVmId = req[1]
            delVM(reVmId)
    
    if serverToBuy:
        
        servertypenum = len(serverToBuy)
        tempstr = '(purchase, '+ (str(servertypenum)) + ')\n'
        res.append(tempstr)
        
        #res.append('(purchase, 1)\n')
        for key,val in serverToBuy.items():
            
            pauseInfo ="("+key+", "
            pauseInfo += str(val)+")\n"
            res.append(pauseInfo)
    else:
        res.append('(purchase, 0)\n')
    
    res.append('(migration, 0)\n')
    
    res.extend(opstr)
    
    serverToBuy.clear()
    opstr = []
     


def choseServer(vmCores, vmMemory, vmTwoNodes, serverId, reVmId):
    global vmOnServer
    global sysServerResource
    global res
    global opstr
    global capacity
    global serverInfos
    global serverTypeChosenOne
    #vmCores = vm[0]
    #vmMemory = vm[1]
    #vmTwoNodes = vm[2]
    serverCoreA = sysServerResource[serverId][0]
    serverCoreB = sysServerResource[serverId][1]
    serverMemoryA = sysServerResource[serverId][2]
    serverMemoryB = sysServerResource[serverId][3]
    
    if vmTwoNodes:
        needCores = vmCores // 2
        needMemory = vmMemory // 2
        if serverCoreA >= needCores and serverCoreB >= needCores and serverMemoryA >= needMemory and serverMemoryB >= needMemory:
            sysServerResource[serverId][0] -= needCores
            sysServerResource[serverId][1] -= needCores
            sysServerResource[serverId][2] -= needMemory
            sysServerResource[serverId][3] -= needMemory
            vmOnServer[reVmId] = [serverId, vmCores,vmMemory,1,2]
            capacity[serverId] = 0.5*((sysServerResource[serverId][0]+sysServerResource[serverId][1])/(serverInfos[serverTypeChosenOne][0]+serverInfos[serverTypeChosenOne][1]))\
                                +0.5*((sysServerResource[serverId][2]+sysServerResource[serverId][3])/(serverInfos[serverTypeChosenOne][2]+serverInfos[serverTypeChosenOne][3]))
            opstr.append('('+ str(serverId)+ ')\n')
            #res.append('('+ str(serverId)+ ')\n')
            return True
        else:
            return False
    elif serverCoreA >= vmCores and serverMemoryA >= vmMemory:
        sysServerResource[serverId][0] -= vmCores
        sysServerResource[serverId][2] -= vmMemory
        vmOnServer[reVmId] = [serverId, vmCores,vmMemory,1]
        capacity[serverId] = 0.5*((sysServerResource[serverId][0]+sysServerResource[serverId][1])/(serverInfos[serverTypeChosenOne][0]+serverInfos[serverTypeChosenOne][1]))\
                                +0.5*((sysServerResource[serverId][2]+sysServerResource[serverId][3])/(serverInfos[serverTypeChosenOne][2]+serverInfos[serverTypeChosenOne][3]))
        opstr.append('(' + str(serverId) + ', A)\n')
        #res.append('(' + str(serverId) + ', A)\n')
        return True
    elif serverCoreB >= vmCores and serverMemoryB >= vmMemory:
        sysServerResource[serverId][1] -= vmCores
        sysServerResource[serverId][3] -= vmMemory
        vmOnServer[reVmId] = [serverId, vmCores,vmMemory,2]
        capacity[serverId] = 0.5*((sysServerResource[serverId][0]+sysServerResource[serverId][1])/(serverInfos[serverTypeChosenOne][0]+serverInfos[serverTypeChosenOne][1]))\
                                +0.5*((sysServerResource[serverId][2]+sysServerResource[serverId][3])/(serverInfos[serverTypeChosenOne][2]+serverInfos[serverTypeChosenOne][3]))
        opstr.append('(' + str(serverId) + ', B)\n')
        #res.append('(' + str(serverId) + ', B)\n')
        return True
    
    return False

def delVM(vmId_):
    global vmOnServer
    global serverRunVms
    global sysServerResource
    global serverInfos
    global serverTypeChosenOne
    global capacity
    global serverTypeChosenOne
    #vmId_ = delVmInfo[1]
    vmInfo_ = vmOnServer[vmId_] #[serverId, vmCores,vmMemory,1,2]
    serverId_ = vmInfo_[0]

    serverRunVms[serverId_] -= 1 #减少id为serverid的服务器上的虚拟机数量

    if len(vmInfo_) == 5:
        cores = vmInfo_[1] // 2
        memory = vmInfo_[2] // 2
        sysServerResource[serverId_][0] += cores
        sysServerResource[serverId_][1] += cores
        sysServerResource[serverId_][2] += memory
        sysServerResource[serverId_][3] += memory
        capacity[serverId_] = 0.5*((sysServerResource[serverId_][0]+sysServerResource[serverId_][1])/(serverInfos[serverTypeChosenOne][0]+serverInfos[serverTypeChosenOne][1]))\
                                +0.5*((sysServerResource[serverId_][2]+sysServerResource[serverId_][3])/(serverInfos[serverTypeChosenOne][2]+serverInfos[serverTypeChosenOne][3]))
        
    else:
        cores = vmInfo_[1] 
        memory = vmInfo_[2] 
        if vmInfo_[3] == 1:
            sysServerResource[serverId_][0] += cores
            sysServerResource[serverId_][2] += memory
        else:
            sysServerResource[serverId_][1] += cores
            sysServerResource[serverId_][3] += memory
        capacity[serverId_] = 0.5*((sysServerResource[serverId_][0]+sysServerResource[serverId_][1])/(serverInfos[serverTypeChosenOne][0]+serverInfos[serverTypeChosenOne][1]))\
                                +0.5*((sysServerResource[serverId_][2]+sysServerResource[serverId_][3])/(serverInfos[serverTypeChosenOne][2]+serverInfos[serverTypeChosenOne][3]))

def serverPower():
    global serverRunVms
    global POWERCOST
    global sysServerResource
    for i in range(serverNumber):
        if serverRunVms[i] != 0:
            POWERCOST += sysServerResource[i][5]

def createVM(reqVmType, reVmId, day):
    global vmInfos
    global serverNumber
    global sysServerResource # 第i个服务器，list【cpu，core，cost，cost】
    global serverRunVms #list
    global capacity
    global requestdays

    vm = vmInfos[reqVmType]
    vmCores = vm[0]
    vmMemory = vm[1]
    vmTwoNodes = vm[2]
    
    """
    sortedServerIdList = list()
    if day < requestdays // 2:
        sortedServerIdList = range(serverNumber)
    else:
        sortedServerIdList = np.argsort(np.array(capacity))
    """

    success = -1
    for i in range(serverNumber):
    #sortedServerIdList = np.argsort(np.array(capacity))
    #for i in sorted(range(serverNumber),key=capacity.__getitem__):
        #server = copy.deepcopy(sysServerResource[i])
        if choseServer(vmCores, vmMemory, vmTwoNodes, i, reVmId):
            serverRunVms[i] += 1
            success = 1
            break
        assert (sysServerResource[i][0]>=0 and sysServerResource[i][1]>=0 and sysServerResource[i][2]>=0 and sysServerResource[i][3]>=0)
    return success

def output():
    global res
    for s in res:
        print(s, end='')
    sys.stdout.flush()

def main():

    global serverInfos
    global vmInfos
    global requestInfos
    global serverInfosList
    global requestdays

    if is_test:
        file = open('training-2.txt')
        N = file.readline()  # 代表多少台服务器
    else:
        N = input()
    sys.stdout.flush()

    # 服务器的配置
    for i in range(int(N)):
        #server = file.readline().split(',')
        if is_test:
            server = file.readline().split(',')
        else:
            server = input().split(',')
        sys.stdout.flush()
        # CPU     内存     成本      能耗成本
    
        serverInfos[server[0].strip('(').split()[0]] = [int(server[1].split()[0])/2, int(server[1].split()[0])/2, \
                                                    int(server[2].split()[0])/2, int(server[2].split()[0])/2,
                                                        int(server[3].split()[0]), int(server[4].strip(')\n').split()[0])]
        serverInfosList.append([server[0].strip('(').split()[0], int(server[1].split()[0]), int(server[2].split()[0]),\
                                int(server[3].split()[0]), int(server[4].strip(')\n').split()[0])])
            
    
    serverInfosList = sorted(serverInfosList,key=lambda x:x[3])
            

            
    if is_test:
        
        M = file.readline()  # 代表虚拟机的配置
    else:
        M = input()
    sys.stdout.flush()
    # 虚拟机
    for i in range(int(M)):
        if is_test:
            vm = file.readline().split(',')
        else:
            vm = input().split(',')
        sys.stdout.flush()
        vmInfos[vm[0].strip('(').split()[0]] = [int(vm[1].split()[0]), int(vm[2].split()[0]), int(vm[3].strip(')\n').split()[0])]



    requestdays = 0
    dayRequestNumber = 0
    #requestdays = int(file.readline())  # 天数
    op = ''
    if is_test:
        requestdays = int(file.readline())
    else:
        requestdays = int(input())
    sys.stdout.flush()

    #bugServer()
    """
    for lis in serverInfosList:
        lis.append(serverXingjiabi(lis[1],lis[2],lis[3],lis[4],0,int(requestdays)))
    serverInfosList = sorted(serverInfosList, key=lambda x:x[5])
    """

    # 操作天数
    for day in range(requestdays):
        if is_test:
            dayRequestNumber = file.readline()  # 操作数
        else:
            dayRequestNumber = input()
        sys.stdout.flush()

        requestInfos.clear()  # 每天开始时清空上一次循环购买的服务器
        # 每天操作数
        for r in range(int(dayRequestNumber)):
            if is_test:
                op = file.readline().split(',')
            else:
                op = input().split(',')
            sys.stdout.flush()

            if len(op) == 2:  # 删除del操作
                op_del_list = [op[0].strip('(').split()[0], op[1].strip(')\n').split()[0]]
                requestInfos.append(op_del_list)
            else:  # 添加add操作
                requestInfos.append([op[0].strip('(').split()[0], op[1].split()[0], op[2].strip(')\n').split()[0]])

        match(day)
        if is_test:
            serverPower()

            TOTALCOST = SERVERCOST + POWERCOST

    output()
    if is_test:
        print(TOTALCOST)
    pass

if __name__ == "__main__":
    main()
