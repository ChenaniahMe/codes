import sys
import copy
class Server(object):
    def __init__(self):
        #读取全部的数据集
        self.dataset = []

        # 服务器信息和数量
        self.serverInfos = {}
        self.serverNum = 0

        # 请求
        self.requestAll = {}

        ##虚拟机信息
        self.vmInfos = {}

        ##天数
        self.dayNum = 0

        #每一天的信息
        self.days = { i:[] for i in range(0,1000)}

        #每一天的请求次数
        self.reqEDayNum = []

        #买到的服务器
        self.ownServer = {}
        #买到的服务器数量
        self.ownServerNum = 0
        #服务器功耗
        self.serverCost = 0

        #虚拟机在哪个服务器上运行
        self.vmOnServer = {}
        #最后的结果
        self.res = []
        #删除多余的服务器
        self.delRem = 0

    def loadServer(self):
        self.serverNum = int(self.dataset[0])
        for i in range(1, self.serverNum+1):
            tdata = self.dataset[i].split()
            type = tdata[0][1:-1]  # 服务器名 host0Y6DP
            cpu = int(tdata[1][:-1])  # CPU数 300 需要类型转换
            mem = int(tdata[2][:-1])  # 内存数 830 需要类型转换
            price = int(tdata[3][:-1])
            power = int(tdata[4][:-1])
            #服务器双结点部署,最后一项代表是否被使用过
            self.serverInfos[type] = [cpu/2,cpu/2, mem/2,mem/2, price, power,0]

    def loadVm(self):
        self.vmNum = int(self.dataset[self.serverNum+1])
        for i in range(self.serverNum+2, self.serverNum+2+self.vmNum):
            tdata = self.dataset[i].split()
            type = tdata[0][1:-1]
            cpu = int(tdata[1][:-1])
            mem = int(tdata[2][:-1])
            #是否为单双结点部署
            isST = int(tdata[3][:-1])
            self.vmInfos[type] = [cpu, mem, isST]

    def loadDay(self):
        self.dayNum = int(self.dataset[self.serverNum + self.vmNum + 2])
        #开始的索引值
        treqN = 0
        tindex = self.serverNum + self.vmNum + 3 + treqN
        for i in range(0, self.dayNum):
            # self.reqEDayNum.append(int(self.dataset[self.serverNum + self.vmNum + 3]))
            treqN = int(self.dataset[tindex])
            l= tindex + 1
            r= l + treqN
            for j in range(l, r):
                tdata = self.dataset[j].split()
                flag = tdata[0][1:-1]  # 请求类别标志：增/删
                if flag == "add":
                    type = tdata[1][:-1]  # 虚拟机类型
                    id = int(tdata[2][:-1])  # 虚拟机ID[type,id]
                    self.days[i].append([type,id])
                else:
                    id = int(tdata[1][:-1])  # 虚拟机ID
                    self.days[i].append([id])
            tindex = r

    def loadData(self,method):
        # for line in sys.stdin:
        if method=="test":
            f = open(r"training-2.txt", "r", encoding="utf-8")
            for line in f:
                self.dataset.append(line)
        else:
            for line in sys.stdin:
                self.dataset.append(line)
        self.loadServer()
        self.loadVm()
        self.loadDay()

    def chooseServer(self, vmInfo,i,req):
        #虚拟机cpu，虚拟机vmMem,虚拟机是否为单双结点
        vmCpu,vmMem,isST = vmInfo[0],vmInfo[1],vmInfo[2]
        own_server =  self.ownServer[i]
        serverCpuA, serverCpuB, serverMemA, serverMemB = own_server[0],own_server[1],own_server[2],own_server[3]
        if req[1]==699543921 and i==2499:
            print("Debug")
        if isST ==1:
            vmCpuA, vmCpuB = vmCpu / 2, vmCpu / 2
            vmMemA, vmMemB = vmMem / 2, vmMem / 2
            if vmCpuA<=serverCpuA and vmCpuB<=serverCpuB and vmMemA<=serverMemA and vmMemB<=serverMemB:
                own_server[0] -= vmCpuA
                own_server[1] -= vmCpuB
                own_server[2] -= vmMemA
                own_server[3] -= vmMemB
                self.res.append('('+str(i)+')')
                self.vmOnServer[req[1]] = [i, req[0],""]
                self.ownServer[i][-1] = 1
                return True
            else:
                return False
        else:
            if vmCpu<=serverCpuA and vmMem<=serverMemA:
                own_server[0] -= vmCpu
                own_server[2] -= vmMem
                self.res.append('(' + str(i) + ',A)')
                # 指定这个需求运行在自己的哪个服务器上, 虚拟机的名字, 以及部署的结点
                self.vmOnServer[req[1]] = [i, req[0],"A"]
                self.ownServer[i][-1] = 1
                return True
            elif vmCpu<=serverCpuB and vmMem<=serverMemB:
                own_server[1] -= vmCpu
                own_server[3] -= vmMem
                self.res.append('(' + str(i) + ',B)')
                # 指定这个需求运行在自己的哪个服务器上, 虚拟机的名字, 以及部署的结点
                self.vmOnServer[req[1]] = [i, req[0], "B"]
                self.ownServer[i][-1] = 1
                return True
        return False

    def delVm(self, vmId):
        #虚拟机cpu，虚拟机vmMem,虚拟机是否为单双结点
        serverId, vmName,node = self.vmOnServer[vmId]
        vmCpu, vmMem, isST = self.vmInfos[vmName][0],self.vmInfos[vmName][1],self.vmInfos[vmName][2]
        ownServer = self.ownServer[serverId]
        if isST:
            vmCpuA, vmCpuB = vmCpu / 2, vmCpu / 2
            vmMemA, vmMemB = vmMem / 2, vmMem / 2
            ownServer[0] += vmCpuA
            ownServer[1] += vmCpuB
            ownServer[2] += vmMemA
            ownServer[3] += vmMemB
        else:
            if node=="A":
                ownServer[0] += vmCpu
                ownServer[2] += vmMem
            else:
                ownServer[1] += vmCpu
                ownServer[3] += vmMem


    def dealRequest(self):
        self.res.append('(purchase, 2500)')
        self.res.append('(hostUY41I, 1250)')
        self.res.append('(host78BMY, 1250)')
        self.res.append('(migration, 0)')
        #先暂时考虑买两种服务器
        for i in range(1250):
            self.ownServer[self.ownServerNum] = copy.deepcopy(self.serverInfos["hostUY41I"])
            self.ownServerNum += 1

        for i in range(1250):
            self.ownServer[self.ownServerNum] = copy.deepcopy(self.serverInfos["host78BMY"])
            self.ownServerNum += 1

        for key in self.days:
            # sum_cpu,sum_mem = 0, 0
            for req in self.days[key]:
                if len(req)==2:
                    #需求虚拟机的类型名
                    vmInfo = self.vmInfos[req[0]]
                    for i in range(self.ownServerNum):
                        if self.chooseServer(vmInfo,i,req):
                            break
                else:
                    vmId = req[0]
                    self.delVm(vmId)
            self.res.append('(purchase, 0)')
            self.res.append('(migration, 0)')

    #计算代价
    def computeCost(self):
        for i in range(1250, 2500):
            if self.ownServer[i][-1]==0:
                self.delRem += 1

server = Server()
server.loadData("test")
server.dealRequest()
for i in range(len(server.res)):
    if i==2:
        print('(host78BMY,'+str(1250-server.delRem)+')')
    else:
        print(server.res[i])
