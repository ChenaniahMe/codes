# encoding: utf-8

import numpy as np
import sys

test = []


def Physerver_Load(line, physerver, physerver_type):  # 服务器统计
    line = line.split()  # (host0Y6DP, 300, 830, 141730, 176)分割为['(host0Y6DP,', '300,', '830,', '141730,', '176)']
    temp_physerver_type = line[0][1:-1]  # 服务器名 host0Y6DP
    cpu_limit = int(line[1][:-1])  # CPU数 300 需要类型转换
    ram_limit = int(line[2][:-1])  # 内存数 830 需要类型转换
    physerver[temp_physerver_type] = [cpu_limit, ram_limit]  # 构造服务器信息字典
    # 每天要购买的物理服务器类型 字典
    physerver_type[temp_physerver_type] = 0  # 构造每天要购买的服务器类型字典


def Virtualmachine_Load(line, vm):  # 虚拟机统计 规则同上
    line = line.split()
    vm_type = line[0][1:-1]
    cpu_usage = int(line[1][:-1])
    ram_usage = int(line[2][:-1])
    numa_id = int(line[3][:-1])
    vm[vm_type] = [cpu_usage, ram_usage, numa_id]  # CPU核数 内存大小 单双节点


def Purchase_Physerver(physerver, vm, vmid, id_num, vm_type, own_physerver, own_num, cur_cpu_sum, cur_ram_sum,
                       physerver_deploy, physerver_type, machine_type):
    # 单节点：服务器的资源采取等分策略，若虚拟机为单节点，则将虚拟机分配给服务器的A节点
    if (vm[vm_type][2] == 0):
        numa_node_cpu_limit = physerver[machine_type][0] / 2
        numa_node_ram_limit = physerver[machine_type][1] / 2
        # own_num为服务器编号，每拥有一台+1
        #                         服务器型号             A-CPU剩余                       B-CPU剩余                A-RAM剩余                      B-RAM剩余
        own_physerver[own_num] = [machine_type, numa_node_cpu_limit - cur_cpu_sum, numa_node_cpu_limit,
                                  numa_node_ram_limit - cur_ram_sum, numa_node_ram_limit]
        vmid[id_num][1] = own_num
        vmid[id_num][2] = 'A'
        physerver_deploy.append('(%d, A)' % own_num)
    # 双节点：服务器的资源采取等分策略，若虚拟机为双节点，则将虚拟机分配给服务器的A节点与B节点
    if (vm[vm_type][2] == 1):
        numa_node_cpu_limit = physerver[machine_type][0] / 2
        numa_node_ram_limit = physerver[machine_type][1] / 2
        own_physerver[own_num] = [machine_type, numa_node_cpu_limit - cur_cpu_sum / 2,
                                  numa_node_cpu_limit - cur_cpu_sum / 2, numa_node_ram_limit - cur_ram_sum / 2,
                                  numa_node_ram_limit - cur_ram_sum / 2, ]
        vmid[id_num][1] = own_num
        vmid[id_num][2] = 0
        physerver_deploy.append('(%d)' % own_num)


def Deploy_Req_Single(vmid, id_num, own_physerver, physerver_deploy, cur_cpu_sum, cur_ram_sum):  # 单节点配置：A部分满足要求，则挂载至A；B部分满足要求，则挂载至B
    flag = 0
    for key in own_physerver:
        if (own_physerver[key][1] > cur_cpu_sum and own_physerver[key][3] > cur_ram_sum):
            own_physerver[key][1] = own_physerver[key][1] - cur_cpu_sum  # 所拥有的cpu资源余量更新
            own_physerver[key][3] = own_physerver[key][3] - cur_ram_sum  # 所拥有的内存资源余量更新
            vmid[id_num][1] = key  # 挂载的服务器
            vmid[id_num][2] = 'A'  # 挂载的服务器的节点
            physerver_deploy.append('(%d, A)' % key)  # 挂载信息记录
            flag = 1
            break
        if (own_physerver[key][2] > cur_cpu_sum and own_physerver[key][4] > cur_ram_sum):
            own_physerver[key][2] = own_physerver[key][2] - cur_cpu_sum
            own_physerver[key][4] = own_physerver[key][4] - cur_ram_sum
            vmid[id_num][1] = key
            vmid[id_num][2] = 'B'
            physerver_deploy.append('(%d, B)' % key)
            flag = 1
            break
    return flag


def Deploy_Req_Double(vmid, id_num, own_physerver, physerver_deploy, cur_cpu_sum, cur_ram_sum):
    flag = 0
    for key in own_physerver:
        if (
                own_physerver[key][1] > cur_cpu_sum / 2 and own_physerver[key][2] > cur_cpu_sum / 2
                and
                own_physerver[key][3] > cur_ram_sum / 2 and own_physerver[key][4] > cur_ram_sum / 2
        ):  # AB节点均需满足需求
            own_physerver[key][1] = own_physerver[key][1] - cur_cpu_sum / 2
            own_physerver[key][2] = own_physerver[key][2] - cur_cpu_sum / 2
            own_physerver[key][3] = own_physerver[key][3] - cur_ram_sum / 2
            own_physerver[key][4] = own_physerver[key][4] - cur_ram_sum / 2
            vmid[id_num][1] = key
            vmid[id_num][2] = 0
            physerver_deploy.append('(%d)' % key)
            flag = 1
            break
    return flag


def main():
    # 临时参数
    machine_flag = 0
    machine_type = 'str'

    # 数据预处理参数
    load_flag = 0  # 载入状态
    physerver = {}
    physerver_num = 0
    virtualmachine = {}
    virtualmachine_num = 0

    # 处理每日请求
    day_num = 0
    req_cnt = 0
    add_del_flag = "str"
    # 当前这条请求的需求
    cur_cpu_sum = 0
    cur_ram_sum = 0

    own_physerver = {}
    own_num = 0
    virtualmachine = {}
    vmid = {}

    # 输出参数
    Q = 0
    physerver_type = {}
    W = 0
    physerver_deploy = []
    res = []
    f = open(r"training-2.txt", "r", encoding="utf-8")
    for line in f:
    # for line in sys.stdin:  # 逐行读取
        # sys.stdin是一个标准化输入的方法;
        # python3中使用sys.stdin.readline()可以实现标准输入，其中默认输入的格式是字符串，
        # 如果是int，float类型则需要强制转换。

        # line = input()
        # line = sys.stdin.readline()
        # 物理服务器列表载入
        if load_flag == 0:  # 1 读取服务器数目
            physerver_num = int(line[:])
            load_flag = 1  # 修改文件载入flag
        elif load_flag == 1:  # 2 可购买的服务器详情
            physerver_num -= 1
            Physerver_Load(line, physerver, physerver_type)
            if physerver_num == 0:
                load_flag = 2  # 可购买的服务器信息读取完毕
            # 临时变量
            if machine_flag == 0:  # 取第一个服务器作为容量不足时购买的服务器，可优化！！！
                line = line.split()
                machine_type = line[0][1:-1]
                machine_flag = 1
        # 虚拟机列表载入
        elif load_flag == 2:  # 3 读取虚拟机类型数目
            virtualmachine_num = int(line[:])
            load_flag = 3
        elif load_flag == 3:  # 4 虚拟机类型信息详情
            virtualmachine_num -= 1
            Virtualmachine_Load(line, virtualmachine)
            if virtualmachine_num == 0:
                load_flag = 4  # 可购买的虚拟机信息读取完毕
        # 统计一共有几天
        elif load_flag == 4:  # 5 读取天数
            day_num = int(line[:])
            load_flag = 5
        # 统计当日有多少条请求
        elif load_flag == 5:  # 6 读取单日请求数
            day_num -= 1
            req_cnt = int(line[:]) #需求数量
            load_flag = 6
        # 处理每日请求
        elif load_flag == 6:  # 7 单日请求处理
            req_cnt -= 1  # 逐条处理请求
            line = line.split()
            add_del_flag = line[0][1:-1]  # 请求类别标志：增/删
            if (add_del_flag == 'add'):  # 增：add，虚拟机类型，虚拟机ID
                # 虚拟机id--虚拟机型号+服务器id+节点（先初始化为0）
                id_num = int(line[2][:-1])  # 虚拟机ID
                vm_type = line[1][:-1]  # 虚拟机类型
                vmid[id_num] = [vm_type, 0, 0]  # 虚拟机id--虚拟机型号+服务器id+节点（先初始化为0）
                cur_cpu_sum = virtualmachine[vm_type][0]  # 该虚拟机的cpu需求
                cur_ram_sum = virtualmachine[vm_type][1]  # 该虚拟机的内存需求
                # 遇到同一天 先add后del 同一个ID的情况
                # 不应该在一天全部序列计算完毕后判断所需最大资源
                # 应该在这一天中判断所需最大资源：单日所需最大资源出现的时间节点不一定是在单日结束！！

                # 剩余的cpu和ram资源 能满足 当前的cpu和ram需求
                # 可能出现剩余能满足但是分配时无法分配的情况
                # 先判断购买什么类型的物理服务器
                # 此处可优化，先跳过x
                # 默认购买NV603

                # 单节点
                if (virtualmachine[vm_type][2] == 0):
                    flag = Deploy_Req_Single(vmid, id_num, own_physerver, physerver_deploy, cur_cpu_sum, cur_ram_sum)
                # 双节点
                if (virtualmachine[vm_type][2] == 1):
                    flag = Deploy_Req_Double(vmid, id_num, own_physerver, physerver_deploy, cur_cpu_sum, cur_ram_sum)
                if (flag == 0):  # 购买服务器
                    Purchase_Physerver(
                        physerver, virtualmachine, vmid, id_num, vm_type, own_physerver, own_num,
                        cur_cpu_sum, cur_ram_sum, physerver_deploy, physerver_type, machine_type
                    )
                    own_num += 1
                    # 初始数量应该设为0， 建议对所有类型列字典，每天清0
                    physerver_type[machine_type] += 1
                    # Q += 1
            else:  # 删：del，虚拟机ID
                id_num = int(line[1][:-1])  # 虚拟机ID
                vm_type = vmid[id_num][0]  # 虚拟机类型
                own_index = vmid[id_num][1]  # 所在服务器
                cur_cpu_sum = virtualmachine[vm_type][0]  # 该虚拟机的cpu需求
                cur_ram_sum = virtualmachine[vm_type][1]  # 该虚拟机的内存需求
                # 双节点
                if (vmid[id_num][2] == 0):  # 双节点资源释放
                    own_physerver[own_index][1] += cur_cpu_sum / 2
                    own_physerver[own_index][2] += cur_cpu_sum / 2
                    own_physerver[own_index][3] += cur_ram_sum / 2
                    own_physerver[own_index][4] += cur_ram_sum / 2
                # A节点
                if (vmid[id_num][2] == 'A'):  # A节点资源释放
                    own_physerver[own_index][1] += cur_cpu_sum
                    own_physerver[own_index][3] += cur_ram_sum
                # B节点
                if (vmid[id_num][2] == 'B'):  # B节点资源释放
                    own_physerver[own_index][2] += cur_cpu_sum
                    own_physerver[own_index][4] += cur_ram_sum
                del vmid[id_num]
            # 必须放在最后等部署完再打印
            # 这一天的请求结束
            if req_cnt == 0:  # 单日请求处理完毕
                # 保存每日信息
                if (physerver_type[machine_type] > 0):  # 当日资源不足，则需购买新服务器
                    res.append('(purchase, 1)')
                    res.append('(%s, %d)' % (machine_type, physerver_type[machine_type]))
                    physerver_type[machine_type] = 0  # 购买标志位置0
                else:
                    res.append('(purchase, 0)')
                # 每日清0
                # Q = 0
                # for key,values in physerver_type.items():
                #    if(values > 0):
                #        print('(%s, %d)' % (key,values))
                #        #每日购买的服务器类型清 0
                #        physerver_type[key] = 0
                res.append('(migration, 0)')  # 默认不搬移，可优化
                for i in physerver_deploy:
                    res.append(i)
                physerver_deploy.clear()
                if day_num == 0:
                    load_flag = 7
                    for i in res:
                        print(i)
                    break
                else:
                    load_flag = 5  # 单日请求处理完成 继续处理后一日


if __name__ == "__main__":
    main()
    print('end')