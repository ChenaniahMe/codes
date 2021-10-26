class HMM(object):
    def __init__(self):
        # 状态值集合viterbi
        self.state_lists = ['B', 'M', 'E', 'S']
        # 状态转移概率
        self.A = {}
        # 观测概率
        self.B = {}
        # 初始概率
        self.pi = {}
        # 统计B, M, E, S状态出现的次数，并且求出相应的概率
        self.count_dict = {}

    def train(self, path):
        def init_parameters():
            for state in self.state_lists:
                #初始化状态转移概率
                self.A[state] = {s: 0.0 for s in self.state_lists}
                self.B[state] = {}
                #初始化初始概率
                self.pi[state] = 0.0
                self.count_dict[state] = 0

        def makeLabels(text):
            output = []
            if len(text) == 1:
                output.append('S')
            else:
                output += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return output

        init_parameters()
        line_nums = 0

        with open(path, encoding='utf-8') as f:
            for line in f:
                line_nums += 1

                line = line.strip()
                if not line:
                    continue

                word_lists = [i for i in line if i != ' ']
                line_list = line.split()

                line_states = []
                for w in line_list:
                    line_states.extend(makeLabels(w))

                assert len(word_lists) == len(line_states)

                for k, v in enumerate(line_states):
                    self.count_dict[v] += 1
                    if k == 0:
                        self.pi[v] += 1
                    else:
                        #从上一个状态到v状态
                        self.A[line_states[k - 1]][v] += 1
                        #从状态到单词
                        self.B[line_states[k]][word_lists[k]] = \
                            self.B[line_states[k]].get(word_lists[k], 0) + 1.0
        #self.pi中的结果和为1
        self.pi = {k: v * 1.0 / line_nums for k, v in self.pi.items()}
        self.A = {k: {k1: v1 / self.count_dict[k] for k1, v1 in v.items()} for k, v in self.A.items()}
        self.B = {k: {k1: (v1 + 1) / self.count_dict[k] for k1, v1 in v.items()} for k, v in
                      self.B.items()}
        return self

    def viterbi(self, text, states, start_p, trans_p, ober_p):
        """这是一个递推式算法，不需要迭代更新
        text:要切分的句子
        states: B,M,E,S
        start_p:初始概率
        trans_p:转移概率矩阵
        ober_p:观测概率矩阵
        """
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * ober_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            never_find = text[t] not in ober_p['S'].keys() and \
                        text[t] not in ober_p['M'].keys() and \
                        text[t] not in ober_p['E'].keys() and \
                        text[t] not in ober_p['B'].keys()
            for y in states:
                emitP = ober_p[y].get(text[t], 0) if not never_find else 1.0
                (pro, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0) for y0 in states if V[t - 1][y0] >= 0])
                V[t][y] = pro
                newpath[y] = path[state] + [y]
            path = newpath
        (pro, state) = max([(V[len(text) - 1][y], y) for y in ['E', 'S']])
        return pro, path[state]

    def cut(self, text):
        pro, pos_list = self.viterbi(text, self.state_lists, self.pi, self.A, self.B)
        begin = 0
        result = []
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                result.append(text[begin: i + 1])
            elif pos == 'S':
                result.append(char)

        return result

hmm = HMM()
hmm.train('./msr_training.utf8')
def text2index(path, cut=True):
   with open(path,encoding='utf-8') as f:
       dict = {}
       i = 0
       for line in f:
           line = line.strip()
           if cut:
               res = line.split()
           else:
               res = hmm.cut(line)
           dict[i] = []
           nums = 0
           for s in res:
               dict[i].append((nums, nums + len(s) - 1))
               nums += len(s)
           i += 1
#            print(dict)
   return dict


def evaluate(evaluate, gold):
   dict_evaluate = text2index(evaluate, cut=False)
   dict_gold = text2index(gold)

   linelen = len(dict_evaluate)
   assert len(dict_evaluate) == len(dict_gold)

   nums_evaluate = 0
   nums_gold = 0
   nums_correct = 0
   for i in range(linelen):
       seq_evaluate = dict_evaluate[i]
       seq_gold = dict_gold[i]
       nums_evaluate += len(seq_evaluate)
       nums_gold += len(seq_gold)
       for t in seq_evaluate:
           if t in seq_gold:
               nums_correct += 1

   P = nums_correct / nums_evaluate
   R = nums_correct / nums_gold
   F1 = 2*P * R / (P + R)
   return P, R, F1
hmm = HMM()
hmm.train('./msr_training.utf8')
text = '这是一个非常好的方案！'
res = hmm.cut(text)
print(text)
print(res)

P, R, F1 = evaluate('./msr_test.utf8', './msr_test_gold.utf8')
print("HMM的精确率：", round(P, 3))
print("HMM的召回率：", round(R, 3))
print("HMM的F1值：", round(F1, 3))