import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
with open('cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.split('\n')
data = data[:]

en_data = [line.split('\t')[0] for line in data]
ch_data = ['\t' + line.split('\t')[1] + '\n' for line in data]
print('英文数据:\n', en_data[:10], len(en_data))
print('\n中文数据:\n', ch_data[:10], len(ch_data))
# 分别生成中英文字典
en_vocab = set(''.join(en_data))
id2en = list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

# 分别生成中英文字典
ch_vocab = set(''.join(ch_data))
id2ch = list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)

# 利用字典，映射数据
en_num_data = [[en2id[en] for en in line ] for line in en_data]
ch_num_data = [[ch2id[ch] for ch in line] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line][1:] for line in ch_data]

print('char:', en_data[1])
print('index:', en_num_data[1])
print(ch_num_data[:10])
print(de_num_data[:10])

import numpy as np

# 获取输入输出端的最大长度
max_encoder_seq_length = max([len(txt) for txt in en_num_data])
max_decoder_seq_length = max([len(txt) for txt in ch_num_data])
print('max encoder length:', max_encoder_seq_length)
print('max decoder length:', max_decoder_seq_length)

# 将数据进行onehot处理
encoder_input_data = np.zeros((len(en_num_data), max_encoder_seq_length, len(en2id)), dtype='float32')
decoder_input_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')
decoder_target_data = np.zeros((len(ch_num_data), max_decoder_seq_length, len(ch2id)), dtype='float32')

for i in range(len(ch_num_data)):
    for t, j in enumerate(en_num_data[i]):
        encoder_input_data[i, t, j] = 1.
    for t, j in enumerate(ch_num_data[i]):
        decoder_input_data[i, t, j] = 1.
    for t, j in enumerate(de_num_data[i]):
        decoder_target_data[i, t, j] = 1.

print('index data:\n', en_num_data[1])
print('one hot data:\n', encoder_input_data[1])

# 模型参数
EN_VOCAB_SIZE = len(en2id)
CH_VOCAB_SIZE = len(ch2id)
HIDDEN_SIZE = 256

LEARNING_RATE = 0.003
BATCH_SIZE = 100
EPOCHS = 1

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.optimizers import Adam
#输入是将英文翻译为中文
#None表示可以处理任意长度的序列
encoder_inputs = Input(shape=(None, EN_VOCAB_SIZE))
#emb_inp = Embedding(output_dim=HIDDEN_SIZE, input_dim=EN_VOCAB_SIZE)(encoder_inputs)
encoder_h1, encoder_state_h1, encoder_state_c1 = LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(encoder_inputs)
encoder_h2, encoder_state_h2, encoder_state_c2 = LSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

# ==============decoder=============
decoder_inputs = Input(shape=(None, CH_VOCAB_SIZE))
#emb_target = Embedding(output_dim=HIDDEN_SIZE, input_dim=CH_VOCAB_SIZE, mask_zero=True)(decoder_inputs)
lstm1 = LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
lstm2 = LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
#decoder最后输出的是中文，所以直接做一个Dense变为中文
decoder_dense = Dense(CH_VOCAB_SIZE, activation='softmax')
decoder_h1, _, _ = lstm1(decoder_inputs, initial_state=[encoder_state_h1, encoder_state_c1])
decoder_h2, _, _ = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
decoder_outputs = decoder_dense(decoder_h2)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.)

# encoder模型和训练相同
encoder_model = Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])

# 预测模型中的decoder的初始化状态需要传入新的状态
decoder_state_input_h1 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c1 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_h2 = Input(shape=(HIDDEN_SIZE,))
decoder_state_input_c2 = Input(shape=(HIDDEN_SIZE,))

# 使用传入的值来初始化当前模型的输入状态
decoder_h1, state_h1, state_c1 = lstm1(decoder_inputs, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
decoder_outputs = decoder_dense(decoder_h2)

decoder_model = Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2],
                      [decoder_outputs, state_h1, state_c1, state_h2, state_c2])

for k in range(40, 50):
    test_data = encoder_input_data[k:k + 1]
    h1, c1, h2, c2 = encoder_model.predict(test_data)
    target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
    target_seq[0, 0, ch2id['\t']] = 1
    outputs = []
    while True:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_index_test = np.argmax(output_tokens[0, 0, :]) #这个表达式和上边的应该是一样的，都是为了得到这个的最大值
        outputs.append(sampled_token_index)
        target_seq = np.zeros((1, 1, CH_VOCAB_SIZE))
        target_seq[0, 0, sampled_token_index] = 1
        if sampled_token_index == ch2id['\n'] or len(outputs) > 20: break

    print(en_data[k])
    print(''.join([id2ch[i] for i in outputs]))