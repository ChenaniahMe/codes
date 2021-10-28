import pickle
import random
import time

import numpy as np
import torch

from config import device, logger, data_file, vocab_file
from transformerme.transformer import Transformer

def main():
    filename = 'transformer.pt'
    print('loading {}...'.format(filename))
    start = time.time()
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
    model = model.to(device)
    model.eval()

    start = time.time()
    with open(vocab_file, 'rb') as file:
        data = pickle.load(file)
        src_char2idx = data['dict']['src_char2idx']
        tgt_idx2char = data['dict']['tgt_idx2char']
    elapsed = time.time() - start
    logger.info('load {} elapsed: {:.4f} seconds'.format(vocab_file, elapsed))
    

    samples = ['and just act like a normal human being ',
               'once again i ask myself',
               "you know why i ve been so hard on you ",
               'and what is my tell',
               'look   she only suspects something okay ']
    res = []

    for sample in samples:
        id_word = [src_char2idx[word] for word in sample.split()]
        input = torch.from_numpy(np.array(id_word, dtype=np.long)).to(device)
        input_length = torch.LongTensor([len(id_word)]).to(device)

        with torch.no_grad():
            nbest_hyps = model.recognize(input=input, input_length=input_length, char_list=tgt_idx2char)

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [tgt_idx2char[idx] for idx in out]
            out = ''.join(out)
            out = out.replace('<sos>', '').replace('<eos>', '')
            res.append(out)
    for i in range(len(res)):
        print(samples[i], '\t', res[i])


if __name__ == '__main__':
        main()


