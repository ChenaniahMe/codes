import math
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import device, print_freq, sos_id, eos_id, n_src_vocab, n_tgt_vocab, grad_clip, logger
from data_gen import AiChallenger2017Dataset, pad_collate
from transformerme.decoder import Decoder
from transformerme.encoder import Encoder
from transformerme.loss import cal_performance
from transformerme.optimizer import TransformerOptimizer
from transformerme.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient


# from torch import nn


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(n_src_vocab, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, n_tgt_vocab,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiChallenger2017Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=args.num_workers)
    valid_dataset = AiChallenger2017Dataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=False, num_workers=args.num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           writer=writer)

        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/learning_rate', optimizer.lr, epoch)

        print('\nLearning rate: {}'.format(optimizer.lr))
        print('Step num: {}\n'.format(optimizer.step_num))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        writer.add_scalar('epoch/valid_loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger, writer):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    times = AverageMeter()

    start = time.time()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        pred, gold = model(padded_input, input_lengths, padded_target)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
        try:
            assert (not math.isnan(loss.item()))
        except AssertionError:
            print('n_correct: ' + str(n_correct))
            print('data: ' + str(n_correct))
            continue

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer.optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        elapsed = time.time() - start
        start = time.time()

        losses.update(loss.item())
        times.update(elapsed)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch time {time.val:.5f} ({time.avg:.5f})\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), time=times,
                                                                      loss=losses))
            writer.add_scalar('step_num/train_loss', losses.avg, optimizer.step_num)
            writer.add_scalar('step_num/learning_rate', optimizer.lr, optimizer.step_num)

    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in valid_loader:
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        with torch.no_grad():
            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
            try:
                assert (not math.isnan(loss.item()))
            except AssertionError:
                print('n_correct: ' + str(n_correct))
                print('data: ' + str(n_correct))
                continue

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
