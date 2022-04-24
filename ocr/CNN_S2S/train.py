import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import Im2LatexModel, Trainer
from utils import collate_fn, get_checkpoint
from data import Im2LatexDataset
from build_vocab import Vocab, build_vocab, load_vocab


def main():

    # get args
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument("--emb_dim", type=int,
                        default=80, help="Embedding size")
    parser.add_argument("--dec_rnn_h", type=int, default=512,
                        help="The hidden state of the decoder RNN")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("--add_position_features", action='store_true',
                        default=False, help="Use position embeddings or not")
    parser.add_argument("--max_len", type=int,
                        default=150, help="Max size of formula")
    parser.add_argument("--dropout", type=float,
                        default=0.2, help="Dropout probility")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Learning Rate")
    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. ")

    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Learning Rate Decay Patience")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")
    parser.add_argument("--save_dir", type=str,
                        default="./ckpts", help="The dir to save checkpoints")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="The frequency to print message")
    parser.add_argument("--seed", type=int, default=2020,
                        help="The random seed for reproducing ")
    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")

    args = parser.parse_args()
    
    max_epoch = args.epochs
    loadCheckpoint = args.from_check_point
    if loadCheckpoint:
        checkpoint_path = get_checkpoint(args.save_dir)
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']
    print("Training args:", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Building vocab
    print("Load vocab...")
    vocab = load_vocab(args.data_path)

    use_cuda = True if args.cuda and torch.cuda.is_available() else False
    device = torch.device("cuda" if use_cuda else "cpu")

    # data loader
    print("Construct data loader...")
    train_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'train', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=args.num_workers)
    val_loader = DataLoader(
        Im2LatexDataset(args.data_path, 'validate', args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=args.num_workers)

    model = Im2LatexModel(
        len(vocab), args.emb_dim, args.dec_rnn_h,
        add_pos_feat=args.add_position_features,
        dropout=args.dropout
    )
    model = model.to(device)
    # construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=args.lr_decay,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr)

    if loadCheckpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_sche'])
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=epoch, last_epoch=max_epoch)
    else:
        trainer = Trainer(optimizer, model, lr_scheduler,
                          train_loader, val_loader, args,
                          use_cuda=use_cuda,
                          init_epoch=1, last_epoch=args.epochs)
    trainer.train()


if __name__ == "__main__":
    build_vocab("./processed/")
    main()

