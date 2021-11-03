import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Stock_Dataset import StockDataset
import argparse

def train(encoder, decoder,transformer,                                  ## Model
          encoder_optimizer, decoder_optimizer, transformer_optimizer,   ## Optimizer
          Partition, loss_fn,args):                                      ## Data, loss function, argument
    trainloader = DataLoader(Partition['train'],
                             batch_size = args.batch_size,
                             shuffle=False, drop_last=True)

    encoder.train()
    decoder.train()
    transformer.train()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    transformer_optimizer.zero_grad()

    train_loss = 0.0
    for i, (x,y) in enumerate(trainloader):
        encoder.train()
        decoder.train()
        transformer.train()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        transformer_optimizer.zero_grad()

        x = x.to(args.device)
        y_true = y.to(args.device)

        encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

        y_pred_encoder, hidden_encoder = encoder(x)

        hidden_decoder = hidden_encoder

        y_pred_decoder, attention_weight = decoder(x, hidden_decoder)



        loss = loss_fn(y_pred_decoder,y_true)

        loss.backward()

    return

def validation(encoder, decoder,transformer,
               partition, loss_fn, args):

def test(encoder, decoder, transformer,
               partition, loss_fn, args):

def experiment(partition, args):
    encoder = args.encoder()
    decoder = args.decoder()
    transformer = args.transformer()

    encoder.to(args.device)
    decoder.to(args.device)
    transformer.to(args.device)

    loss_fn = nn.BCELOSS() ## loss function for classification : cross entropy

    if args.optim == 'SGD':
        enc_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.SGD(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
        transformer_optimizer = optim.SGD(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        enc_optimizer = optim.RMSprop(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.RMSprop(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
        transformer_optimizer = optim.RMSprop(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        enc_optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.l2)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.l2)
        transformer_optimizer = optim.Adam(transformer.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    # ===================================== #

    for epoch in range(args.epoch):


# '^KS11' : KOSPI
# '^KQ11' : 코스닥
# '^IXIC' : 나스닥
# '^GSPC' : SNP 500 지수
# '^DJI' : 다우존수 산업지수
# '^HSI' : 홍콩 항생 지수
# '^N225' : 니케이지수
# '^GDAXI' : 독일 DAX
# '^FTSE' : 영국 FTSE
# '^FCHI' : 프랑스 CAC
# '^IBEX' : 스페인 IBEX
# '^TWII' : 대만 기권
# '^AEX' : 네덜란드 AEX
# '^BSESN' : 인도 센섹스
# 'RTSI.ME' : 러시아 RTXI
# '^BVSP' : 브라질 보베스파 지수
# 'GC=F' : 금 가격
# 'CL=F' : 원유 가격 (2000/ 8 / 20일 부터 데이터가 있음)
# 'BTC-USD' : 비트코인 암호화폐
# 'ETH-USD' : 이더리움 암호화폐




parser = argparse.ArgumentParser()
args = parser.parse_args("")

args.data_symbol_list = ['^IXIC', 'AAPL', 'AMZN','MSFT','TSLA','GOOG','FB','NVDA','AMD']
args.data_count = len(args.data_symbol_list)
args.train_start = "2012-01-01"
args.train_end = "2016-12-31"
args.validation_start = "2017-01-01"
args.validation_end = "2018-12-31"
args.test_start = "2019-01-01"
args.test_end = "2020-12-31"
args.x_frames = 10
args.y_frames = 1


trainset = StockDataset(args.data_symbol_list, args.x_frames, args.y_frames,args.train_start)
valset = StockDataset(args.data_symbol_list, args.x_frames, args.y_frames)
testset = StockDataset(args.data_symbol_list, args.x_frames, args.y_frames)
partition = {'train': trainset, 'val': valset, 'test': testset}




