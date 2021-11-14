import sys
sys.path.append('C:\\Users\\USER\\PycharmProjects\\DTML')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Stock_Dataset import StockDataset
import argparse
from Encoder_Decoder_LSTM import LSTM_enc
from Encoder_Decoder_LSTM import LSTM_dec
from Transformer_Encoder import Transformer
import numpy as np


def train(encoder, decoder,transformer,                                  ## Model

          encoder_optimizer, decoder_optimizer, transformer_optimizer,   ## Optimizer
          Partition, args):                                      ## Data, loss function, argument
    trainloader = DataLoader(Partition['train'],
                             batch_size = args.batch_size,
                             shuffle=False, drop_last=True)

    train_loss = 0.0
    for i, (x,y) in enumerate(trainloader):
        encoder.train()
        decoder.train()
        transformer.train()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        transformer_optimizer.zero_grad()
        data_out_list = []
        for i in range(len(x)):
            encoder.train()
            decoder.train()
            transformer.train()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            transformer_optimizer.zero_grad()

            input_x = x[i].transpose(0,1).float().to(args.device)
            true_y = y[i].squeeze().float().to(args.device)



            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

            y_pred_encoder, hidden_encoder = encoder(input_x)
            hidden_decoder = hidden_encoder

            decoder_input = torch.zeros(args.decoder_x_frames, args.batch_size, args.input_dim).to(args.device)

            y_pred_decoder, attention_weight, attn_applied = decoder(decoder_input, hidden_decoder)

            data_out_list.append(attn_applied)

        index_output = data_out_list[0] # torch.Size([128, 10])
        stock_output = data_out_list[1] # torch.Size([128, 10])

        Transformer_input = index_output + stock_output

        Transformer_input = Transformer_input.transpose(0,2)



        output = transformer(Transformer_input)

        output = torch.where(output >= 0.5, 1, 0)

        print(true_y.size())

        print(output.size())

        loss = args.loss_fn(output, true_y)
        loss.backward()

        encoder_optimizer.step()  ## parameter 갱신
        decoder_optimizer.step()
        transformer_optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    return encoder, decoder, transformer, train_loss


# def validation(encoder, decoder,transformer,
#                partition, args):
#
# def test(encoder, decoder, transformer,
#                partition, args):


def experiment(partition, args):
    encoder = args.encoder(args.input_dim, args.hid_dim, args.num_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.output_dim, args.num_layers, args.batch_size,
                           args.dropout, args.use_bn, args.attention_head, args.attn_size,
                           activation="ReLU")
    transformer = args.transformer(args.trans_feature_size, args.trans_num_laysers,
                                   args.dropout, args.batch_size, args.x_frames, args.trans_nhead)

    encoder.to(args.device)
    decoder.to(args.device)
    transformer.to(args.device)

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
        encoder, decoder, transformer, train_loss = train(encoder, decoder, transformer,
                enc_optimizer, dec_optimizer, transformer_optimizer,
                partition, args)


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





# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

# ========= experiment setting ========== #
parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'





## ======== data ============= #
args.index = ['^IXIC']
args.stock_list = ['AAPL', 'AMZN','MSFT','TSLA','GOOG','FB','NVDA','AMD']
args.data_count = len(args.stock_list)
args.train_start = "2012-01-01"
args.train_end = "2016-12-31"
args.validation_start = "2017-01-01"
args.validation_end = "2018-12-31"
args.test_start = "2019-01-01"
args.test_end = "2020-12-31"

# ====== hyperparameter ======= #
args.batch_size = 128
args.x_frames = 10
args.y_frames = 1
args.input_dim = 11
args.output_dim = 1
args.dropout = 0.0
args.use_bn = True
args.loss_fn = nn.BCELoss()  ## loss function for classification : cross entropy
args.optim = 'Adam'
args.lr = 0.01
args.l2 = 0.00001 #?
args.epoch = 250

# ============= model ================== #
args.encoder = LSTM_enc
args.decoder = LSTM_dec
args.transformer = Transformer

# ====== att_lstm hyperparameter ======= #
args.hid_dim = 10
args.attention_head = 1
args.attn_size = 10
args.num_layers = 1
args.decoder_x_frames = 1

# ====== transformer hyperparameter ======= #
args.trans_feature_size = 250
args.trans_num_laysers = 1
args.trans_nhead = 10


for stock in args.stock_list:
    args.entire_datalist = args.index + [stock]
    # 0번째에 index 1번째에 stock 1개가 input으로 들어감
    trainset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                            args.train_start, args.train_end)
    valset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                          args.validation_start,args.validation_end)
    testset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                           args.test_start, args.test_end)
    partition = {'train': trainset, 'val': valset, 'test': testset}

    experiment(partition, args)

