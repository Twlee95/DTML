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
import time
from metric import metric_acc as ACC
import csv
import os

def train(encoder, decoder,transformer,                                  ## Model
          encoder_optimizer, decoder_optimizer, transformer_optimizer,   ## Optimizer
          Partition, args):                                      ## Data, loss function, argument
    trainloader = DataLoader(Partition['train'],
                             batch_size = args.batch_size,
                             shuffle=False, drop_last=True)

    encoder.train()
    decoder.train()
    transformer.train()

    train_loss = 0.0
    for i, (x,y) in enumerate(trainloader):
        data_out_list = []
        for i in range(len(x)):

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            transformer_optimizer.zero_grad()

            linear1 = nn.Linear(11,10)
            tanh1 = nn.Tanh()

            input_x = tanh1(linear1(x[i].float())).transpose(0,1).float().to(args.device)
            if i == 1:
                true_y = y[i].squeeze().float().to(args.device)

            encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

            y_pred_encoder, hidden_encoder = encoder(input_x)
            hidden_decoder = hidden_encoder

            decoder_input = torch.zeros(args.decoder_x_frames, args.batch_size, args.input_dim).to(args.device)

            y_pred_decoder, attention_weight, attn_applied = decoder(decoder_input, hidden_decoder)

            data_out_list.append(attn_applied)

        index_output = data_out_list[0] * args.market_beta # torch.Size([128, 10])
        stock_output = data_out_list[1] # torch.Size([128, 10])

        Norm_ = nn.LayerNorm(10,device=args.device)
        stock_output = Norm_(stock_output)

        Transformer_input = index_output + stock_output

        Transformer_input = Transformer_input.transpose(0,2)


        output1 = transformer(Transformer_input)


        # output_ = torch.where(output1 >= 0.5, 1.0, 0.0)
        # output_.requires_grad=True
        loss = args.loss_fn(output1, true_y)
        loss.backward()

        encoder_optimizer.step()  ## parameter 갱신
        decoder_optimizer.step()
        transformer_optimizer.step()

        train_loss += loss.item()

    train_loss = train_loss / len(trainloader)
    return encoder, decoder, transformer, train_loss


def validation(encoder, decoder,transformer,
               partition, args):
    valloader = DataLoader(partition['val'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    encoder.eval()
    decoder.eval()
    transformer.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(valloader):

            data_out_list = []
            for i in range(len(x)):
                encoder.train()
                decoder.train()
                transformer.train()

                linear1 = nn.Linear(11, 10)
                tanh1 = nn.Tanh()

                input_x = tanh1(linear1(x[i].float())).transpose(0, 1).float().to(args.device)
                if i==1:
                    true_y = y[i].squeeze().float().to(args.device)
                encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

                y_pred_encoder, hidden_encoder = encoder(input_x)
                hidden_decoder = hidden_encoder

                decoder_input = torch.zeros(args.decoder_x_frames, args.batch_size, args.input_dim).to(args.device)

                y_pred_decoder, attention_weight, attn_applied = decoder(decoder_input, hidden_decoder)

                data_out_list.append(attn_applied)

            index_output = data_out_list[0] * args.market_beta  # torch.Size([128, 10])
            stock_output = data_out_list[1]  # torch.Size([128, 10])

            Norm_ = nn.LayerNorm(10, device=args.device)
            stock_output = Norm_(stock_output)

            Transformer_input = index_output + stock_output

            Transformer_input = Transformer_input.transpose(0, 2)

            output1 = transformer(Transformer_input)

            loss = args.loss_fn(output1, true_y)

            val_loss += loss.item()

        val_loss = val_loss / len(valloader)
        return encoder, decoder, transformer, val_loss


def test(encoder, decoder, transformer,
               partition, args):
    testloader = DataLoader(partition['test'],
                           batch_size=args.batch_size,
                           shuffle=False, drop_last=True)
    encoder.eval()
    decoder.eval()
    transformer.eval()

    ACC_metric = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):

            data_out_list = []
            for i in range(len(x)):

                # feature transform
                linear1 = nn.Linear(11, 10)
                tanh1 = nn.Tanh()

                input_x = tanh1(linear1(x[i].float())).transpose(0, 1).float().to(args.device)
                if i==1:
                    true_y = y[i].squeeze().float().to(args.device)

                encoder.hidden = [hidden.to(args.device) for hidden in encoder.init_hidden()]

                y_pred_encoder, hidden_encoder = encoder(input_x)
                hidden_decoder = hidden_encoder

                decoder_input = torch.zeros(args.decoder_x_frames, args.batch_size, args.input_dim).to(args.device)

                y_pred_decoder, attention_weight, attn_applied = decoder(decoder_input, hidden_decoder)

                data_out_list.append(attn_applied)

            index_output = data_out_list[0] * args.market_beta  # torch.Size([128, 10])
            stock_output = data_out_list[1]  # torch.Size([128, 10])

            Norm_ = nn.LayerNorm(10, device=args.device)
            stock_output = Norm_(stock_output)

            Transformer_input = index_output + stock_output

            Transformer_input = Transformer_input.transpose(0, 2)

            output1 = transformer(Transformer_input)

            output_ = torch.where(output1 >= 0.5, 1.0, 0.0)

            output_.requires_grad = True

            ACC_metric += ACC(output_, true_y)

        ACC_metric = ACC_metric / len(testloader)

        return ACC_metric



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
        ts = time.time()
        encoder, decoder, transformer, train_loss = train(encoder, decoder, transformer,
                enc_optimizer, dec_optimizer, transformer_optimizer, partition, args)

        encoder, decoder, transformer, val_loss = validation(encoder, decoder, transformer,
                                                             partition, args)

        te = time.time()

        ## 각 에폭마다 모델을 저장하기 위한 코드
        torch.save(encoder.state_dict(), args.new_file_path + '\\' + str(epoch) +'_epoch' +'_encoder' +'.pt')
        torch.save(decoder.state_dict(), args.new_file_path + '\\' + str(epoch) +'_epoch' +'_decoder' +'.pt')
        torch.save(transformer.state_dict(), args.new_file_path + '\\' + str(epoch) +'_epoch' +'_transformer' +'.pt')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print('Epoch {}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'
              .format(epoch, train_loss, val_loss, te - ts))

    ## val_losses에서 가장 값이 최소인 위치를 저장함
    site_val_losses = val_losses.index(min(val_losses)) ## 10 epoch일 경우 0번째~9번째 까지로 나옴
    encoder = args.encoder(args.input_dim, args.hid_dim, args.num_layers, args.batch_size)
    decoder = args.decoder(args.input_dim, args.hid_dim, args.output_dim, args.num_layers, args.batch_size,
                           args.dropout, args.use_bn, args.attention_head, args.attn_size,
                           activation="ReLU")
    transformer = args.transformer(args.trans_feature_size, args.trans_num_laysers,
                                   args.dropout, args.batch_size, args.x_frames, args.trans_nhead)
    encoder.to(args.device)
    decoder.to(args.device)
    transformer.to(args.device)

    encoder.load_state_dict(torch.load(args.new_file_path + '\\' + str(site_val_losses) +'_epoch'+'_encoder' + '.pt'))
    decoder.load_state_dict(torch.load(args.new_file_path + '\\' + str(site_val_losses) +'_epoch' +'_decoder'+ '.pt'))
    transformer.load_state_dict(torch.load(args.new_file_path + '\\' + str(site_val_losses) + '_epoch' + '_transformer' + '.pt'))

    ACC = test(encoder, decoder, transformer,partition, args)
    print('ACC: {}'.format(ACC))

    with open(args.new_file_path + '\\'+ str(site_val_losses)+'Epoch_test_metric' +'.csv', 'w') as fd:
        print('ACC: {}'.format(ACC), file=fd)

    result = {}

    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['ACC'] = ACC

    return vars(args), result



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

args.save_file_path = "C:\\Users\\USER\\PycharmProjects\\DTML\\results"

## ======== data ============= #
args.index = ['^IXIC']


args.stock_list = ['AAPL', 'AMZN', 'BA', 'BAC', 'BHP', 'BRK-B', 'CMCSA', 'CVX', 'D', 'DCM', 'DIS', 'DOW', 'DUK', 'EXC', 'GE', 'GOOGL', 'HD', 'INTC', 'JNJ', 'JPM', 'KO', 'MA', 'MMM', 'MO', 'MRK', 'MSFT', 'NGG', 'NTT', 'NVS', 'ORCL', 'PEP', 'PFE', 'PG', 'PTR', 'RDS-B', 'RIO', 'SO', 'SPY', 'SYT', 'T', 'TM', 'TOT', 'UNH', 'UPS', 'VALE', 'VZ', 'WFC', 'WMT', 'XOM'] # 'CHL','FB'
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
args.input_dim = 10
args.output_dim = 1
args.dropout = 0.15
args.use_bn = True
args.loss_fn = nn.BCELoss()  ## loss function for classification : cross entropy
args.optim = 'Adam'
args.lr = 0.0001
args.l2 = 0.00001 #?
args.epoch = 200

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
##
args.market_beta = 0.1

with open(args.save_file_path + '\\' + 'DTML_result_t.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(["model", "stock", "entire_exp_time",  "avg_test_acc"])

    for stock in args.stock_list:
        est = time.time()
        setattr(args, 'symbol', stock)
        args.new_file_path = args.save_file_path + '\\' + "DTML_" + args.symbol
        os.makedirs(args.new_file_path)

        args.entire_datalist = args.index + [stock]
        # 0번째에 index 1번째에 stock 1개가 input으로 들어감
        trainset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                                args.train_start, args.train_end)
        valset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                              args.validation_start,args.validation_end)
        testset = StockDataset(args.entire_datalist, args.x_frames, args.y_frames,
                               args.test_start, args.test_end)
        partition = {'train': trainset, 'val': valset, 'test': testset}


        setting, result = experiment(partition, args)

        eet = time.time()

        entire_exp_time = eet - est

        # csv파일에 기록하기
        wr.writerow(["DTML", args.symbol,entire_exp_time, result['ACC']])