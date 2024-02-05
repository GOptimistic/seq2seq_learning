import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, hid_dim * 2)

    def forward(self, input):
        # input = [batch size, sequence len]
        batch_size = input.shape[0]
        embedding = self.embedding(input)
        # embedding = [batch size, sequence len, emb_dim]
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid_dim * directions]
        # hidden =  [n_layers * directions, batch size, hid_dim]

        # 因为 Encoder 是双向RNN，所以需要对同一层两个方向的 hidden state 进行拼接
        # hidden = [num_layers * directions, batch size, hid dim] --> [num_layers, directions, batch size, hid dim]
        hidden = hidden.view(self.n_layers, 2, batch_size, -1)
        # s = [num_layers, batch size, hid dim * 2]
        s = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        s = torch.tanh(self.fc(s))
        return outputs, s


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2 + hid_dim * 2, hid_dim * 2, bias=False)
        self.v = nn.Linear(hid_dim * 2, 1, bias=False)

    def forward(self, enc_output, s):
        # s = [num_layers, batch_size, hid_dim * 2]
        # enc_output = [batch_size, seq_len, hid_dim * 2]
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        # s_attn = [num_layers, batch_size, seq_len, hid_dim * 2] -> [batch_size, seq_len, hid_dim * 2]
        s_attn = s.unsqueeze(2).repeat(1, 1, seq_len, 1)
        s_attn = torch.mean(s_attn, 0)
        # E = [batch_size, seq_len, hid_dim * 2]
        E = torch.tanh(self.attn(torch.cat((s_attn, enc_output), dim=2)))
        # attention = [batch_size, seq_len]
        attention = self.v(E).squeeze(2)
        # return result: [batch_size, seq_len]
        return nn.functional.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2  # 因为 Encoder 是双向的
        self.n_layers = n_layers
        self.attention = attention
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim + self.hid_dim, self.hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim + self.hid_dim + emb_dim, cn_vocab_size)
        # self.embedding2vocab2 = nn.Linear((self.hid_dim + self.hid_dim + emb_dim) * 2, cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, s, enc_output):
        # input = [batch size]
        # s = [num_layers, batch_size, hid_dim * 2]
        # enc_output = [batch_size, seq_len, hid_dim * 2]
        # Decoder 是单向，所以 directions=1
        input = input.unsqueeze(1)
        # embedded = [batch size, 1, emb_dim]
        embedded = self.dropout(self.embedding(input))
        # a = [batch_size, 1, seq_len]
        a = self.attention(enc_output, s).unsqueeze(1)
        # c = [batch_size, 1, hid_dim * 2]
        c = torch.bmm(a, enc_output)
        # rnn_input = [batch_size, 1, emb_dim + hid_dim * 2]
        rnn_input = torch.cat((embedded, c), dim=2)
        # dec_output = [batch_size, 1, hid_dim * 2]
        # s = [num_layers, batch_size, hid_dim * 2]
        dec_output, s = self.rnn(rnn_input, s)

        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # 将 RNN 的输出向量的维数转换到target语言的字典大小
        # output = [batch size, vocab size]
        output = self.embedding2vocab1(torch.cat((dec_output, c, embedded), dim=1))
        # output = self.embedding2vocab2(output)

        return output, s


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len]
        # target = [batch size, target len]
        # teacher_forcing_ratio 是使用正解训练的概率
        print(input.shape)
        print(target.shape)
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(input)

        dec_input = target[:, 0]  # 'BOS'
        preds = []
        for t in range(1, target_len):
            output, s = self.decoder(dec_input, s, encoder_outputs)
            outputs[:, t] = output
            # 决定是否用正解来训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # teacher force 为 True 用正解训练，否则用预测到的最大概率的词训练
            dec_input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target):
        # 测试模型
        # input  = [batch size, input len]
        # target = [batch size, target len]
        batch_size = input.shape[0]
        input_len = input.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 准备一个tensor存储输出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # encoder_outputs用来计算attention，s 用来初始化 Decoder
        encoder_outputs, s = self.encoder(input)

        dec_input = target[:, 0]  # 'BOS'
        preds = []
        for t in range(1, input_len):
            output, s = self.decoder(dec_input, s, encoder_outputs)
            outputs[:, t] = output
            # 取出输出概率最大的词
            top1 = output.argmax(1)
            # 用预测到的最大概率的词进行下一步预测
            dec_input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds