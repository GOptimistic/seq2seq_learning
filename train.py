import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torchinfo import summary as infosummary
import matplotlib.pyplot as plt

from dataset.EN2CNDataset import EN2CNDataset, get_dictionary, load_data
from model import Encoder, Decoder, Attention, Seq2Seq
from utils import infinite_iter, tokens2sentence, computebleu, save_model, load_model, schedule_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "./dataset/cmn-eng"  # 数据集的位置
store_model_path = "./saved_model"  # 储存模型的位置
valid_res_path = "./valid_result"  # 储存模型的位置
max_output_len = 45  # 输出句子的最大长度
batch_size = 64  # batch_size
emb_dim = 256  # word embedding向量的维度
hid_dim = 512  # RNN隐藏状态的维度
n_layers = 4  # RNN的层数
dropout = 0.5  # dropout的概率p
learning_rate = 0.001  # 初始化学习率
# teacher_forcing_ratio = 0.5      # 使用正解训练的概率
summary_steps = 30000  # 总训练batch数
kk = np.argmin([np.abs(summary_steps / 2 - x * np.log(x)) for x in range(1, summary_steps)])

word2int_cn, int2word_cn, cn_vocab_size = get_dictionary(data_path, 'cn')  # 中文字典
word2int_en, int2word_en, en_vocab_size = get_dictionary(data_path, 'en')  # 英文字典
vocab = [word2int_cn, int2word_cn, word2int_en, int2word_en]



training_data = load_data(data_path, 'training')
val_data = load_data(data_path, 'validation')
testing_data = load_data(data_path, 'testing')

'''
打印结果
cn vocab size: 3805
en vocab size: 3922
training dataset size: 18000
validation dataset size: 500
testing dataset size: 2636
'''

train_dataset = EN2CNDataset(training_data, max_output_len, vocab)
val_dataset = EN2CNDataset(val_data, max_output_len, vocab)
test_dataset = EN2CNDataset(testing_data, max_output_len, vocab)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=1)
test_loader = data.DataLoader(test_dataset, batch_size=1)

encoder = Encoder(en_vocab_size, emb_dim, hid_dim, n_layers, dropout)
attention = Attention(hid_dim)
decoder = Decoder(cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, attention)
model = Seq2Seq(encoder, decoder, device).to(device)
print(model)
loss_function = nn.CrossEntropyLoss(ignore_index=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(optimizer)
print('num of trained parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
infosummary(model,
            [(batch_size, max_output_len), (batch_size, max_output_len), (1,)],
            dtypes=[torch.int64, torch.int64, torch.float])

best_loss = 1e9
best_step = 0

# Train and Valid
model.train()
model.zero_grad()
train_losses, val_losses, val_bleu_scores = [], [], []
loss_sum = 0.0
accuracy_sum = 0.0
train_iter = infinite_iter(train_loader)

for step in range(summary_steps):
    model.train()
    sources, targets = next(train_iter)
    sources, targets = sources.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs, preds = model(sources, targets, schedule_sampling(step, summary_steps, c=0, k=kk))
    # targets 的第一个 token 是 '<BOS>' 所以忽略
    outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
    targets = targets[:, 1:].reshape(-1)
    loss = loss_function(outputs, targets)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    accuracy = torch.eq(outputs.argmax(1), targets).float().mean().item()
    accuracy_sum += accuracy

    loss_sum += loss.item()
    if (step + 1) % 10 == 0:
        loss_sum = loss_sum / 10
        accuracy_sum = accuracy_sum / 10
        print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}, Accuracy: {:.3f}".format(step + 1, loss_sum, np.exp(loss_sum), accuracy_sum), end=" ")
        train_losses.append(loss_sum)
        loss_sum = 0.0

        if (step + 1) % 600 == 0:
            # 每600轮 batch 训练进行验证，并存储模型
            model.eval()
            loss_val, bleu_val, acc_val = 0.0, 0.0, 0,0
            n = 0
            result_val = []
            for sources_val, targets_val in val_loader:
                sources_val, targets_val = sources_val.to(device), targets_val.to(device)
                batch_size = sources_val.size(0)
                # print(batch_size)
                outputs_val, preds_val = model.inference(sources_val, targets_val)
                # targets 的第一个 token 是 '<BOS>' 所以忽略
                outputs_val = outputs_val[:, 1:].reshape(-1, outputs_val.size(2))
                targets_val = targets_val[:, 1:].reshape(-1)
                loss = loss_function(outputs_val, targets_val)
                loss_val += loss.item()
                acc_val += torch.eq(outputs_val.argmax(1), targets_val).float().mean().item()

                # 将预测结果转为文字
                targets_val = targets_val.view(sources_val.size(0), -1)
                preds_val = tokens2sentence(preds_val, int2word_cn)
                sources_val = tokens2sentence(sources_val, int2word_en)
                targets_val = tokens2sentence(targets_val, int2word_cn)
                # 记录验证集结果
                for source, pred, target in zip(sources_val, preds_val, targets_val):
                    result_val.append((source, pred, target))
                # 计算 Bleu Score
                bleu_val += computebleu(preds_val, targets_val)
                n += batch_size
            loss_val = loss_val / len(val_loader)
            acc_val = acc_val / len(val_loader)
            bleu_val = bleu_val / n
            val_losses.append(loss_val)
            val_bleu_scores.append(bleu_val)
            print("\n", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, bleu-4 score: {:.3f}, accuracy: {:.3f} ".format(step + 1, loss_val,
                                                                                                np.exp(loss_val),
                                                                                                bleu_val,
                                                                                                acc_val))
            # 储存结果
            with open(valid_res_path + '/valid_output_{}.txt'.format(step + 1), 'w') as f:
                for line in result_val:
                    print(line, file=f)
            if loss_val < best_loss:
                best_loss = loss_val
                best_step = step + 1
            # 储存模型
            save_model(model, optimizer, store_model_path, step + 1)


print("Best loss: {}, best step: {}".format(best_loss, best_step))
# Test
load_model_path = store_model_path + "/model_attention_{}.ckpt".format(best_step)  # 读取模型位置

model = load_model(model, load_model_path)  # 读取模型
model.to(device)
model.eval()
# 测试模型
loss_test, bleu_test, acc_test = 0.0, 0.0, 0.0
n = 0
result = []
for sources_test, targets_test in test_loader:
    sources_test, targets_test = sources_test.to(device), targets_test.to(device)
    batch_size = sources_test.size(0)
    # print(batch_size)
    outputs_test, preds_test = model.inference(sources_test, targets_test)
    # targets 的第一个 token 是 '<BOS>' 所以忽略
    outputs_test = outputs_test[:, 1:].reshape(-1, outputs_test.size(2))
    targets_test = targets_test[:, 1:].reshape(-1)
    loss = loss_function(outputs_test, targets_test)
    loss_test += loss.item()
    acc_test += torch.eq(outputs_test.argmax(1), targets_test).float().mean().item()

    # 将预测结果转为文字
    targets_test = targets_test.view(sources_test.size(0), -1)
    preds_test = tokens2sentence(preds_test, int2word_cn)
    sources_test = tokens2sentence(sources_test, int2word_en)
    targets_test = tokens2sentence(targets_test, int2word_cn)
    for source, pred, target in zip(sources_test, preds_test, targets_test):
        result.append((source, pred, target))
    # 计算 Bleu Score
    bleu_test += computebleu(preds_test, targets_test)
    n += batch_size
loss_test = loss_test / len(test_loader)
acc_test = acc_test / len(test_loader)
bleu_test = bleu_test / n
print('test loss: {}, bleu-4 score: {}, acc: {}'.format(loss_test, bleu_test, acc_test))
# 储存结果
with open('./test_attention_output.txt', 'w') as f:
    for line in result:
        print(line, file=f)

# 绘图
plt.figure(1)
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('train loss')
plt.savefig('./train_loss.png')

plt.figure(2)
plt.plot(range(1, len(val_losses) + 1), val_losses)
plt.xlabel('step per 600')
plt.ylabel('loss')
plt.title('valid loss')
plt.savefig('./valid_loss.png')

plt.figure(3)
plt.plot(range(1, len(val_bleu_scores) + 1), val_bleu_scores)
plt.xlabel('step per 600')
plt.ylabel('bleu')
plt.title('valid bleu')
plt.savefig('./valid_bleu.png')
