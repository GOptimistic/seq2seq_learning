import numpy as np
import torch.utils.data as data
import re
import torch
import os
import json

from tqdm import tqdm


class EN2CNDataset(data.Dataset):
    def __init__(self, data, max_output_len, vocab):
        self.max_output_len = max_output_len
        self.word2int_cn, self.int2word_cn = vocab[0], vocab[1]  # 中文字典
        self.word2int_en, self.int2word_en = vocab[2], vocab[3]  # 英文字典
        self.data = data

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)

    def seq_pad(self, label, pad_token):
        # 将不同长度的句子pad到相同长度，以便训练
        label = np.pad(label, (0, (self.max_output_len - label.shape[0])), mode='constant', constant_values=pad_token)
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # 将英文句子和中文句子分开
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        # print (sentences)
        assert len(sentences) == 2

        # 特殊token
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在句子开头添加'<BOS>'，结尾添加'<EOS>'，字典中没有的词标记为'<UNK>'
        en, cn = [BOS], [BOS]
        # 英文句子分词后转为字典索引向量
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # 中文句子分词后转为字典索引向量
        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)
        # if len(en)>30 or len(cn)>30:
        #    print(len(en),len(cn))

        # 用 '<PAD>' 将句子pad到相同长度
        en = self.seq_pad(en, self.word2int_en['<PAD>'])
        cn = self.seq_pad(cn, self.word2int_cn['<PAD>'])
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        # return英文和中文句子的向量
        return en, cn

# 载入字典
def get_dictionary(root, language):
    with open(os.path.join(root, 'word2int_{}.json'.format(language)), "r") as f:
        word2int = json.load(f)
    with open(os.path.join(root, 'int2word_{}.json'.format(language)), "r") as f:
        int2word = json.load(f)
    print('{} vocab size: {}'.format(language, len(word2int)))
    return word2int, int2word, len(word2int)

# 载入数据 (training/validation/testing)
def load_data(root, set_name):
    data = []
    with open(os.path.join(root, '{}.txt'.format(set_name)), "r") as f:
        for line in f:
            data.append(line)
    print('{} dataset size: {}'.format(set_name, len(data)))

    return data


if __name__ == '__main__':
    data_path = "./cmn-eng"  # 数据集的位置
    max_output_len = 45  # 输出句子的最大长度
    batch_size = 64  # batch_size
    word2int_cn, int2word_cn, cn_vocab_size = get_dictionary(data_path, 'cn')  # 中文字典
    word2int_en, int2word_en, en_vocab_size = get_dictionary(data_path, 'en')  # 英文字典
    vocab = [word2int_cn, int2word_cn, word2int_en, int2word_en]

    training_data = load_data(data_path, 'training')
    val_data = load_data(data_path, 'validation')
    testing_data = load_data(data_path, 'testing')

    train_dataset = EN2CNDataset(training_data, max_output_len, vocab)
    val_dataset = EN2CNDataset(val_data, max_output_len, vocab)
    test_dataset = EN2CNDataset(testing_data, max_output_len, vocab)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    test_loader = data.DataLoader(test_dataset, batch_size=1)

    for batch_en, batch_cn in tqdm(val_loader):
        print(batch_en.shape)
        print(batch_cn.shape)
        print(batch_cn)
        print(batch_cn.dtype)