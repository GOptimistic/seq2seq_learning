import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), '{}/model_attention_{}.ckpt'.format(store_model_path, step))
    return


def load_model(model, load_model_path):
    print('Load model from {}'.format(load_model_path))
    model.load_state_dict(torch.load('{}.ckpt'.format(load_model_path)))
    return model


def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))
    return score


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


def schedule_sampling(step, summary_steps, c, k):
    if c == 0:
        # Inverse sigmoid decay: ϵi = k/(k+exp(i/k))
        # k = np.argmin([np.abs(summary_steps / 2 - x * np.log(x)) for x in range(1, summary_steps)])
        e = k / (k + np.exp(step / k))
    elif c == 1:
        # Linear decay: ϵi = -1/k * i + 1
        e = -1 / summary_steps * step + 1
    elif c == 2:
        # Exponential decay: ϵi = k^i
        e = np.power(0.999, step)
    return e
