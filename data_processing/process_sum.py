import os
import json
import nltk
from tqdm import tqdm
from datasets import load_dataset
import sentencepice as spm




def save_file(obj, dir, f_name):
    with open(f'data/sum/{dir}/{f_name}', 'w') as f:
        f.write('\n'.join(obj))



def doc_split(docs):
    for doc in docs:
        nltk.tokenize.sent_tokenize(doc)
    return rst


def seq2ids(seq):
    return ids




def process(obj):
    train, valid, test = obj['train'], obj['validation'], obj['test']

    #split src, trg from data obj
    train_src, train_trg = train['article'], train['highlights']
    valid_src, valid_trg = valid['article'], valid['highlights']
    test_src, test_trg = test['article'], test['highlights']


    #split src_doc into sentences
    train_src = doc_split(train_src)
    valid_src = doc_split(valid_src)
    test_src = doc_split(test_src)


    #split trg sentences
    train_trg = [sum.split('\n') for sum in train_trg]
    valid_trg = [sum.split('\n') for sum in valid_trg]
    test_trg = [sum.split('\n') for sum in test_trg]


    #save seq datasets
    save_file(train_src, 'seq', 'train.src')
    save_file(train_trg, 'seq', 'train.trg')

    save_file(valid_src, 'seq', 'valid.src')
    save_file(valid_trg, 'seq', 'valid.trg')

    save_file(test_src, 'seq', 'test.src')
    save_file(test_trg, 'seq', 'test.trg')


    #create concat file and save it to build vocab
    concat_file = []
    save_file(concat_file, 'concat_file')

    #build vocab
    


    #convert tokens into ids
    train_ids = train_src



if __name__ == '__main__':
    os.makedirs('data/cnn', exist_ok=True)
    cnn = load_dataset('cnn_dailymail', '3.0.0')
    nltk.download('punkt')
    process(cnn)