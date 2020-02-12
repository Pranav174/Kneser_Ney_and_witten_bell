import sys
import itertools 
import spacy
import re
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

unigram = {}
bigram = {}
trigram = {}

def convert_some_to_unk(tokenized_sentences):
    ratio = 0.005
    vocab = {}
    for sent in tokenized_sentences:
        for token in sent:
            if token in vocab.keys():
                vocab[token] += 1
            else:
                vocab[token] = 1
    to_be_changed = int(ratio * len(vocab))
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
    words_to_be_changed = list(vocab.keys())[0:to_be_changed]
    for i,sent in enumerate(tokenized_sentences):
        for j,token in enumerate(sent):
            if token in words_to_be_changed:
                tokenized_sentences[i][j] = '<unk>'
    return tokenized_sentences


def train_on_corpus(corpus='corpus.txt'):
    f = open(corpus, "r").read()
    f = re.sub('}',' ',f)
    f = re.sub('\s+',' ',f)
    doc = nlp(f)
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)
    tokenized_sentences = []
    for sent in sentences:
        tokenized_sentences.append([str(tokens).lower() for tokens in tokenizer(sent)])
    tokenized_sentences = convert_some_to_unk(tokenized_sentences)
    # build the unigram, bigram, and trigram
    for sent in tokenized_sentences:
        for token in sent:
            if token in unigram.keys():
                unigram[token] += 1
            else:
                unigram[token] = 1
    for sent in tokenized_sentences:
        temp = ['<start>'] + sent
        for i in range(len(sent)):
            if temp[i] not in bigram.keys():
                bigram[temp[i]] = {}
            if temp[i+1] not in bigram[temp[i]].keys():
                bigram[temp[i]][temp[i+1]] = 1
            else:
                bigram[temp[i]][temp[i+1]] += 1
    for sent in tokenized_sentences:
        temp = ['<start>','<start>'] + sent
        for i in range(len(sent)):
            if temp[i] not in trigram.keys():
                trigram[temp[i]] = {}
            if temp[i+1] not in trigram[temp[i]].keys():
                trigram[temp[i]][temp[i+1]] = {}
            if temp[i+2] not in trigram[temp[i]][temp[i+1]].keys():
                trigram[temp[i]][temp[i+1]][temp[i+2]] = 1
            else:
                trigram[temp[i]][temp[i+1]][temp[i+2]] += 1

def witten_bell(n, n_gram):
    if n==1:
        # return normal unigram probability
        return unigram[n_gram[0]]/sum([item[1] for item in unigram.items()])
    if n==2:
        unique_follow_words = len(bigram[n_gram[0]])
        total_realizations = sum([item[1] for item in bigram[n_gram[0]].items()])
        backoff_weight = unique_follow_words/(unique_follow_words+total_realizations)
        backoff_prob = backoff_weight * witten_bell(1,n_gram[1:])
        if n_gram[1] in bigram[n_gram[0]].keys():
            return (1-backoff_weight)*(bigram[n_gram[0]][n_gram[1]]/total_realizations) + backoff_prob
        return backoff_prob
    if n==3:
        if n_gram[1] in trigram[n_gram[0]].keys():
            unique_follow_words = len(trigram[n_gram[0]][n_gram[1]])
            total_realizations = sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()])
            backoff_weight = unique_follow_words/(unique_follow_words+total_realizations)
            backoff_prob = backoff_weight * witten_bell(2,n_gram[1:])
            if n_gram[2] in trigram[n_gram[0]][n_gram[1]].keys():
                return (1-backoff_weight)*(trigram[n_gram[0]][n_gram[1]][n_gram[2]]/total_realizations) + backoff_prob
            else:
                return backoff_prob
        else:
            return 0.5 * witten_bell(2,n_gram[1:])

def kneser_ney(n,n_gram,high_order=True):
    d=0.75
    if n==1:
        if high_order:
            return max((unigram[n_gram[0]]-d),0)/sum([item[1] for item in unigram.items()])
        else:
            continuation_count = len(set([item[0] for item in bigram.items() if n_gram[0] in item[1].keys()]))
            return continuation_count/len(bigram.keys())
    if n==2:
        lambd = (d * len(bigram[n_gram[0]]))/(sum([item[1] for item in bigram[n_gram[0]].items()])) 
        if high_order:
            count = bigram[n_gram[0]][n_gram[1]] if n_gram[1] in bigram[n_gram[0]].keys() else 0
            among = sum([item[1] for item in bigram[n_gram[0]].items()])
        else:
            count = 0
            for first in trigram.keys():
                if n_gram[0] in trigram[first].keys():
                    if n_gram[1] in trigram[first][n_gram[0]].keys():
                        count += 1
            among = len(trigram.keys())
        return (max(0,count-d)/among)+(lambd*kneser_ney(1,n_gram[1:],False))
    if n==3:
        try:
            lambd = (d * len(trigram[n_gram[0]][n_gram[1]]))/(sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()]))
        except:
            return d*kneser_ney(2,n_gram[1:],False)
        count = trigram[n_gram[0]][n_gram[1]][n_gram[2]] if n_gram[2] in trigram[n_gram[0]][n_gram[1]].keys() else 0
        among = sum([item[1] for item in trigram[n_gram[0]][n_gram[1]].items()])
        return (max(0,count-d)/among)+(lambd*kneser_ney(2,n_gram[1:],False))

if __name__ == "__main__":
    if len(sys.argv)!=4:
        print("please provide all the arguments")
        exit()
    n = int(sys.argv[1])
    smoothing = witten_bell
    if sys.argv[2]=='k':
        smoothing = kneser_ney
    print("Training on corpus")
    train_on_corpus(sys.argv[3])
    print("Training complete")
    sentence = input("Input sentence: ")
    sentence = [str(tokens).lower() for tokens in tokenizer(sentence)]
    for i,token in enumerate(sentence):
        sentence[i] = token if token in unigram.keys() else "<unk>"
    length = len(sentence)
    for i in range(n-1):
        sentence.insert(0,'<start>')
    ans = 1
    for i in range(length):
        prob = smoothing(n,sentence[i:i+n])
        print("for",sentence[i:i+n],':',prob)
        ans = ans * prob
    print("Final output:", ans)