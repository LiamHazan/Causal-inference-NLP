import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from pip._internal.utils import logging
from wordfreq import word_frequency
from wordfreq import top_n_list
import nltk
# nltk.download()
# from words_class import syn_df
import csv
import string
import json
import pandas as pd
# import gensim
# from gensim.models import Word2Vec
import numpy as np
# from torchtext.vocab import Vocab
import torchtext



class RegisterConverter():
    def __init__(self,classified_words_path, syn_path, vocab_file, LOW_REG_THRESH=0.1,
    HIGH_REG_THRESH=0.5, HIGH_FREQ_THRESH=0.0007, LOW_FREQ_THRESH = 1e-06,
    SENTIM_DIFF=0.35, REG_DIFF=0.24, FREQ_DIFF=0.5, MIN_FREQ=-np.log(1 / (10 ** 8)),
    WORD_EMB_DIM=100):

        self.BERT_VOCAB = set()
        for x in vocab_file:
            self.BERT_VOCAB.add(x[:-1])
        self.syn_df = pd.read_csv(syn_path)
        self.class_df = pd.read_csv(classified_words_path)
        self.class_df = self.class_df[['Word', 'PoS', 'Level.Predicted.RF', 'Level.Predicted.NN', 'Level.Predicted.SVM']]
        columns_dict = {key: key[-2:] for key in self.class_df.columns[2:]}
        self.class_df.rename(columns=columns_dict, inplace=True)
        self.class_df.rename(columns={'VM': 'SVM'}, inplace=True)
        classes_to_numeric = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        for c in classes_to_numeric.keys():
            self.class_df.replace(c, classes_to_numeric[c], inplace=True)

        self.LOW_REG_THRESH = LOW_REG_THRESH
        self.HIGH_REG_THRESH = HIGH_REG_THRESH
        self.HIGH_FREQ_THRESH = HIGH_FREQ_THRESH
        self.LOW_FREQ_THRESH = LOW_FREQ_THRESH
        self.SENTIM_DIFF = SENTIM_DIFF
        self.REG_DIFF = REG_DIFF
        self.FREQ_DIFF = FREQ_DIFF
        self.MIN_FREQ = MIN_FREQ
        self.WORD_EMB_DIM = WORD_EMB_DIM

        self.glove = torchtext.vocab.GloVe(name="6B", dim=self.WORD_EMB_DIM)
        self.sia = SentimentIntensityAnalyzer()
        self.syn_dict = {}
        self.adjectives = set()


    def calc_avg(self, word):
        cond = (self.class_df['Word'] == word) & (self.class_df['PoS'] == 'ADJ')
        # print(word,self.class_df.loc[cond].values)
        if len(self.class_df.loc[cond].values)==0:
            return 0
        entry = self.class_df.loc[cond].values[0]
        # print(entry)
        sum = 0
        count = 0
        for i in range(2, 5):
            if entry[i] != 'Unknown':
                sum += entry[i]
                count += 1
        # print(avg)
        if count==0: return 0
        return (sum / count) / 6 # normalized


    def is_adj(self, syn):
        cond = (self.syn_df['word'] == syn) & (self.syn_df['pos'] == 'adj')
        synonyms = self.syn_df.loc[cond].values
        if len(synonyms) == 0: return False
        return True


    def best_suit(self, word, subs, word_sentim, word_score, word_emb):
        frequencies = []
        emb_vectors = []
        word_emb = word_emb/sum(word_emb)
        for i,sub in enumerate(subs):
            emb_vectors.append(self.glove[sub])
            # print(emb_vectors[i], type(emb_vectors[i]))
            # sub_freq = word_frequency(sub, 'en')
            # if sub_freq > 0:
            #     frequencies.append(-np.log(sub_freq) / self.MIN_FREQ)  # normalized
            # else: frequencies.append(sub_freq)
        argmin, minimum = word, np.infty
        for i,cand in enumerate(subs):
            # cand_sentim = self.sia.polarity_scores(cand)['compound']
            # sentim_factor = abs(word_sentim - cand_sentim) / 2  # normalized (values between -1 to 1)
            # cand_score = round(self.calc_avg(cand), 3)
            # if cand_score == 0 and frequencies[i] > 0:
            #     cand_score = -np.log(frequencies[i]) / self.MIN_FREQ  # normalized
            # elif cand_score == 0 and frequencies[i] == 0:
            #     cand_score = 1
            # register_factor = abs(word_score - cand_score)
            cand_emb = emb_vectors[i]/sum(emb_vectors[i])
            emb_factor = torch.sum(abs(word_emb - cand_emb))
            # value = (sentim_factor+register_factor+emb_factor/3)  # the lower the better
            value = emb_factor
            # print("value = ",value)
            if value < minimum:
                minimum = value
                argmin = cand
        return argmin


    def find_substitute(self,word,word_score,lower):
        if word in self.syn_dict.keys():
            synonyms = self.syn_dict[word]
        else:
            cond = (self.syn_df['word'] == word) & (self.syn_df['pos'] == 'adj') & (len(self.syn_df['synonyms'][0]) > 1)
            # cond = (self.syn_df['word'] == word) & (self.syn_df['pos'] == 'adj')
            synonyms = self.syn_df.loc[cond]['synonyms'].values
            for i,syn in enumerate(synonyms):
                syn = syn[2:-2]
                syn = syn.split(', ')
                for j in range(len(syn)):
                    syn[j] = syn[j][1:-1]
                synonyms[i] = syn
            self.syn_dict[word] = synonyms

        subs = []
        word_sentim = self.sia.polarity_scores(word)['compound']
        # word_score = round(self.calc_avg(word), 3)
        word_freq = word_frequency(word, 'en')
        for option in synonyms:
            if len(option[0])>0:
                for syn in option:
                    if syn not in self.BERT_VOCAB: continue
                    # try:
                    syn_pos = nltk.pos_tag([syn])
                    if syn_pos == [(syn, 'JJR')] or syn_pos == [(syn, 'RBR')]:
                        continue
                    syn_freq = word_frequency(syn, 'en')
                    syn_sentim = self.sia.polarity_scores(syn)['compound']
                    # if abs(word_sentim - syn_sentim) > self.SENTIM_DIFF or \
                    #         syn_freq > self.HIGH_FREQ_THRESH: continue
                    if abs(word_sentim-syn_sentim)>self.SENTIM_DIFF or word in syn:
                        continue
                    if syn not in self.adjectives:
                        if self.is_adj(syn):
                            self.adjectives.add(syn)
                        else:
                            continue  # separated if for faster function computation

                    # print("syn is adj")
                    syn_score = round(self.calc_avg(syn),3)
                    if syn_score == 0 and syn_freq>0:
                        syn_score = -np.log(syn_freq) / self.MIN_FREQ  # normalized
                    elif syn_score == 0 and syn_freq==0:
                        syn_score = 1
                    if lower == False:
                        if syn_score > self.HIGH_REG_THRESH and abs(syn_score - word_score) <= self.REG_DIFF and \
                                abs(-np.log(syn_freq)/self.MIN_FREQ - -np.log(word_freq)/self.MIN_FREQ) <= self.FREQ_DIFF:
                            subs.append(syn)
                    else:
                        if syn_score <= self.HIGH_REG_THRESH and abs(syn_score - word_score) <= self.REG_DIFF and \
                                abs(-np.log(syn_freq)/self.MIN_FREQ - -np.log(word_freq)/self.MIN_FREQ) <= self.FREQ_DIFF:
                            subs.append(syn)
                    # except:
                    #     continue

        # chosen = min(subs, key=lambda x: word_frequency(x, 'en'))
        # chosen = min(subs, key=lambda x: abs(word_sentim-sia.polarity_scores(x)['compound']))
        # print("subs = ", subs)
        if len(subs) == 0: return word, 0
        word_emb = self.glove[word]
        chosen = self.best_suit(word, subs, word_sentim,word_score,word_emb)
        self.syn_dict[word][0] = [chosen]
        # print("chosen = ",chosen)
        chosen_freq = word_frequency(chosen, 'en')
        substitute_score = round(self.calc_avg(chosen), 3)
        if substitute_score == 0 and chosen_freq > 0:
            substitute_score = -np.log(chosen_freq) / self.MIN_FREQ  # normalized
        elif substitute_score == 0 and chosen_freq == 0:
            substitute_score = 1

        return chosen, substitute_score


    def change_paragraph(self, para, lower=False):
        print('\n')
        # print(para)
        tokenized = nltk.word_tokenize(para)
        # print(tokenized)
        separator = ' '
        tagged_words = nltk.pos_tag(tokenized)
        changes = 0
        subs_scores = 0
        words_scores = 0
        for i, (word, tag) in enumerate(tagged_words):
            if word not in self.BERT_VOCAB: continue
            if tag=='JJ' and ~word.endswith('.') and ~word.startswith('.'):
                # print(word, tag)
                # print(word, self.calc_avg(word))
                word = word.lower()
                word_avg = round(self.calc_avg(word),3)
                word_freq = word_frequency(word,'en')
                if word_avg == 0 and word_freq > 0:
                    word_avg = -np.log(word_freq) / self.MIN_FREQ  # normalized
                elif word_avg == 0 and word_freq == 0:
                    word_avg = 1
                # print("avg = ", word_avg)
                if lower == False:
                    if self.LOW_REG_THRESH <= word_avg <= self.HIGH_REG_THRESH and \
                            self.HIGH_FREQ_THRESH > word_freq > 0:  # if low score
                        substitute, score = self.find_substitute(word,word_avg,lower)
                        if substitute != word:
                            print(f"{word} ({word_avg}) --> {substitute} ({score})")
                            subs_scores += score
                            words_scores += word_avg
                            changes += 1
                            tokenized[i] = substitute
                else:
                    if word_avg > self.HIGH_REG_THRESH and self.HIGH_FREQ_THRESH > word_freq > 0:  # if low score
                        substitute, score = self.find_substitute(word,word_avg,lower)
                        if substitute != word:
                            print(f"{word} ({word_avg}) --> {substitute} ({score})")
                            subs_scores += score
                            words_scores += word_avg
                            changes += 1
                            tokenized[i] = substitute

        restored_para = separator.join(tokenized)
        restored_para = restored_para.replace(' .', '.')
        restored_para = restored_para.replace(' ,', ',')
        restored_para = restored_para.replace(' \'', '\'')
        restored_para = restored_para.replace(' < br / > < br / > ', '<br /><br />')
        if changes > 0:
            return restored_para, changes, round((subs_scores - words_scores) / changes, 3)
        return restored_para, changes, 0



if __name__ == '__main__':
    syn_file = 'syn_file.csv'
    # print(syn_df)
    imdb_file = "IMDB_Dataset.csv"
    imdb_df = pd.read_csv(imdb_file)
    class_file = "AllWordsClassified.csv"
    vocab_file = open('vocabulary.txt', "r", encoding="utf8")
    # s = set()
    # for x in vocab_file:
    #     # print(x)
    #     s.add(x[:-1])
    #
    # for x in s:
    #     print(x)
    #     for y in x:
    #         print(y)
    #     break
    #
    # for y in s:
    #     for i in range(len(y)):
    #         print(y[i]=='\n')
    #     break
    #
    # print('incompatible' in s)
    # exit()


    LOW_REG_THRESH = 0.1
    HIGH_REG_THRESH = 0.5
    HIGH_FREQ_THRESH = 0.0007
    LOW_FREQ_THRESH = 1e-06
    SENTIM_DIFF = 0.15
    REG_DIFF = 0.24
    FREQ_DIFF = 0.5
    MIN_FREQ = -np.log(1 / (10 ** 8))
    WORD_EMB_DIM = 100

    converter = RegisterConverter(class_file,syn_file,vocab_file,LOW_REG_THRESH,HIGH_REG_THRESH,
                                  HIGH_FREQ_THRESH,LOW_FREQ_THRESH,SENTIM_DIFF,
                                  REG_DIFF,FREQ_DIFF,MIN_FREQ,WORD_EMB_DIM)

    ### set the direction of the desired register change ###
    lower = False

    for i in range(30):
        paragraph = imdb_df.iloc[i]['review']
        new_para,changes,avg_change = converter.change_paragraph(paragraph,lower)
        print(f"average level change of this paragraph: {avg_change}")
        print(f"{changes} changes were made in this paragraph")
        # print(new_para)




# import pickle
# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('vectors.txt', binary=False)
#
# filename = 'model.pkl'
# pickle.dump(model, open(filename, 'wb'))
#
# # loaded_model = pickle.load(open(filename, 'rb'))
# # result = loaded_model.score(X_test, Y_test)
#
# # if you vector file is in binary format, change to binary=True
# sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
# vectors = [model[w] for w in sentence]

# print(imdb_df.iloc[0]['review'])