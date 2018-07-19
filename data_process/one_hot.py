#-*- coding:utf-8-*-
import numpy as np
from build_vocab_v3 import Vocabulary
import pickle
import json

with open('data/zh_vocab.pkl','rb') as f:
    vocab=pickle.load(f)
#print(vocab.word2idx['皮带'])


f_img=open('data/annotations/img_tag.txt','r')
f_save=open('data/annotations/-1_stop.txt','w')

for line in f_img:
    id,tokens=json.loads(line)
    l=[]
    l=[-1 for i in range(1032)]
    for j in range(len(tokens)):
        try:
            l[j]=vocab.word2idx[tokens[j]]
        except:
            pass
    #print(sum(l))
    x=[id,l,len(tokens)]
    f_save.write('%s\n'%json.dumps(x))
f_img.close()
f_save.close()
