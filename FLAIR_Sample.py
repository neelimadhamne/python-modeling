#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flair.data import Sentence
from flair.models import SequenceTagger


# In[2]:


# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)


# In[3]:


print(sentence)
print('The following NER tags are found:')
print(sentence.to_tagged_string())


# In[4]:


banker_almanac_text = open("./bankeralmanac.txt",'r')
lines = banker_almanac_text.readlines()
print(lines)


# In[6]:


for sen in lines:
    sentence=Sentence(sen)
    tagger.predict(sentence)
    print(sentence)
    print('{} : NER tags: {}'.format(sentence, sentence.to_tagged_string()))
       


# In[ ]:




