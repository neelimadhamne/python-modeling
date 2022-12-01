#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


# In[2]:


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")


# In[5]:


model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


# In[6]:


nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"


# In[7]:


ner_results = nlp(example)
print(ner_results)


# In[21]:


banker_almanac_text = open("./bankeralmanac.txt",'r')


# In[22]:


lines = banker_almanac_text.readlines()
print(lines)


# In[18]:


type(lines)


# In[23]:


ner_results = nlp(lines)
type(ner_results)
print(ner_results)


# In[15]:


banker_almanac_text.close()


# In[ ]:




