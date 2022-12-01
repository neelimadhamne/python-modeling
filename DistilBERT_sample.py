#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification  # for pytorch
from transformers import TFAutoModelForTokenClassification  # for tensorflow
from transformers import pipeline


# #THis model trained for NER in French
# #It covers following entity Types
#     Date (DAT)
#     Event (EVE)
#     Facility (FAC)
#     Location (LOC)
#     Money (MON)
#     Organization (ORG)
#     Percent (PCT)
#     Person (PER)
#     Product (PRO)
#     Time (TIM)

# In[3]:


model_name_or_path = "HooshvareLab/distilbert-fa-zwnj-base-ner" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Pytorch
# model = TFAutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Tensorflow


# In[4]:


nlp = pipeline("ner", model=model, tokenizer=tokenizer)


# In[5]:


example = "در سال ۲۰۱۳ درگذشت و آندرتیکر و کین برای او مراسم یادبود گرفتند."


# In[6]:


ner_results = nlp(example)
print(ner_results)

