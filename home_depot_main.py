import numpy as np
import pandas as pd
import re, csv

from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.data import load
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

from collections import defaultdict, OrderedDict

from hd_functions import remove_bad_words, digits_match, count_digits, word_punct_tokenizer, word_match_sentence, word_match_list, lemmatizer, count_words, remove_digits



##################TRANSFORM FUNCTIONS############################

def tag_list(column, list_of_words):
    new_column=[]
    for word in column:
        if word.lower() in list_of_words:
            new_column.append(word.lower())
    return new_column

def remove_list(column, list_of_words):
    for word in column:
        if word.lower() in list_of_words:
            column.remove(word)
    return column

def dict_switch_tuples(column):
    d = {}
    for t in column:
        if t[1] in d: d[t[1]].append(t[0])
        else: d[t[1]]=[t[0]]
    return d

def get_tags(column, tag):
    if tag not in column.keys():return []
    return column[tag]

#######################

def match_texts(columns):
    count = 0
    search_terms = columns[0]
    text = columns[1]
    for word in search_terms:
        try:
            reg = re.compile(r'\b'+word+r'\b', re.I)
            if reg.search(text): count += 1
        except:
            pass
    return count

def both_manuf(columns):
    if columns[0]==0: return 0
    elif columns[1]==1:return 1
    else: return -1


def in_limits(x):
    global possibles
    if x<1: return 1
    if x>3: return 3
    for i in range(len(possibles)-1):
        if x>=possibles[i] and x<=possibles[i+1]:
            if (x-possibles[i])<(possibles[i+1]-x):
                return possibles[i]
            else:
                return possibles[i+1]

def freq_of_words(df):
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    l_badwords = tagdict['IN'][1].split(" ") + tagdict['DT'][1].split(" ") + tagdict['CC'][1].split(" ")
    search_terms = df['search_term'].values
    d  = defaultdict(int)
    for text in search_terms:
        for word in text.split():
            if word not in l_badwords:
                d[word]=d[word]+1
    d2 = OrderedDict(sorted(d.items(), key=lambda t: t[1]))
    N=len(search_terms)
    for key, value in d2.items(): d2[key]=value*1.0/N
    return d2

def weight_words(column, d_freq):
    total = 0
    for word in column.split():
        if word in d_freq:
            total = total + d_freq[word]
    return total

def my_tagging(column, tagger):
    return nltk.tag._pos_tag(column, tagger=tagger, tagset='universal')



# IMPORTING AND JOINING FILES
print ("LOADING ....")
training = pd.read_csv("train.csv", engine='python')
real_test = pd.read_csv("test.csv", engine='python')
attributes = pd.read_csv("attributes.csv", engine='python')
descriptions = pd.read_csv("product_descriptions.csv", engine='python')

len_training = len(training)
id_test = real_test['id']
df = pd.concat([training,real_test], axis = 0, ignore_index=True)
df = pd.merge(df,descriptions, how="left", on = "product_uid")

attributes['new_value']=attributes['value'].apply(lambda s : str(s)+"  ")
grouped_at = attributes.groupby('product_uid')
group_at = grouped_at[['new_value']].sum()
group_at['product_uid']=group_at.index
group_at.columns = ['attributes','product_uid']

df = pd.merge(df,group_at,on="product_uid",how='left')
df['attributes'].fillna("NO_ATTRIBUTES", inplace=True)

possibles = list(np.sort(training['relevance'].unique()))

# EXTRACTING INTERESTING ATTRIBUTES AND CREATING REGEX PATTERNS

print ("CREATING INTERESTING TAGS ....")

attributes['value']=attributes['value'].str.lower()
manuf = attributes.where(attributes['name']=='MFG Brand Name')
list_manufacturers = manuf['value'].unique()
material = attributes.where(attributes['name']=='Material')
list_materials = material['value'].unique()


### MAIN PROGRAM BODY

print ("BUILDING COUNT AND MATCH FEATURES ...")
print ("...Building tokeinized_search")

df['tokenized_search'] = df['search_term'].apply(word_tokenize)

# print ("...Building MANUF_search")
# df['MANUF_search'] = df['tokenized_search'].apply(tag_list, args=(list_manufacturers,))
# df['tokenized_search'] = df['tokenized_search'].apply(remove_list, args=(list_manufacturers,))
# print ("...Building MATERIAL_search")
# df['MATERIAL_search'] = df['tokenized_search'].apply(tag_list, args=(list_materials,))
# df['tokenized_search'] = df['tokenized_search'].apply(remove_list, args=(list_materials,))
# print ("...Building tagged_search")
# df['tagged_search'] = df['tokenized_search'].apply(my_tagging, args=(tagger,))
# print ("tags are built, now dicts")
# df['tagged_search']=df['tagged_search'].apply(dict_switch_tuples)
# print ("...Building the rest_search")
# df['NOUN_search']=df['tagged_search'].apply(get_tags, args=('NOUN',))
# df['NUM_search']=df['tagged_search'].apply(get_tags, args=('NUM',))
# df['ADJ_search']=df['tagged_search'].apply(get_tags, args=('ADJ',))
# df['VERB_search']=df['tagged_search'].apply(get_tags, args=('VERB',))

print ("....Building match_XXX_")

#for tag in ['MANUF','NOUN','NUM','ADJ','VERB','MATERIAL']:
for tag in ['tokenized']:
    for place in ['product_title','product_description','attributes']:
        column_name = "match_"+tag+"_"+place
        df[column_name]= df[[tag+'_search',place]].apply(match_texts, axis=1)

print ("....Building count_XXX")
#for tag in ['MANUF','NOUN','NUM','ADJ','VERB','MATERIAL']:
for tag in ['tokenized']:
    df["count_"+tag]=df[tag+'_search'].apply(len)


# d_freq=freq_of_words(df)
# df['weighted_words']=df['search_term'].apply(weight_words, args=(d_freq,))


################
# Model fitting
################

print ("RUNNING MODEL ....")

# df = df.drop(['id','search_term','product_uid','product_title','attributes','product_description','tagged_search','tokenized_search','NOUN_search','MANUF_search','MATERIAL_search','NUM_search','VERB_search','ADJ_search'], axis=1)
df2 = df.drop(['id','search_term','product_uid','product_title','attributes','product_description','tokenized_search'], axis=1)
df_train = df2.iloc[:len_training]
df_test = df2.iloc[len_training:]

# ################################
X = df_train.drop('relevance', axis=1).values
y = df_train['relevance'].values

df_test = df_test.drop('relevance', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#in_limits = np.vectorize(in_limits,otypes=[np.float])
#y_pred = in_limits(y_pred)
RMSE = mean_squared_error(y_test, y_pred)**0.5
print ("RMSE: ",RMSE)

# # for the submission

# test_pred = clf.predict(df_test)
# test_pred = in_limits(test_pred)


# print ("CREATING SUBMISSION FILE")

# pd.DataFrame({"id": id_test, "relevance": test_pred}).to_csv('submission.csv',index=False)




