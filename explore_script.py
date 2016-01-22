import numpy as np
import pandas as pd
import re, csv
#plotting
import matplotlib.pyplot as plt
import seaborn as sns
#fitting methods
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer

training_data = pd.read_csv("train.csv")
attribute_data = pd.read_csv("attributes.csv")
descriptions = pd.read_csv("product_descriptions.csv")
real_test_data = pd.read_csv("test.csv")

sub = training_data.copy()
id_test = real_test_data['id']

full_data = pd.merge(sub,descriptions, on = "product_uid")
real_full_test = pd.merge(real_test_data,descriptions, on = "product_uid")


word_punct_tokenizer = WordPunctTokenizer()
wordnet_lemmatizer = WordNetLemmatizer()


#### FEATURE FUNCTIONS
def count_words(s):
	any_word = re.compile('\w+')
	return len(any_word.findall(s))

def count_with_digits(s):
	any_word = re.compile('\d+')
	return len(any_word.findall(s))

def word_match_list(line):
	words = [word.lower() for word in line[0]]
	sentence = [word.lower() for word in line[1]]
	count = 0
	for word in words:
		if word in sentence: count +=1
	return count

def word_match_sentence(line):
	words = [word.lower() for word in line[0]]
	count = 0
	for word in words:
		if word not in ['*','.','?','^']:
			pattern = re.compile(r'\b'+word+r'\b', re.I)
			if pattern.findall(line[1]): count+=1
	return count

def lemmatizer(l):
	new_l=[]
	for word in l:
		new_l.append(wordnet_lemmatizer.lemmatize(word))
	return new_l

def in_limits(x):
	if x<1: return 1
	if x>3: return 3
	return x


### COMPELTING TABLES WITH FEATURES

full_data['tokenized_search'] = full_data['search_term'].apply(word_punct_tokenizer.tokenize)
full_data['tokenized_search'] = full_data['tokenized_search'].apply(lemmatizer)
print "TOKENIZED SEARCH"
full_data['tokenized_description'] = full_data['product_description'].apply(word_punct_tokenizer.tokenize)
full_data['tokenized_description'] = full_data['tokenized_description'].apply(lemmatizer)
print "TOKENIZED DESCRIPTION"

full_data['count_words'] =full_data['search_term'].apply(count_words)
full_data['count_with_digits'] =full_data['search_term'].apply(count_with_digits)
full_data['count_in_description'] = full_data[['tokenized_search','tokenized_description']].apply(word_match_list, axis = 1)
full_data['count_in_title'] = full_data[['tokenized_search','product_title']].apply(word_match_sentence, axis = 1)

real_full_test['tokenized_search'] = real_full_test['search_term'].apply(word_punct_tokenizer.tokenize)
real_full_test['tokenized_search'] = real_full_test['tokenized_search'].apply(lemmatizer)
print "TOKENIZED SEARCH"
real_full_test['tokenized_description'] = real_full_test['product_description'].apply(word_punct_tokenizer.tokenize)
real_full_test['tokenized_description'] = real_full_test['tokenized_description'].apply(lemmatizer)
print "TOKENIZED DESCRIPTION"

real_full_test['count_words'] =real_full_test['search_term'].apply(count_words)
real_full_test['count_with_digits'] =real_full_test['search_term'].apply(count_with_digits)
real_full_test['count_in_description'] = real_full_test[['tokenized_search','tokenized_description']].apply(word_match_list, axis = 1)
real_full_test['count_in_title'] = real_full_test[['tokenized_search','product_title']].apply(word_match_sentence, axis = 1)




################
# Model fitting
################

print "STARTING MODEL"

X = full_data[['count_words','count_with_digits', 'count_in_description','count_in_title']].values
y = full_data['relevance'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)


real_X_test = real_full_test[['count_words','count_with_digits', 'count_in_description','count_in_title']].values

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

#y_pred = regr.predict(X_test)

in_limits = np.vectorize(in_limits,otypes=[np.float])

#y_pred = in_limits(y_pred)

test_pred = regr.predict(real_X_test)
test_pred = in_limits(test_pred)

#RMSE = mean_squared_error(y_test, y_pred)**0.5

#print RMSE


################################

print "CREATING SUBMISSION FILE"

pd.DataFrame({"id": id_test, "relevance": test_pred}).to_csv('submission.csv',index=False)




