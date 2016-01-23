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

# USEFUL REGEX PATTERNS
p_number = re.compile(r'\b[0-9]+\S*', re.I)

#### FEATURE FUNCTIONS
def count_words(s):
	any_word = re.compile('\w+')
	return len(any_word.findall(s))

def count_digits(s):
	global p_number
	return len(p_number.findall(s))

def word_match_list(line):
	words = [word.lower() for word in line[0]]
	sentence = [word.lower() for word in line[1]]
	count = 0
	for word in words:
		if word in sentence:
			count +=1
	return count

def word_match_sentence(line):
	words = [word.lower() for word in line[0]]
	count = 0
	for word in words:
		if word not in ['*','.','?','^']:
			pattern = re.compile(r'\b'+word+r'\b', re.I)
			if pattern.findall(line[1]):
				count+=1
	return count

def lemmatizer(l):
	new_l=[]
	for word in l:
		new_l.append(wordnet_lemmatizer.lemmatize(word))
	return new_l



def digits_match(columns):
	count=0
	global p_number
	for number in p_number.findall(columns[0]):
		pattern = re.compile(r'\b'+number+r'\b', re.I)
		if pattern.findall(columns[1]):
			count+=1
	return count

def remove_digits(column):
	global p_number
	return p_number.sub("",column)


### COMPLETING TABLES WITH FEATURES

full_data['match_d_title'] = full_data[['search_term','product_title']].apply(digits_match, axis=1)
full_data['match_d_description'] = full_data[['search_term','product_description']].apply(digits_match, axis=1)
full_data['count_digits'] =full_data['search_term'].apply(count_digits)
full_data['new_search'] = full_data['search_term'].apply(remove_digits)
full_data['count_words'] =full_data['new_search'].apply(count_words)

print "TOKENIZING"
full_data['tokenized_search'] = full_data['new_search'].apply(word_punct_tokenizer.tokenize)
full_data['tokenized_search'] = full_data['tokenized_search'].apply(lemmatizer)
full_data['tokenized_description'] = full_data['product_description'].apply(word_punct_tokenizer.tokenize)
full_data['tokenized_description'] = full_data['tokenized_description'].apply(lemmatizer)

print "COUNTING MATCHING WORDS IN DESCRIPTION AND TITLE"
full_data['match_w_description'] = full_data[['tokenized_search','tokenized_description']].apply(word_match_list, axis = 1)
full_data['match_w_title'] = full_data[['tokenized_search','product_title']].apply(word_match_sentence, axis = 1)

################
################

real_full_test['match_d_title'] = real_full_test[['search_term','product_title']].apply(digits_match, axis=1)
real_full_test['match_d_description'] = real_full_test[['search_term','product_description']].apply(digits_match, axis=1)
real_full_test['count_digits'] =real_full_test['search_term'].apply(count_digits)
real_full_test['new_search'] = real_full_test['search_term'].apply(remove_digits)
real_full_test['count_words'] =real_full_test['new_search'].apply(count_words)

print "TOKENIZING"
real_full_test['tokenized_search'] = real_full_test['new_search'].apply(word_punct_tokenizer.tokenize)
real_full_test['tokenized_search'] = real_full_test['tokenized_search'].apply(lemmatizer)
real_full_test['tokenized_description'] = real_full_test['product_description'].apply(word_punct_tokenizer.tokenize)
real_full_test['tokenized_description'] = real_full_test['tokenized_description'].apply(lemmatizer)

print "COUNTING MATCHING WORDS IN DESCRIPTION AND TITLE"
real_full_test['match_w_description'] = real_full_test[['tokenized_search','tokenized_description']].apply(word_match_list, axis = 1)
real_full_test['match_w_title'] = real_full_test[['tokenized_search','product_title']].apply(word_match_sentence, axis = 1)



################
# Model fitting
################

def model_fit():

	def in_limits(x):
		if x<1: return 1
		if x>3: return 3
		return x

	print "STARTING MODEL"
	X = full_data[['count_words','count_digits','match_d_title','match_d_description','match_w_title','match_w_description']].values
	y = full_data['relevance'].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	
	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)
	in_limits = np.vectorize(in_limits,otypes=[np.float])
	y_pred = in_limits(y_pred)
	RMSE = mean_squared_error(y_test, y_pred)**0.5
	print "RMSE: ",RMSE

	# for the submission
	real_X_test = real_full_test[['count_words','count_digits','match_d_title','match_d_description','match_w_title','match_w_description']].values
	test_pred = regr.predict(real_X_test)
	test_pred = in_limits(test_pred)

	return test_pred


################################

print "CREATING SUBMISSION FILE"
test_pred = model_fit()
pd.DataFrame({"id": id_test, "relevance": test_pred}).to_csv('submission.csv',index=False)




