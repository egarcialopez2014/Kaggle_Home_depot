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



training_data = pd.read_csv("train.csv")
attribute_data = pd.read_csv("attributes.csv")
descriptions = pd.read_csv("product_descriptions.csv")
#real_test_data = pd.read_csv("test.csv")

sub = training_data.copy()
#id_test = real_test_data['id']

full_data = pd.merge(sub,descriptions, on = "product_uid")
#real_full_test = pd.merge(test_data,descriptions, on = "product_uid")


#### FEATURE FUNCTIONS
def count_words(s):
	any_word = re.compile('\w+')
	return len(any_word.findall(s))

def count_with_digits(s):
	any_word = re.compile('\d+')
	return len(any_word.findall(s))

def word_match(line):
	any_word = re.compile('\w+', re.I)
	words = any_word.findall(line[0])
	count = 0
	for word in words:
		pattern = re.compile(r'\b'+word+r'\b', re.I)
		if pattern.findall(line[1]): count+=1
	return count

def in_limits(x):
	if x<1: return 1
	if x>3: return 3
	return x


### COMPELTING TABLES WITH FEATURES

full_data['count_words'] =full_data['search_term'].apply(count_words)
full_data['count_with_digits'] =full_data['search_term'].apply(count_with_digits)
full_data['count_in_description'] = full_data[['search_term','product_description']].apply(word_match, axis = 1)
full_data['count_in_title'] = full_data[['search_term','product_title']].apply(word_match, axis = 1)

# full_test['count_words'] =full_test['search_term'].apply(count_words)
# full_test['count_with_digits'] =full_test['search_term'].apply(count_with_digits)
# full_test['count_in_description'] = full_test[['search_term','product_description']].apply(word_match, axis = 1)
# full_test['count_in_title'] = full_test[['search_term','product_title']].apply(word_match, axis = 1)


################
# Model fitting
################

print "STARTING MODEL"

X = full_data[['count_words','count_with_digits', 'count_in_description','count_in_title']].values
y = full_data['relevance'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#real_X_test = real_full_test[['count_words','count_with_digits', 'count_in_description','count_in_title']].values

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

in_limits = np.vectorize(in_limits,otypes=[np.float])

y_pred = in_limits(y_pred)

RMSE = mean_squared_error(y_test, y_pred)**0.5

print RMSE


################################

#print "CREATING SUBMISSION FILE"

#pd.DataFrame({"id": id_test, "relevance": rel_test}).to_csv('submission.csv',index=False)




