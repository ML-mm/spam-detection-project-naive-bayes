import nltk
from nltk import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

dataset = pd.read_csv('SMSSpamCollection.csv', sep='\t', names=['Label', 'Message'])

dataset['Message'] = dataset['Message'].str.replace(r'[^\w\s]', '').str.lower()

nltk.download('punkt')
nltk.download('stopwords')

dataset['Message'] = dataset['Message'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
dataset['Message'] = dataset['Message'].apply(lambda x: [word for word in x if word not in stop_words])

stemming = PorterStemmer()
dataset['Message'] = dataset['Message'].apply(lambda x: [stemming.stem(word) for word in x])

# Vectorization - for the significance of words in csv

dataset['Message'] = dataset['Message'].apply(lambda x: ''.join(x))

vectorize = TfidfVectorizer()  # TF-IDF matrix words in column, messages in row.

X = vectorize.fit_transform(dataset['Message'])

# Labels (essential for classification) i.e. spam or not spam into num.

encoder = LabelEncoder()

y = encoder.fit_transform(dataset['Label'])  # labels turned into num 1 for spam 0 for non spam

# Now to train the model - split test and train data. Then use the Bayes classifier (Naive)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using SMOTE for undersampled class as data is imbalanaced
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = MultinomialNB()
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

folds = 5

scores = cross_val_score(model, X, y, cv=folds, scoring='accuracy')

# Output the results
print(f"Accuracy scores for each fold: {scores}")
print(f"Average accuracy: {scores.mean()}")

# checking if data is imbalanced.
label_c = dataset['Label'].value_counts()

print(label_c)
