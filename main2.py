import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

train_df['tweet'] = train_df['tweet'].apply(remove_special_characters)
test_df['tweet'] = test_df['tweet'].apply(remove_special_characters)

train_df['tweet'] = train_df['tweet'].str.lower()
test_df['tweet'] = test_df['tweet'].str.lower()

train_df['tweet'] = train_df['tweet'].apply(word_tokenize)
test_df['tweet'] = test_df['tweet'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
def remove_stop_words(text):
    return [word for word in text if word not in stop_words]

train_df['tweet'] = train_df['tweet'].apply(remove_stop_words)
test_df['tweet'] = test_df['tweet'].apply(remove_stop_words)

from sklearn.utils import resample

# Separate majority and minority classes
df_majority = train_df[train_df.label == 0]
df_minority = train_df[train_df.label == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,      # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
train_df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle data to avoid any bias
train_df_balanced = train_df_balanced.sample(frac=1, random_state=123).reset_index(drop=True)

train_text = train_df_balanced['tweet'].tolist()
train_labels = train_df_balanced['label'].tolist()
test_text = test_df['tweet'].tolist()

train_text = [" ".join(text) for text in train_df_balanced['tweet']]
test_text = [" ".join(text) for text in test_df['tweet']]

train_text = [text.lower() for text in train_text]
test_text = [text.lower() for text in test_text]

vectorizer = CountVectorizer()
train_text = vectorizer.fit_transform(train_text)
test_text = vectorizer.transform(test_text)

train_text, val_text, train_labels, val_labels = train_test_split(train_text, train_labels, test_size=0.2, random_state=0)

from sklearn.model_selection import GridSearchCV

parameters = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
clf = MultinomialNB()
clf = GridSearchCV(clf, parameters, cv=5)
clf.fit(train_text, train_labels)

train_accuracy = clf.score(train_text, train_labels)
print("Train Accuracy: ", train_accuracy)

val_accuracy = clf.score(val_text, val_labels)
print("Validation Accuracy: ", val_accuracy)

from sklearn.metrics import classification_report

train_pred = clf.predict(train_text)
val_pred = clf.predict(val_text)

print("Train classification report:")
print(classification_report(train_labels, train_pred))

print("Validation classification report:")
print(classification_report(val_labels, val_pred))

def classify_text(text):
    # Convert text to numerical representation using vectorizer
    text_vector = vectorizer.transform([text])
    
    # Predict class label using trained classifier (clf)
    prediction = clf.predict(text_vector)
    
    # Return the prediction label as string
    return "Hate speech" if prediction[0] == 1 else "Not hate speech"

# Console input section
if __name__ == "__main__":
    while True:
        text = input("Enter text to classify (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        prediction = classify_text(text)
        print("Prediction: ", prediction)
