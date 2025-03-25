import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

train_texts = [
    'Congratulations! You won a lottery!',
    'Call me back as soon as you can.',
    'You have been selected for a cash prize.',
    'Hi, just checking in. How are you?'
]
train_labels = [1, 0, 1, 0]  # 1 - спам, 0 - не спам

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(train_texts)

mnb = MultinomialNB()
mnb.fit(X_train, train_labels)

new_text = ['Hello, mate!']
vectorized_text = tfidf.transform(new_text)

prediction = mnb.predict(vectorized_text)
print("Клас", prediction[0])  # 1 - спам, 0 - не спам
