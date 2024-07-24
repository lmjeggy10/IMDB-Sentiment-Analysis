import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv('preprocessed_IMDB_Dataset.csv')

# Split the data
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# SGD Classifier with Linear SVM
sgd_svm_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
sgd_svm_model.fit(X_train_vect, y_train)
sgd_svm_pred = sgd_svm_model.predict(X_test_vect)

print("SGD Classifier with Linear SVM Accuracy:", accuracy_score(y_test, sgd_svm_pred))
print(classification_report(y_test, sgd_svm_pred))
