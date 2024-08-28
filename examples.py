from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split

from lad.lad import LADClassifier

# Load
X, y = datasets.load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1
)

# Clasisfier
clf = LADClassifier(mode="lazy")
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print(classification_report(y_test, y_hat))
print(clf)
