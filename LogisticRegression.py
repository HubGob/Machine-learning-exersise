from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# Load dataset
df = pd.read_csv("enron_spam_data.csv", delimiter=',', header=None, quotechar='"', names=['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date'])
df = df.drop([0])  # Drop the first row which may be a header row
df = df.dropna()  # Drop rows with missing values

# Map labels to binary values (spam = 1, ham = 0)
df["Spam/Ham"] = df["Spam/Ham"].map({"spam": 1, "ham": 0})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Spam/Ham"], test_size=0.3, random_state=42)

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# To predict the new message:
new_message = ["Hey John, I hope youre doing well! I wanted to follow up on the meeting we had last week. Could you send me the presentation slides when you have a moment?Thanks, Sarah"
]

# Transform the new message using the same vectorizer
new_message_vec = vectorizer.transform(new_message)

# Predict using the trained model
new_message_pred = model.predict(new_message_vec)

# Print the prediction (1 = spam, 0 = ham)
print(f"Prediction for '{new_message[0]}': {'spam' if new_message_pred[0] == 1 else 'ham'}")
