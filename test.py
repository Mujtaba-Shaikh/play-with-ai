import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

df.head(10)


# Encode the sentiment column
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['review'])
X = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(X)


# Load Model for Prediction
from tensorflow.keras.models import load_model
model = load_model('sentiment_analysis_model.h5')


# Example Prediction
new_review = "Spiderman movie was not good"
new_review_seq = tokenizer.texts_to_sequences([new_review])
new_review_seq = pad_sequences(new_review_seq, maxlen=X.shape[1])
prediction = model.predict(new_review_seq)
print(prediction)

# Convert prediction to human-readable format
threshold_positive = 0.6
threshold_negative = 0.4
if prediction > threshold_positive:
    sentiment = "positive"
elif prediction < threshold_negative:
    sentiment = "negative"
else:
    sentiment = "neutral"

print(f"The sentiment of the review is: {sentiment}")