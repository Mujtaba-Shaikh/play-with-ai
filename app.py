from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from openai import OpenAI
client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)
import json

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

# Predict through OpenAI
@app.post("/sentiment-analysis/")
async def sentiment_analysis(request: SentimentRequest):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that can tell the sentiment of the movie reviews."},
        {"role": "user", "content": f"return the sentiment in json format for this sentence:{request.text} And the format should be like this: 'sentiment':''"}
    ],
    response_format={"type": "json_object"}
    )
    sentiment = completion.choices[0].message.content
    # print(sentiment)
    parsed_sentiment = json.loads(sentiment)
    return parsed_sentiment


# Mount the static files directory to serve CSS and JavaScript files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Encode the sentiment column
label_encoder = LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['review'])
X = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(X)

# Model Definition
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class TrainRequest(BaseModel):
    message: str

class PredictRequest(BaseModel):
    review: str

@app.post('/train')
def train(request: TrainRequest):
    # Model Training
    model.fit(X, df['sentiment'], epochs=5, batch_size=128, validation_split=0.2)

    # Save Model
    model.save('sentiment_analysis_model.h5')

    return {'message': 'Model trained successfully'}

@app.post('/predict')
def predict(request: PredictRequest):
    # Load Model for Prediction
    model = load_model('sentiment_analysis_model.h5')

    # Tokenize and pad sequence
    new_review_seq = tokenizer.texts_to_sequences([request.review])
    new_review_seq = pad_sequences(new_review_seq, maxlen=X.shape[1])

    # Predict sentiment
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

    return {'sentiment': sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
