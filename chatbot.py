import os
from openai import OpenAI

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

message=[{"role": "system", "content": "You are a helpful assistant chatbot"}]

while(True):
    userInput=input("You: ")    
    message.append({"role": "user", "content": userInput})
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,
        )
    response = completion.choices[0].message.content
    message.append({"role":"assistant", "content":response})
    print("chatbot: ",response)