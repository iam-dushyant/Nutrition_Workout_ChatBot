# Code for developing Chatbot

from sentence_transformers import SentenceTransformer, util
import os

# Loading model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Loading content
content = []
folder = 'data/general/'
for file in os.listdir(folder):
    with open(os.path.join(folder, file)) as f:
        content = f.read()
        if '**Source:**' in content:
            main, source = content.split('**Source:**')
        else:
            main, source = content, 'Unknown'
        content.append({
            'text': main.strip(),
            'source': source.strip(),
            'embedding': model.encode(main.strip())
        })

# User question
print("Hello! I am your PT-bot!!")
print("It's great to have you take your fitness journey, and I am glad to assist you achieve it.")
question = input("Ask me anything about workouts, nutrition, or how to start your joruney: ")
q_embedding = model.encode(question)

# Compare
scores = [(util.cos_sim(q_embedding, item['embedding']).item(), item) for item in content]
best = max(scores, key=lambda x: x[0])

# Output
print(f"\n🤖 {best[1]['text']}\n\n🔗 {best[1]['source']}")