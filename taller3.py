import nltk
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download the Punkt tokenizer
nltk.download('punkt')

# Read the text from the file
with open('sample_text.txt', 'r') as file:
    text = file.read()

# Split the text into paragraphs
paragraphs = nltk.sent_tokenize(text)



# Load a pre-trained model and tokenizer (e.g., BERT)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to generate embeddings for a text
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Generate embeddings for each paragraph
paragraph_embeddings = [generate_embeddings(paragraph) for paragraph in paragraphs]


def get_most_related_chunks(question, paragraph_embeddings, top_N=5):
    question_embedding = generate_embeddings(question)
    
    # Convert the question_embedding to a NumPy array and transpose it
    question_embedding = question_embedding.cpu().detach().numpy()
    question_embedding = question_embedding.transpose()
    
    # Convert paragraph_embeddings to a NumPy array and reshape it
    paragraph_embeddings = np.array([emb.cpu().detach().numpy() for emb in paragraph_embeddings])
    paragraph_embeddings = paragraph_embeddings.reshape(paragraph_embeddings.shape[0], -1)
    
    # Calculate cosine similarity using NumPy's dot product and norms
    similarities = np.dot(paragraph_embeddings, question_embedding) / (
        np.linalg.norm(paragraph_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # Convert similarities to a list
    similarities_list = similarities.tolist()
    
    # Sort paragraphs by similarity and get the top N
    sorted_paragraphs = sorted(enumerate(similarities_list), key=lambda x: x[1], reverse=True)[:top_N]
    related_chunks = [paragraphs[i] for i, _ in sorted_paragraphs]
    return related_chunks

#Example usage
user_question = "What is the importance of AI?"
top_related_chunks = get_most_related_chunks(user_question, paragraph_embeddings)
for i, chunk in enumerate(top_related_chunks):
    print(f"Chunk {i + 1}: {chunk}")

