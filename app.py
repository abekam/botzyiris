
!pip install sentence-transformers
!pip install openai

!pip install -U spacy
!python -m spacy download en_core_web_sm

"""STEP 1 Embedings:

STEP 2: Summaries
"""

import openai

# Set up OpenAI API credentials
openai.api_key = "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"

# Set up model ID
model_id = "text-davinci-003"

# Load the text from file
filename = "threedeedocs.txt"
with open(filename, 'rb') as file:
    raw_data = file.read()
    detected_encoding = chardet.detect(raw_data)['encoding']
print("Step 1: Detecting file encoding...")
print(f"Detected file encoding: {detected_encoding[:20]}")
print("")

with codecs.open(filename, 'r', detected_encoding) as file:
    text = file.read()

# Split the text into smaller chunks
chunk_size = 3000
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Generate a summary of each chunk
def generate_summary(chunk):
    prompt = f"Please provide a brief summary of the following Product Reference Guide for Zebra technologies's Smart industrial camera Iris GTX. Provide valuable info for installing, configuring, trouble shooting and maintenance:\n\n{chunk}\n\nSummary:"
    result = openai.Completion.create(
        engine=model_id,
        prompt=prompt,
        max_tokens=70,
        n=1,
        stop=None,
        temperature=0,
    )
    summary = result.choices[0].text.strip()
    return summary

# Generate summaries for all chunks and combine them
summaries = [generate_summary(chunk) for chunk in chunks]
combined_summary = " ".join(summaries)

# Print the first part of each chunk summary
print("First part of each chunk summary:")
for i, summary in enumerate(summaries):
    first_part = summary[:len(summary) // 2]
    print(f"Chunk {i+1} summary: {first_part}")
    summary_filename = "irisgtx_summary{}.npy".format(i + 1)
    np.save(summary_filename, summary)
    print(f"Summary saved to {summary_filename}.")

# Print the first part of the overall summary
print("\nFirst part of the overall summary:")
first_part_combined_summary = combined_summary[:len(combined_summary) // 2]
print(first_part_combined_summary)

# Save the embeddings to a file

from google.colab import drive
drive.mount('/content/drive')

"""Step 3: Cleaning summaries: Lemmatizing and removing stop words and punctuation"""

!pip install spacy
!python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

# Pre-process summaries
def preprocess_summaries(summaries):
    preprocessed_summaries = []
    for summary in summaries:
        doc = nlp(summary)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        named_entities = [entity.text for entity in doc.ents]
        tokens += named_entities
        preprocessed_summaries.append(tokens)
    return preprocessed_summaries

preprocessed_summaries = preprocess_summaries(summaries)

print("preprecessed summaries")
print(preprocessed_summaries)

"""Printing stuff"""

# Print original summaries
print("Original summaries:")
for i, summary in enumerate(summaries):
    print(f"Summary {i + 1}: {summary}")

# Print preprocessed summaries
print("\nPreprocessed summaries:")
for i, preprocessed_summary in enumerate(preprocessed_summaries):
    print(f"Preprocessed Summary {i + 1}: {preprocessed_summary}")

# Print original embeddings
print("\nOriginal embeddings:")
for i, embedding in enumerate(embeddings):
    print(f"Embedding {i + 1}: {embedding}")

"""Step 4: extracting relevant Keywords

KEYWORD NEW TRAIL
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load spacy model for preprocessing
nlp = spacy.load("en_core_web_sm")

processed_summaries = []
for summary in summaries:
    doc = nlp(summary)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
    processed_summaries.append(" ".join(tokens))

# Create document-term matrix
vectorizer = CountVectorizer()
doc_term_matrix = vectorizer.fit_transform(processed_summaries)

# Fit LDA model
num_topics = 10
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(doc_term_matrix)

# Assign topic probabilities to each summary
topic_probabilities = lda_model.transform(doc_term_matrix)

# Extract keywords using topic probabilities as weights
def extract_keywords_lda(keywords, topic_probabilities, vectorizer, num_keywords=5):
    text = " ".join(keywords)
    # Transform query to document-term matrix
    query_matrix = vectorizer.transform([text])
    # Assign topic probabilities to query
    query_topics = lda_model.transform(query_matrix)
    # Compute weighted keyword scores
    keyword_scores = {}
    for topic_idx, prob in enumerate(query_topics[0]):
        topic_keywords = lda_model.components_[topic_idx]
        top_keyword_indices = topic_keywords.argsort()[:-num_keywords-1:-1]
        for idx in top_keyword_indices:
            keyword = list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(idx)]
            score = prob * topic_probabilities[:, topic_idx].sum()
            if keyword not in keyword_scores:
                keyword_scores[keyword] = score
            else:
                keyword_scores[keyword] += score
    # Sort keywords by score
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:num_keywords]]

# Extract keywords using LDA
my_keywords = ['Vision system', 'fixed scanner', 'documentation', 'reference guide', 'installation']
fixed_keywords = ['Zebra', 'IRIS GTX', 'product',"install, troubleshoot, usb"]
filter_keywords = extract_keywords_lda([], topic_probabilities, vectorizer, num_keywords=200)

print("My keywords:", my_keywords)
print("Fixed keywords:", fixed_keywords)
print("Filter keywords:", filter_keywords)

"""BOTZY"""

import numpy as np

# Load embeddings and preprocessed summaries
embeddings = np.load("threedeedocs_embedding.npy")
preprocessed_summaries = preprocess_summaries(summaries)

# Print preprocessed_summaries
print("Preprocessed Summaries:")
for summary in preprocessed_summaries:
    print(summary)

# Print embeddings
print("Embeddings:")
print(embeddings)

"""GENERATE RESPONCE"""



import spacy
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
import openai

# Set up OpenAI API credentials
openai.api_key = "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"


# Load spacy model for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_summaries(summaries):
    preprocessed_summaries = []
    for summary in summaries:
        doc = nlp(summary)
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                if token.ent_type_:
                    tokens.append(token.ent_type_ + "_" + token.lemma_)
                else:
                    tokens.append(token.lemma_)
        preprocessed_summaries.append(tokens)
    return preprocessed_summaries



# Load embeddings and preprocessed summaries
embeddings = np.load("threedeedocs_embedding.npy")
preprocessed_summaries = preprocess_summaries(summaries)


def generate_response(query):
    # Encode query using the create_embeddings function
      # Encode query using the create_embeddings function
    query_embedding = create_embeddings(query)
    query_embedding = np.array(query_embedding).reshape(1, -1) # Reshape to 2D array



    # Find best matching summary
    summary_scores = cosine_similarity(query_embedding, embeddings)
    best_match_index = np.argmax(summary_scores)
    # Print shapes of embeddings and summaries
    #print("Embeddings shape:", embeddings.shape)
    #print("Summaries shape:", len(summaries))

    # Print example summary and corresponding embedding
    example_index = 0
   # print("Example Summary:", summaries[example_index])
    #print("Example Embedding:", embeddings[example_index])

    # Print summary scores
    #print("Summary Scores:", summary_scores)
    #print("Summary Scores shape:", summary_scores.shape)
    summary_scores = cosine_similarity(query_embedding, embeddings)
    best_match_index = np.argmax(np.squeeze(summary_scores))
    best_match_score = np.squeeze(summary_scores)[best_match_index]
    best_match_summary = summaries[best_match_index]
    
    # Extract relevant text from original document
    with open("threedeedocs.txt", 'r', encoding=detected_encoding) as file:
        text = file.read()
        start_index = text.find(best_match_summary)
        end_index = start_index + len(best_match_summary)
        relevant_text = text[start_index:end_index]
        print("relevant text")
        print(relevant_text)
    # Extract relevant phrases using keywords
    keywords = [token for token in nlp(query) if not token.is_stop and not token.is_punct]
    relevant_phrases = []
    for phrase in nlp(relevant_text).noun_chunks:
        phrase_tokens = [token for token in phrase if not token.is_stop and not token.is_punct]
        if any(keyword.text.lower() in [token.text.lower() for token in phrase_tokens] for keyword in keywords):
            relevant_phrases.append(phrase.text)
            print("relevant phrases")
            print(relevant_phrases)
    # Generate response
    if not relevant_phrases:
        if not keywords:
            response = "I am only trained to answer questions about Zebra's Iris GTX"
        else:
            prompt = f"You are a chatbot for the VS40 reference manual.\nUser: {query}\nDocument: {best_match_summary}\nKeywords: {', '.join([token.text for token in nlp(query) if not token.is_stop and not token.is_punct])}\nChatbot:"
            print("Generated Prompt:")
            print(prompt)

            openai_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0,
            )
            response = openai_response.choices[0].text.strip()
    else:
            response = f"Here is some information that might help:\n\n{relevant_phrases[0]}"

    return response







    
top_keywords = filter_keywords
print("tTTTTTTTTTTopkeywords")
print(top_keywords)
print("filter keywords")
print(filter_keywords)
# Initialize knowledge base
knowledge_base = {keyword: "Keyword related information" for keyword in filter_keywords}
context = " ".join([f"{keyword}: {info}" for keyword, info in knowledge_base.items()])


print("KKKKKKKKnowledge_base")
print(knowledge_base)
print("CCCContext")
print (context)
# Start chatbot
print("BOTZYY")
while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit command
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    # Generate response using the OpenAI API
    response = generate_response(user_input)

    # Check for keyword matches
    for keyword in top_keywords:
        if keyword.lower() in user_input.lower() and "Keyword related information" in response:
            response = knowledge_base[keyword]
            break

    # Print chatbot response
    print("Chatbot:", response)

"""GUI Trial

"""
