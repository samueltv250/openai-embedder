import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)
def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


class EmbeddingDataFrame:
    EMBEDDING_MODEL = "text-embedding-ada-002"
    openai.api_key = "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"  #platform.openai.com

    def __init__(self):
        self.df = self.generate_df()

    def get_embedding(self, text: str, model: str = EMBEDDING_MODEL):
        result = openai.Embedding.create(
            model=model,
            input=text
        )
        return result["data"][0]["embedding"]

    def generate_df(self):
        column_names = ['link', 'content', 'embedding']
        df = pd.DataFrame(columns=column_names)
        return df

    def add_chunks_to_df(self, link, text):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            self.add_to_df(link, chunk, embedding)

    def add_to_df(self, link, content, embedding):
        # self.df = self.df.append({'link': link, 'content': content, 'embedding': embedding}, ignore_index=True)
        new_row = pd.DataFrame({'link': [link], 'content': [content], 'embedding': [embedding]})
        self.df = pd.concat([self.df, new_row], ignore_index=True)
    def save_df(self, filename):
        self.df.to_pickle(filename)

    def load_df(self, filename):
        self.df = pd.read_pickle(filename)

    def get_sorted_df(self, text):
        embedding = self.get_embedding(text)
        self.df['similarity'] = self.df['embedding'].apply(lambda x: vector_similarity(x, embedding))
        return self.df.sort_values(by=['similarity'], ascending=False).reset_index(drop=True)

