from embedd import EmbeddingDataFrame
import numpy as np
import openai
import tiktoken
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
openai.api_key = "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"  #platform.openai.com

tokenizer = tiktoken.get_encoding('cl100k_base')
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
embeddedDF = EmbeddingDataFrame()

embeddedDF.load_df("embeddedDF.pickle")


MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, embeddedDF):
    """
    Fetch relevant 
    """
    df = embeddedDF.get_sorted_df(question)
    context = df.loc[0, 'content']
    print(df)
    url = embeddedDF.get_sorted_df(question).loc[0, 'link']
    context = text_splitter.split_text(context)[0]


    # Useful diagnostic information
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know. Provide the link to the content in a appropriate manner"\n\nUrl:"""+url+"""\n\nContext:\n"""
    
    return header + "".join(context) + "\n\n Q: " + question + "\n A:", url





def answer_query_with_context(query: str, embeddedDF):
    

    prompt, url = construct_prompt(query, embeddedDF)
    primer = "You are a Q&A bot that represents grupo provivienda. You are given a question and a context. Answert the question in the language you are asked in."
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": prompt}
        ],
        temperature=0, 
        max_tokens=300,
    )
    return (res['choices'][0]['message']['content']), url



# answer, url = answer_query_with_context("cual es la mejor offerta que tienen?", embeddedDF)
# answer, url = answer_query_with_context("estoy intentando poner una tele nueva y necesito poner clavos en la pared, como debo hacer esto?", embeddedDF)
# answer, url = answer_query_with_context("como estas?", embeddedDF)

# print(answer)
# print(url)
