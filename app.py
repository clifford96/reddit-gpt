import streamlit as st
import pandas as pd
import datetime as dt
import toml

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from transformers import pipeline
import openai

import os

# Set TRANSFORMERS_CACHE environment variable
os.environ["TRANSFORMERS_CACHE"] = "/cache"

api_key = os.environ.get("OPENAI_API_KEY")
#secrets = toml.load("secrets.toml")
#openai.api_key = secrets["openai_api_key"]
#openai.api_key = "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"

# Load data from CSV files
posts_df = pd.read_csv('my_data.csv')
comments_df = pd.read_csv('my_comments.csv')

# Create a new column for post creation date
posts_df['created_date'] = posts_df['created_utc'].apply(lambda x: dt.datetime.fromtimestamp(x))
posts_df['created_year'] = posts_df['created_date'].dt.year

# Merge posts and comments dataframes on post_id
comments_posts_merged = posts_df.merge(comments_df, on='post_id', how='left')
# Remove any rows with missing comments
comments_posts_merged = comments_posts_merged[~comments_posts_merged['comment'].isnull()]

# Define a function to create a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(
        collocation_threshold=2,
        width=1000,
        height=500,
        background_color='white'
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Create Streamlit widgets
year_slider = st.slider(
    "Select a year",
    min_value=2019,
    max_value=2023,
    value=2023,
    step=1
)

subreddit_dropdown = st.selectbox(
    "Select a subreddit",
    options=comments_posts_merged['subreddit'].unique()
)

#Filter posts by selected year and plot word cloud of post titles
#st.subheader(f"Word cloud of post titles from {year_slider}")
#filtered_posts_year = comments_posts_merged[comments_posts_merged['created_year'] == year_slider]
#post_title_text_year = ' '.join(title for title in filtered_posts_year['post_title'].str.lower())
#generate_wordcloud(post_title_text_year)

# Filter posts by selected subreddit and plot word cloud of post titles
#st.subheader(f"Word cloud of post titles in r/{subreddit_dropdown}")
#filtered_posts_subreddit = comments_posts_merged[comments_posts_merged['subreddit'] == subreddit_dropdown]
#post_title_text_subreddit = ' '.join(title for title in filtered_posts_subreddit['post_title'].str.lower())
#generate_wordcloud(post_title_text_subreddit)

# Filter posts by selected subreddit and year and plot word cloud of post titles
st.subheader(f"Word cloud of post titles in r/{subreddit_dropdown} from {year_slider}")
filtered_posts_subreddit_year = comments_posts_merged[
    (comments_posts_merged['subreddit'] == subreddit_dropdown) &
    (comments_posts_merged['created_year'] == year_slider)
]
post_title_text_subreddit_year = ' '.join(title for title in filtered_posts_subreddit_year['post_title'].str.lower())
generate_wordcloud(post_title_text_subreddit_year)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600
    
    # defining LLM ChatGPT 3.5 turbo
    
    llm_predictor = LLMPredictor(llm = OpenAI(temperature = 0, model_name = "gpt-3.5-turbo", max_tokens = num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit = chunk_size_limit)
    
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex(documents,
                                llm_predictor = llm_predictor,
                                prompt_helper = prompt_helper)
    
    index.save_to_disk('index.json')
    
    return index

# Define the function to query the index
def ask_me_anything(question):
    index_path = 'index.json'
    index = GPTSimpleVectorIndex.load_from_disk(index_path)
    response = index.query(question, response_mode="compact")
    
    #st.write(f"You asked: **{question}**")
    st.write(f"***Bot Answered:*** {response.response}")

st.title("Ask Me Anything")

# Load the index
index_path = 'index.json'
if not os.path.exists(index_path):
    st.warning("Index not found. Please run the `construct_index` function first.")
else:
    # Create the input field
    input_question = st.text_input("Question:")

    # Create the button to trigger the `ask_me_anything` function
    if st.button("Submit"):
        ask_me_anything(input_question)