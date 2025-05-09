import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from PIL import Image
import custom_functions as fn
import plotly.express as px
import plotly.io as pio
pio.templates.default='streamlit'

### ChatBot Imports
import streamlit as st 
if st.__version__ <"1.31.0":
    streaming=False
else:
    streaming=True

import time,os
# from streamlit_chat

## LLM Classes 
from langchain_openai.chat_models import ChatOpenAI
# from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


# ## Memory Modules
# from langchain.chains.conversation.memory import (ConversationBufferMemory, 
#                                                   ConversationSummaryBufferMemory,
#                                                   ConversationBufferWindowMemory,
#                                                   ConversationSummaryMemory)
# Template for changing conversation chain's "flavor"
# from langchain.prompts.prompt import PromptTemplate


########### END OF IMPORTS

# Changing the Layout
st.set_page_config( #layout="wide", 
                   page_icon="⭐️Amazon Reviews NLP Dash")



# Get Fpaths
@st.cache_data
def get_app_fpaths(fpath='config/filepaths.json'):
	import json
	with open(fpath ) as f:
		return json.load(f)



##Load in the data
import json
with open("config/filepaths.json") as f:
    FPATHS = json.load(f)
    

    

@st.cache_data    
def load_df(fpath):
    import joblib
    return joblib.load(fpath)

@st.cache_data
def load_metadata(fpath):
    import pandas as pd
    return pd.read_json(fpath)

@st.cache_data
def get_rating_percent_by_year(df,**kwargs):
    return fn.get_rating_percent_by_year(df,**kwargs)

@st.cache_data
def get_average_rating_by_year(df, **kwargs):
    return fn.get_average_rating_by_year(df,**kwargs)

df = load_df(FPATHS['data']['processed-nlp']['processed-reviews-with-target_joblib'])
meta_df = load_metadata(FPATHS['data']['app']['product-metadata_json'])
product= meta_df.iloc[0]



## Title /header
# st.header("Exploratory Data Analysis of Amazon Reviews ")
# st.divider()
st.header("Amazon Customer Reviews Analysis")

st.image(FPATHS['images']['banner_png'],width=700,use_column_width='always')
st.divider()
## Product metasata
st.markdown("##### ***👈 Select the Display Options to enable/disable app components.***")
st.divider()
# st.subheader("Exploratory Analysis ")

# Setting sidebar controls ahead of time
### TEXT SELECTION OPTIONS FOR SIDEBAR
## Select which text column/preprocessing
st.sidebar.header("Display Options")
st.sidebar.markdown(">*Select visualizations for main pane.*")
show_product= st.sidebar.checkbox("Show Product Information", value=True)
show_review_graphs = st.sidebar.checkbox("Show rating distibutions.", value=True)
show_yearly =st.sidebar.checkbox('Show yearly trends in reviews', value=True)
show_wordclouds = st.sidebar.checkbox('Show Word Couds', value=True)
show_scattertext = st.sidebar.checkbox("Show ScatterText Visual", value=False)
st.sidebar.divider()

st.sidebar.header("Text Preprocessing Options")
st.sidebar.markdown(">*Select form of text for NLP EDA visuals.*")

text_col_map  ={"Original Text":'review-text-full',
                "Tokenized Text (no stopwords)":'tokens',
            'Lemmatzied Text':'lemmas'
            }
text_preprocessing_selection  =  st.sidebar.radio("Select Tokenization",options=list(text_col_map.keys()),# ['Original','Lemmas','Cleaned Tokens'],
                                            index=0)
text_col_selection = text_col_map[text_preprocessing_selection]
## Select # of words/ngrams
ngram_map = {'Unigrams/Tokens (1 Word)':1,
            'Bigrams (2 words)':2,
            'Trigrams (3 words)':3,
            'Quadgrams (4 words)':4}
ngram_selection = st.sidebar.radio("Select ngrams", options=list(ngram_map.keys()), #['Single Words','Bigrams','Trigrams','Quadgrams'],
                        index=1)
ngram_n = ngram_map[ngram_selection]
# Select custom stopwords
add_stopwords_str = st.sidebar.text_input("Enter list of words to exclude:",value='five,one,star,angel, hair,miracle,noodle,shirataki,pasta')
stopwords_list = fn.get_stopwords_from_string(add_stopwords_str)


st.sidebar.divider()


st.sidebar.subheader("Dev Options")
dev_show_fpaths = st.sidebar.checkbox('[Dev] Show FPATHS?',value=False)
dev_show_frame = st.sidebar.checkbox("[Dev] Show frame?",value=False)

if dev_show_fpaths:
    FPATHS
    
if dev_show_frame:
    st.dataframe(df.head())

st.subheader("Product Information")

if show_product==True:


    # st.markdown(f'Product Title: ***{product["Title (Raw)"]}***')
    # st.divider()
    col1,col2 = st.columns(2)

    # @st.cache_data
    def display_metadata(meta_df,iloc=0):
        # product = meta_df.iloc[iloc]
        # md = "#### Product Being Reviewed"
        md = ""
        md += f'\n- Product Title:\n***\"{product["Title (Raw)"]}\"***'
        # md += f"<p><img src='{product['Product Image']}' width=300px></p>"
        md += f'\n- Brand: {product["Brand"]}'
        md += f"\n- Price: {product['Price']}"
        md += f"\n- Ranked {product['Rank']} (2018)"

        md += f"\n- Categories:\n    - "
        md += "; ".join(product['Categories'])
        # md += f"\n- Categories:{', '.join(product['Categories'])}"
        
        
        return md

    col1.markdown(display_metadata(meta_df))
    col2.image(product['Product Image'],width=300)
else:
    col1,col2 =st.columns(2)
    col1.empty()
    col2.empty()

st.divider()

# st.image(FPATHS['images']['selected-product_jpg'])

## Distrubtion of reviews
# label: color
colors = {
    1: "red",
    2: "orange",
    3: "yellow",
    4:'limegreen',
    5:'green'}
muted_colors = fn.mute_colors_by_key(colors,keys_to_mute=None, saturation_adj=.7, lightness_adj=3)
df = df.sort_values('year', ascending=False)


st.markdown("#### Distribution of Star-Ratings for Selected Product")

if show_review_graphs==True:
    # show_histogram = st.checkbox("Show overall ratings distribution.")

    # if show_histogram:

        ## Plot histogram
    st.plotly_chart(px.histogram(df, 'overall', color='overall',width=600,
                                # title='# of Reviews per Star Rating',
                                color_discrete_map=muted_colors))

else:
    st.empty()

st.divider()

st.markdown("#### Change in Average Ratings By Year")
if show_yearly==True:

    ## Plot average scatter with trendline by year
    avg_by_year = get_average_rating_by_year(df)
    st.plotly_chart(px.scatter(avg_by_year, trendline='ols', width=800, height=400,
                            # title='Average Rating over Time'
                            ))


    st.divider()
    st.markdown("#### Trend in Proportion of Star Ratings over Time")
    # Plot counts by year
    counts_by_year=  get_rating_percent_by_year(df)
    stars_to_plot = st.multiselect('Ratings (Stars) to Include', options=list(counts_by_year.columns),
                                default=[1,5])
    # counts_by_year = counts_by_year.reset_index(drop=False)
    melted_counts_by_year = get_rating_percent_by_year(df, melted=True)
    melted_counts_by_year = melted_counts_by_year[melted_counts_by_year['Stars'].isin(stars_to_plot)]

    st.plotly_chart(px.scatter(melted_counts_by_year, x='year', y='%', color='Stars',
                            color_discrete_map=muted_colors,# title='Trend in Proportion of Star Ratings over Time',
                            trendline='ols'))
else:
    st.empty()

st.divider()
# st.plotly_chart(px.histogram(df, 'overall', color='overall',title='# of Reviews per Star Rating',animation_frame='year'))


## word clouds
st.header("NLP EDA")
st.divider()
st.subheader("Word Clouds")



# Get groups dict
@st.cache_data
def fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True, 
                                        group_col='target-rating', text_col = text_col_selection,
                                        stopwords=stopwords_list):
    kwargs = locals()
    group_texts = fn.get_groups_freqs_wordclouds(**kwargs) #testing stopwords
    return group_texts
## MENU FOR WORDCLOUDS
if show_wordclouds:
    st.markdown("👈 Change Text Preprocessing Options on the sidebar.")



    # wc_col1, wc_col2 = st.columns(2)
    # text_preprocessing_selection  =  wc_col1.radio("Select Text Processing",options=list(text_col_map.keys()),# ['Original','Lemmas','Cleaned Tokens'],
    #                                         index=0)
    # text_col_selection = text_col_map[text_preprocessing_selection]

    # ## Select # of words/ngrams
    # ngram_map = {'Single Words':1,
    #             'Bigrams':2,
    #             'Trigrams':3,
    #             'Quadgrams':4}
    # ngram_selection = wc_cofl2.radio("Select ngrams", options=list(ngram_map.keys()), #['Single Words','Bigrams','Trigrams','Quadgrams'],
    #                         index=0)
    # ngram_n = ngram_map[ngram_selection]

    # Select custom stopwords
    # add_stopwords_str = wc_col1.text_input("Enter list of words to exclude:",value='five,one,star')
    # stopwords_list = fn.get_stopwords_from_string(add_stopwords_str)

    
    group_texts = fn_get_groups_freqs_wordclouds(df,ngrams=ngram_n, as_freqs=True,group_col='target-rating', text_col = text_col_selection,
                                            stopwords=stopwords_list )
    # preview_group_freqs(group_texts)
    
    col1, col2 = st.columns(2)
    min_font_size = col1.number_input("Minumum Font Size",min_value=4, max_value=50,value=6, step=1)
    max_words = col2.number_input('Maximum # of Words', min_value=10, max_value=1000, value=200, step=5)
    
    fig  = fn.make_wordclouds_from_freqs(group_texts,stopwords=stopwords_list,min_font_size=min_font_size, max_words=max_words)
    
    st.pyplot(fig)
else:
    st.empty()
 
st.divider()


## Add creating ngrams
st.subheader('N-Grams')

# ngrams = st.radio('n-grams', [2,3,4],horizontal=True,index=1)
top_n = st.select_slider('Compare Top # Ngrams',[10,15,20,25],value=15)
## Compare n-grams
ngrams_df = fn.show_ngrams(df,top_n, ngram_n,text_col_selection,stopwords_list=stopwords_list)
fig = fn.plotly_group_ngrams_df(ngrams_df,show=False, title=f"Top {top_n} Most Common ngrams")
st.plotly_chart(fig)

st.divider()

####### CHATBOT


# # Create required session_state containers
# if 'messages' not in st.session_state:
#     st.session_state.messages=[]
    
# if 'API_KEY' not in st.session_state:
#     st.session_state['API_KEY'] = os.getenv('OPENAI_API_KEY') # Could have user paste in via sidebar

# if 'conversation' not in st.session_state:
#     st.session_state['conversation'] = None


# def reset():
#     if 'messages' in st.session_state:
#         st.session_state.messages=[]

#     if 'conversation' in st.session_state:
#         st.session_state['conversation'] = None


# # st.set_page_config(page_title="ChatGPT Clone", page_icon=':robot:')
# # st.header("Hey, I'm your Chat GPT")
# st.header("How can I assist you today?")



### ChatGPTM Options

## Special form of ngrams for chatgpt
chatgpt_stopwords = [*stopwords_list, 'angel','hair','miracle','noodle','shirataki','pasta']
    
def format_ngrams_for_chat(top_n_group_ngrams):
    
    string_table = []
    
    for group_name in top_n_group_ngrams.columns.get_level_values(0).unique():
        print(group_name)
        group_df = top_n_group_ngrams[group_name].copy()
        group_df['Rating Group'] = group_name 
        group_df = group_df.set_index("Rating Group")
        string_table.append(group_df)
        # string_table.append((group_df.values))
    return pd.concat(string_table)



## Define chatbot personalities/flavors
st.header("Ask ChatGPT for summary.")
flavor_options = {
    "Summary(General)": "You are a helpful assistant data analyst who uses ngrams from product reviews to summarize that customers do and do not like.",
    # "Summary(Bartender)": "You are a charming and emotionally intelligent bartender who gives great advice. You annotate your physical actions with new lines and asterisks as you answer. Act as helpful assistant who uses ngrams from product reviews to summarize that customers do and do not like.",
    "Customer (Low Carb/Gluten Free)": "You are a typical consumer who follows a low carb diet and has gluten sensitivity. You know what things you like in your food products.",
    "Customer (General)":  "You are a typical consumer who does not follow a special diet and enjoys eating gluten-containing foods. You know what things you like in your food products.",
}

@st.cache_resource
def load_chatgpt(temp,flavor_name):
    top_n_group_ngrams = fn.show_ngrams(df, top_n=25,ngrams=4, text_col_selection='review-text-full',
                                     stopwords_list=chatgpt_stopwords)
    md_table = format_ngrams_for_chat(top_n_group_ngrams)
    table_message = f"Heres a table of the most common ngrams from Low Rating reviews and high rating reviews. ```{md_table}```" 

    # Clear message history and specify flavor
    st.session_state.session_messages = [
    SystemMessage(content=flavor_options[flavor_name]),
    SystemMessage(content=table_message)
    ]
    return  ChatOpenAI(temperature=temp)
     
     
col1,col2=st.columns(2)
flavor_name = col1.selectbox("Which type of chatbot?", key='no_reset',options=list(flavor_options.keys()), index=0,)
# temp = col2.slider("Select model temperature:",min_value=0.0, max_value=2.0, value=0.1)
temp=0.7
reset_chat = st.sidebar.button("Clear history?")
if reset_chat:
    chat = load_chatgpt()
    # del st.session_state.session_messages


chat = load_chatgpt(temp,flavor_name)
if reset_chat:
    chat = load_chatgpt(temp,flavor_name)

def load_answer(query):#, model_name="gpt-3.5-turbo-instruct"):
    st.session_state.session_messages.append(HumanMessage(content=query))
    # Get answer and append to session state
    ai_answer = chat.invoke(st.session_state.session_messages)
    st.session_state.session_messages.append(AIMessage(content=ai_answer.content))
    return ai_answer.content
    # return show_history()
# Please give me a summary list of what customers liked  and did not like about the product."

def get_text():
    input_text = st.text_input("You: ", key='input', value ="Please give me a summary list of what customers liked  and did not like about this product.")
    return input_text

user_input=get_text()
submit = st.button('Generate')  

if submit:
    st.subheader(f"Answer - {flavor_name.title()}:")

    with st.container():
        response = load_answer(user_input)
    st.markdown(response)
else:
    st.empty()


############ SCATTERTEXT

st.divider()
## scattertext
@st.cache_data
def load_scattertext(fpath):
	with open(fpath) as f:
		explorer = f.read()
		return explorer

st.subheader("ScatterText:")
if show_scattertext:


    # checkbox_scatter = st.checkbox("Show Scattertext Explorer",value=True)
    # if checkbox_scatter:
    with st.spinner("Loading explorer..."):
        html_to_show = load_scattertext(FPATHS['eda']['scattertext-by-group_html'])
        components.html(html_to_show, width=1200, height=800, scrolling=True)
else:
    st.empty()