import streamlit as st
from transformers import pipeline, __version__

# Dictionary will store the summarizer model.
# Serves as a cache where the model can be stored and retrieved 
model_cache = {}


# The load_summarizer function checks if the summarizer model is already present in the model_cache dictionary. 
# If it exists, it is returned from the cache. 
# If not, a new summarizer model is created using the pipeline function from the transformers library. 
# The model is then stored in the cache for future use.
def load_summarizer():
    if "summarizer" not in model_cache:
        if __version__ == "4.4.2":
            model_cache["summarizer"] = pipeline("summarization", model="t5-base", device=0)
        else:
            model_cache["summarizer"] = pipeline("summarization", device=0)
    return model_cache["summarizer"]


# Function takes an input string and splits it into chunks based on a maximum chunk size of 500. 
#  It appends the <eos> token to sentence-ending punctuation marks for proper splitting. 
#  The function returns a list of chunks.
def generate_chunks(inp_str):
    max_chunk = 500
    inp_str = inp_str.replace('.', '.<eos>')
    inp_str = inp_str.replace('?', '?<eos>')
    inp_str = inp_str.replace('!', '!<eos>')

    sentences = inp_str.split('<eos>')
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    return chunks


# Removes the summarizer model from the model_cache dictionary. 
# Allows you to clear the cache manually if needed.
def clear_cache():
    if "summarizer" in model_cache:
        del model_cache["summarizer"]


# Customizing the sidebar color
st.markdown(
    """
    <style>
    body {
        background-color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Text Summarizer")
sentence = st.text_area('Paste your text:', height=30)
button = st.button("Summarize")
clear_cache_button = st.button("Clear Cache")

max_length = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
min_length = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
do_sample = st.sidebar.checkbox("Do sample", value=False)

# Two conditional blocks.
#  The first block handles the "Clear Cache" button.
#  The second block handles the "Summarize" button.
with st.spinner("Summarizing text.."):
    if clear_cache_button:
        clear_cache()
        st.write("Cache cleared.")
    if button and sentence:
        summarizer = load_summarizer()  
        chunks = generate_chunks(sentence)
        res = summarizer(chunks,
                         max_length=max_length,
                         min_length=min_length,
                         do_sample=do_sample)
        text = ' '.join([summ['summary_text'] for summ in res])
        st.write(text)
