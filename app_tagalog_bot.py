import streamlit as st
from streamlit_chat import message as st_message
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from translate import Translator


@st.cache_resource
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Tagalog Chatbot", page_icon="🤖")    
st.title("Hello Chatbot")

translator_en = Translator(from_lang='tl', to_lang='en')
translator_tl = Translator(from_lang='en', to_lang='tl')

def generate_answer(max_length=2048):
    tokenizer, model = get_models()
    user_message = st.session_state.input_text
    translation_en = translator_en.translate(user_message)
    inputs = tokenizer.encode(st.session_state.input_text, return_tensors="pt")
    result = model.generate(inputs, max_length=len(inputs[0]) + max_length, do_sample=False)
    message_bot = tokenizer.decode(
        result[0], skip_special_tokens=True
    )  # .replace("<s>", "").replace("</s>", "")
    translation_tl = translator_tl.translate(message_bot)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": translation_tl, "is_user": False})


st.text_input("Kausapin mo ako 😊", key="input_text", on_change=generate_answer)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) #unpacking
