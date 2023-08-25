from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import streamlit as st

@st.cache_resource
def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/stablecode-instruct-alpha-3b",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    return model

@st.cache_resource
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablecode-instruct-alpha-3b")
    return tokenizer


model = get_model()
tokenizer = get_tokenizer()

prompt = st.text_area(label="Prompt", value="###Instruction\nGenerate a python function to find number of CPU cores###Response\n")

max_new_tokens = st.slider("Max new tokens", min_value=0, max_value=1000, value=48)
temperature = st.slider("Temperature", min_value=0., max_value=2., value=0.2)

run = st.button("Run")
if run:
    pre = time.time()
    with st.spinner(text="请等一下"):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    dt = time.time() - pre
    st.code(output)
    st.write(f"It took {dt:.3f}s")
