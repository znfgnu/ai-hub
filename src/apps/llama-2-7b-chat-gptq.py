from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import streamlit as st
import time


model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_basename = "model"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@st.cache_resource
def get_model():
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)
    return model

model = get_model()

st.header(model_name_or_path)

system_message = st.text_area(
    "System message",
    value="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
    height=250,
)
temperature = st.slider("Temperature", min_value=0.0, max_value=5.0, value=.7)
max_new_tokens = st.slider("Max new tokens", min_value=0, max_value=2048, value=512)
prompt = st.chat_input("Type your message")

prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]

'''

if prompt:
    with st.chat_message("User", avatar="🧐"):
        st.markdown(prompt)
    dt = time.time()
    with st.chat_message("Llama2", avatar="🤖"):
        with st.spinner("Thinking..."):
            input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
            response = tokenizer.decode(output[0])
        dt = time.time() - dt
        st.write(response)
    st.write(f"Done in {dt:.3f}s.")

# # Inference can also be done using transformers' pipeline

# # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
# logging.set_verbosity(logging.CRITICAL)

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.7,
#     top_p=0.95,
#     repetition_penalty=1.15
# )

# print(pipe(prompt_template)[0]['generated_text'])
