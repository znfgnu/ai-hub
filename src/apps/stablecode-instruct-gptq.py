from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import streamlit as st
import time

model_name_or_path = "TheBloke/stablecode-instruct-alpha-3b-GPTQ"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@st.cache_resource
def get_model():
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)
    return model

model = get_model()

st.header(model_name_or_path)

temperature = st.slider("Temperature", min_value=0.0, max_value=5.0, value=.7)
max_new_tokens = st.slider("Max new tokens", min_value=0, max_value=2048, value=512)
prompt = st.text_area("Write some coding task, you can use markdown")
btn = st.button("Process")

prompt_template=f'''###Instruction:
{prompt}

###Response:
'''
if btn:
    dt = time.time()

    with st.chat_message("Stablecode", avatar="🤖"):
        with st.spinner("Thinking..."):
            input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
            output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
            response = tokenizer.decode(output[0])
        dt = time.time() - dt
        st.code(response)
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
