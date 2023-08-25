from transformers import pipeline, DocumentQuestionAnsweringPipeline
import streamlit as st
from PIL import Image, UnidentifiedImageError
import io
from pdf2image import convert_from_bytes
import numpy as np


@st.cache_resource
def get_model() -> DocumentQuestionAnsweringPipeline:
    nlp: DocumentQuestionAnsweringPipeline = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa",
        device="cuda:0",
    )
    return nlp


def combine_images(imgs: list[Image.Image]):
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.vstack([i.resize(min_shape) for i in imgs])
    # save that beautiful picture
    return Image.fromarray( imgs_comb)


nlp = get_model()


uploaded_file = st.file_uploader("Choose an image file")
if uploaded_file is not None:
    raw_data = uploaded_file.getvalue()
    try:  # try to open as an image
        image = Image.open(io.BytesIO(raw_data))
    except UnidentifiedImageError:  # pdf probably
        images = convert_from_bytes(raw_data)[:10]
        image = combine_images(images)
    st.image(image)

question = st.text_input("Question")

if st.button("Generate"):
    with st.spinner(text="请等一下"):
        answer = nlp(image, question)
    st.header("Answer")
    st.write(answer)
