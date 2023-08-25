import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline
import streamlit as st


@st.cache_resource
def get_sd14():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe


@st.cache_resource
def get_sd2base():
    model_id = "stabilityai/stable-diffusion-2-base"
    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    return pipe


@st.cache_resource
def get_sd15():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


@st.cache_resource
def get_openjourney():
    model_id = "prompthero/openjourney-v4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

@st.cache_resource
def get_sdxl():
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    return pipe


@st.cache_resource
def get_rv():
    model_id = "SG161222/Realistic_Vision_V1.4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe
