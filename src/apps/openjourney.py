import streamlit as st
from models import get_sd14, get_sd2base, get_sd15, get_openjourney
import torch
from config import Config
import json

# Load the pre-trained model
pipe = get_openjourney()


it_per_s = 5.8
config = Config()


# Streamlit app
def main():
    st.title("OpenJourney")

    # Text input
    config.prompt = st.text_area(
        "Enter text. The longer, the better.", value=config.prompt
    )

    with st.expander("Advanced settings"):
        config.negative_prompt = st.text_area(
            "Negative prompt", value=config.negative_prompt
        )

        extra_kwargs = {}

        is_manual_seed = st.checkbox("Manual seed?")
        config.manual_seed = (
            st.number_input("Seed", value=1024) if is_manual_seed else None
        )

        config.guidance_scale = st.slider(
            "Guidance scale (7.5-8)",
            min_value=0.0,
            max_value=15.0,
            step=0.1,
            value=config.guidance_scale,
        )
        config.num_inference_steps = st.slider(
            "Inference steps no",
            min_value=1,
            max_value=100,
            value=config.num_inference_steps,
        )
        is_step_by_step = st.checkbox("Generate image for each inference step")

        single_est_dt = config.num_inference_steps / it_per_s
        st.write(f"Single: Approx. {single_est_dt:.2f}s")
        step_by_step_est_dt = (
            config.num_inference_steps * (config.num_inference_steps + 1) / 2 / it_per_s
        )
        st.write(f"Step by step: Approx. {step_by_step_est_dt:.2f}s")

        est_dt = step_by_step_est_dt if is_step_by_step else single_est_dt

    st.write(f"Approx. {est_dt:.2f}s")

    st.download_button(
        label="Get config",
        file_name="openjourney-config.json",
        mime="application/json",
        data=json.dumps(config.as_dict()),
    )

    # Generate image
    if st.button("Generate"):
        with st.spinner(text="请等一下"):
            start_range = 0 if is_step_by_step else config.num_inference_steps - 1
            for i in range(start_range, config.num_inference_steps):
                if config.manual_seed is not None:
                    generator = torch.Generator("cuda").manual_seed(config.manual_seed)
                    extra_kwargs.update({"generator": generator})
                if is_step_by_step:
                    st.write(i + 1)

                img = pipe(
                    prompt=config.prompt,
                    negative_prompt=config.negative_prompt,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=i + 1,
                    **extra_kwargs,
                ).images[0]
                st.image(img, caption=config.prompt.capitalize(), use_column_width=True)
            st.write("Done.")


# Run the app
if __name__ == "__main__":
    main()
