import streamlit as st
from models import get_rv
import torch
from config import Config
import json

st.set_page_config(layout="wide")

# Load the pre-trained model
pipe = get_rv()
pipe.safety_checker = None # lambda images, clip_input: (images, [])


it_per_s = 2.75
config = Config()


# Streamlit app
def main():
    c1, c2 = st.columns(2)

    with c1:
        st.title("Realistic Vision v1.4")
        # Text input
        config.prompt = st.text_area(
            "Enter text. The longer, the better.", value=config.prompt
        )

        config.negative_prompt = st.text_area(
            "Negative prompt", value=config.negative_prompt
        )

        config.manual_seed = st.number_input("Seed", value=0)
        config.seeds_no = st.number_input("How many seeds?", value=1)

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

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.download_button(
                label="Save config",
                file_name="rv-config.json",
                mime="application/json",
                data=json.dumps(config.as_dict()),
            )
        with cc3:
            do_generate = st.button("Generate")
        with cc2:
            auto_mode = st.checkbox("Auto")

    with c2:
        # Generate image
        if do_generate or auto_mode:
            with st.spinner(text="请等一下"):
                start_range = 0 if is_step_by_step else config.num_inference_steps - 1
                for i in range(start_range, config.num_inference_steps):
                    for seed in range(config.manual_seed, config.manual_seed + config.seeds_no):
                        st.write(f"Seed: {seed}, steps: {i+1}")
                        run_pipe(seed, i+1)
                st.write("Done.")


def run_pipe(seed, steps):
    extra_kwargs = {}
    if config.manual_seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
        extra_kwargs.update({"generator": generator})

    img = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        guidance_scale=config.guidance_scale,
        num_inference_steps=steps,
        # safety_checker = None,
        **extra_kwargs,
    ).images[0]
    st.image(img, caption=config.prompt.capitalize(), use_column_width=True)

# Run the app
if __name__ == "__main__":
    main()
