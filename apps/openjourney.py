import streamlit as st
from models import get_sd14, get_sd2base, get_sd15, get_openjourney
import torch

# Load the pre-trained model
pipe, description = get_openjourney()


it_per_s = 5.8


# Streamlit app
def main():
    st.title("OpenJourney")

    # Text input
    input_text = st.text_area(
        "Enter text. The longer, the better."
    )

    with st.expander("Advanced settings"):
        negative_prompt = st.text_area(
            "Negative prompt"
        )

        extra_kwargs = {}
        # is_nsfw = st.checkbox("NSFW?")
        is_manual_seed = st.checkbox("Manual seed?")

        if is_manual_seed:
            seed = st.number_input("Seed", value=1024)

        # if is_nsfw:
        #     extra_kwargs.update({
        #         "requires_safety_checker": False,
        #         "safety_checker": None,
        #     })

        guidance_scale = st.slider(
            "Guidance scale (7.5-8)", min_value=0.0, max_value=15.0, step=0.1, value=7.5
        )
        num_inference_steps = st.slider(
            "Inference steps no", min_value=1, max_value=100, value=50
        )
        is_step_by_step = st.checkbox("Generate image for each inference step")

        single_est_dt = num_inference_steps / it_per_s
        st.write(f"Single: Approx. {single_est_dt:.2f}s")
        step_by_step_est_dt = num_inference_steps*(num_inference_steps+1) / 2 / it_per_s
        st.write(f"Step by step: Approx. {step_by_step_est_dt:.2f}s")

        est_dt = step_by_step_est_dt if is_step_by_step else single_est_dt

    st.write(f"Approx. {est_dt:.2f}s")

    st.download_button("")

    # Generate image
    if st.button("Generate"):
        with st.spinner(text="请等一下"):
            start_range = 0 if is_step_by_step else num_inference_steps - 1
            for i in range(start_range, num_inference_steps):
                if is_manual_seed:
                    generator = torch.Generator("cuda").manual_seed(seed)
                    extra_kwargs.update({"generator": generator})
                if is_step_by_step:
                    st.write(i+1)

                img = pipe(
                    input_text,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=i+1,
                    **extra_kwargs,
                ).images[0]
                st.image(img, caption=input_text.capitalize(), use_column_width=True)
            st.write("Done.")


# Run the app
if __name__ == "__main__":
    main()
