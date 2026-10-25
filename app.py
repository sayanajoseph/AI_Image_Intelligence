import streamlit as st
import torch
from PIL import Image
from io import BytesIO
from transformers import pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="AI Image Intelligence", page_icon="🖼️")

st.title("🖼️ AI Image Intelligence App")
st.write("Classify images or generate image captions using AI.")


device = 0 if torch.cuda.is_available() else -1


@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224",
        device=device
    )



@st.cache_resource
def load_captioner():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.eval()
    return processor, model


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

image = None

if uploaded_file:
    image = Image.open(
        BytesIO(uploaded_file.getvalue())
    ).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)


option = st.selectbox(
    "Choose a feature",
    ["Image Classification", "Image Captioning"]
)


if option == "Image Classification":
    if st.button("Classify Image"):
        if image is None:
            st.warning("Please upload an image first.")
        else:
            try:
                with st.spinner("Classifying image..."):
                    classifier = load_classifier()
                    results = classifier(image, top_k=5)

                st.subheader("Classification Results")

                for result in results:
                    st.write(
                        f"**{result['label']}** — "
                        f"{result['score']:.2%}"
                    )

            except Exception as e:
                st.error(f"Classification failed: {e}")


elif option == "Image Captioning":
    if st.button("Generate Caption"):
        if image is None:
            st.warning("Please upload an image first.")
        else:
            try:
                with st.spinner("Generating caption..."):
                    processor, model = load_captioner()

                    inputs = processor(
                        images=image,
                        return_tensors="pt"
                    )

                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=50)

                    caption = processor.decode(
                        output[0],
                        skip_special_tokens=True
                    )

                st.subheader("Generated Caption")
                st.write(caption)

            except Exception as e:
                st.error(f"Caption generation failed: {e}")

           
