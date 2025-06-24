import streamlit as st
import base64
import numpy as np
import json
import faiss
import os
import google.generativeai as genai
from PIL import Image

# CONFIGURE GEMINI
genai.configure(api_key="AIzaSyB7gKTnDrD4kcjnGbCI72RQbgaioYYMUh0")  # Replace with your actual API key

# Constants
INDEX_PATH = "dog_vectors.index"
METADATA_PATH = "dog_metadata.json"
EMBEDDING_MODEL = "models/embedding-001"
DESCRIPTION_MODEL = "gemini-1.5-flash"

def get_image_description(base64_img):
    prompt = """
You are an expert in dog breed recognition and behavior analysis. Given the image of a dog, analyze it and provide the following details in a consistent, structured bullet-point format (one line per attribute, no extra commentary):

Breed: (e.g., Labrador Retriever, German Shepherd, Mixed, Unknown)
Color & Markings: (e.g., Golden with white chest, Black and tan, Spotted)
Fur Type: (e.g., short, long, curly, wiry)
Size: (small / medium / large)
Ear Type: (e.g., floppy, erect, semi-erect)
Tail Type: (e.g., long, curled, bushy, short)

Return only the above list using the exact same order and labels. Use plain English, and write “Unknown” if a feature is not visible or determinable.
"""

    model = genai.GenerativeModel(DESCRIPTION_MODEL)
    response = model.generate_content([
        prompt,
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64_img
            }
        }
    ])
    return response.text

def embed_description(desc):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=desc.strip(),
        task_type="retrieval_query"
    )
    return np.array(response["embedding"], dtype="float32")

def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        st.error("FAISS index or metadata file not found.")
        st.stop()

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return index, metadata

def normalize(v):
    return v / np.linalg.norm(v)

# Streamlit UI
st.set_page_config(page_title="🐶 Dog Similarity Finder", layout="centered")
st.title("🐶 Dog Similarity Search")
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    with st.spinner("Generating dog description..."):
        description = get_image_description(image_base64)
        st.subheader("🔎 Dog Description")
        st.text(description)

    with st.spinner("Embedding and searching..."):
        embedding = embed_description(description).reshape(1, -1)
        faiss.normalize_L2(embedding)

        index, metadata_list = load_index_and_metadata()

        k = 5
        distances, indices = index.search(embedding, k)

        st.subheader(f"📊 Top {k} Similar Dogs:")
        for rank, idx in enumerate(indices[0]):
            if idx >= len(metadata_list):
                continue
            meta = metadata_list[idx]
            similarity = (1 - distances[0][rank]) * 100

            st.markdown(f"### 🐕 Match #{rank + 1} — {similarity:.2f}% Similarity")

# Replace Colab prefix with local folder path
            image_path = meta.get("image_path", "").replace("/content", "D:/dog_images").replace("\\", "/")
            if image_path and os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=f"Dog ID: {meta['dog_id']}", width=300)
                except Exception as e:
                    st.warning(f"Could not open image: {image_path} — {e}")
            else:
                st.warning(f"Image not found: {image_path}")

            st.text(f"🆔 ID: {meta['dog_id']}")
            st.text(f"📋 Description:\n{meta['description']}")
            st.markdown("---")
