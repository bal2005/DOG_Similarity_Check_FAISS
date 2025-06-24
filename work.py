import base64
import google.generativeai as genai

def get_image_description(base64_img: str) -> str:
    prompt = """You are an expert in dog breed recognition and behavior analysis. Given the image of a dog, analyze it and provide the following details in a consistent, structured bullet-point format (one line per attribute, no extra commentary):

Breed: (e.g., Labrador Retriever, German Shepherd, Mixed, Unknown)

Color & Markings: (e.g., Golden with white chest, Black and tan, Spotted)

Fur Type: (e.g., short, long, curly, wiry)

Size: (small / medium / large)

Ear Type: (e.g., floppy, erect, semi-erect)

Tail Type: (e.g., long, curled, bushy, short)

Return only the above list using the exact same order and labels. Use plain English, and write “Unknown” if a feature is not visible or determinable."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([
    prompt,
    genai.types.Blob(content=base64.b64decode(base64_img), mime_type="image/jpeg")
])
    return response.text

with open("D:\dog_images\IMG-20250619-WA0008.jpg", "rb") as f:
    base64_img = base64.b64encode(f.read()).decode("utf-8")
description = get_image_description(base64_img)
print(description)