from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
# print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Hello from Gemini!, how are you?")
print(response.text)
