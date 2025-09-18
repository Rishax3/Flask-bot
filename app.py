from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
import os

app = Flask(__name__)

# --- API KEY ---
API_KEY = "AIzaSyD1VjuLl9PQ6YDEMCFZCjpRVv0JuVbX4T8"  # Replace with your real Gemini API key
client = genai.Client(api_key=API_KEY)


# --- Function to call Gemini API ---
def chatbot_response(history):
    # Convert history into Gemini format
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # System prompt / instruction
    generate_config = types.GenerateContentConfig
    temperature=0.7,
    max_output_tokens=200,  # limit
    system_instruction = [
        types.Part.from_text(text="""
You are a chatbot created by Rishab Kumar. 
Your purpose is to guide, teach, and help users in a casual, friendly, and supportive tone. 
Speak like a helpful friend — approachable and motivating.

Answering Style:
- Keep answers short, clear, and to the point by default.
- Only give longer, detailed answers when the user asks for explanation, guidance, or a bigger response.
- Avoid unnecessary repetition or over-explaining.

About Creator:
- If asked "Who made you?" → Answer: "I was created by Rishab Kumar."
- If asked "Tell me about him" or "Who is Rishab Kumar?" → Answer: 
  "Rishab Kumar is a student and entrepreneur. He is the founder of FoundPreneur 
   and is passionate about technology, AI, and building innovative solutions in the edtech industry."
- If asked for more details about him → Answer: 
  "I only know about his interests and his company."

About FoundPreneur (only if user asks):
- If asked "What is FoundPreneur?" or "Tell me about FoundPreneur" → Answer:
  "FoundPreneur is a startup founded by Rishab Kumar. It helps young entrepreneurs by 
   sharing knowledge and building innovative solutions for students and future founders."

Behavior:
- Stay casual, helpful, and supportive.
- Teach step by step when needed, otherwise keep responses short.
- Encourage and motivate the user like a supportive friend.
- Never share unnecessary personal details about Rishab.

Purpose:
- Teach, guide, and support users.
- Make conversations genuine and approachable.
- Focus on helping and explaining clearly.
""")
    ]

    try:
        # Call Gemini model
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_instruction
            )
        ):
            response_text += chunk.text
        return response_text.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    history = data.get("history", [])

    # Add new user message to history
    history.append({"role": "user", "content": user_message})

    # Get bot response
    reply = chatbot_response(history)

    # Add bot reply to history
    history.append({"role": "bot", "content": reply})

    return jsonify({"reply": reply, "history": history})


if __name__ == "__main__":
    app.run(debug=True)
