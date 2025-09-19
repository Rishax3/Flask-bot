from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os

app = Flask(__name__)

# --- Configure Gemini API Key ---
# Store your API key in Render Environment Variables (not hardcoded!)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Create a model instance (use latest available Gemini model)
model = genai.GenerativeModel("gemini-1.5-flash")


# --- Function to call Gemini API ---
def chatbot_response(history):
    try:
        # Convert chat history into a prompt
        conversation = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Bot"
            conversation += f"{role}: {msg['content']}\n"

        # Add system instructions
        system_instruction = """
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
"""

        # Generate response
        response = model.generate_content(
            system_instruction + "\n" + conversation
        )

        return response.text.strip()

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
    # On Render, gunicorn will run this file, so no need for custom port
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
