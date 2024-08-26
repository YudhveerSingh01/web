from flask import Flask, render_template, request , jsonify

import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'your_secret_key'


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [
    "signs", "symptoms", "identify", "recognize", "diagnose", "appearance", "fungal diseases", "bacterial diseases", "viral diseases", "spots", "lesions", "yellowing leaves", "wilting", "growths", "cause", "reason", "origin", "sources", "causes of disease", "poor drainage", "contamination", "infected plants", "overwatering", "damp environment", "fungal growth", "root rot", "early symptoms", "initial signs", "early stages", "roots", "leaves", "stems", "discoloration", "leaf spots", "premature leaf drop", "reduced vigor", "root symptoms", "soft roots", "mushy textures", "foul smell", "brown roots", "black roots", "treat", "treatment", "cure", "remedy", "manage", "control", "solutions", "fungicides", "bactericides", "organic methods", "neem oil", "copper sprays", "beneficial microbes", "removing affected parts", "soil drainage", "aeration", "prevent", "avoid", "protection", "best practices", "tips", "precautions", "crop rotation", "disease-resistant varieties", "spacing", "sanitation", "soil health", "plant immunity", "disease outbreaks", "climate conditions", "spread", "environment", "humidity", "warmth", "fungal spread", "bacterial spread", "drought stress", "disease transmission", "isolation", "prompt management"
]




@app.route('/ask', methods=['POST'])
def ask():
    user_message = str(request.form['messageText'])
    
    if not is_medical_query(user_message):
        bot_response_text = "I'm sorry, I can only answer medical-related questions. Please ask a question related to medical topics."
    else:
        bot_response = chat.send_message(user_message)
        bot_response_text = bot_response.text
    
    return jsonify({'status': 'OK', 'answer': bot_response_text})

def is_medical_query(query):
    return any(keyword in query.lower() for keyword in medical_keywords)


