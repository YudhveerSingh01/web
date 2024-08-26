from flask import Flask, render_template, request, jsonify ,url_for
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
import pandas as pd

import os
from werkzeug.utils import secure_filename
from cv import predict_image , cropInfectDetect

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

#cv libraries
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png','jpg','jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit(".",1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


API_KEY = 'AIzaSyCnHiPnc81WluNjSklL6lLR5FO_NbHRCfM'
#'AIzaSyCCrYnLhDIgToWeG4u_nPpQcB9uNJMze0U'
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

medical_keywords = [
    "signs", "symptoms", "identify", "recognize", "diagnose", "appearance", "fungal diseases", "bacterial diseases", "viral diseases", "spots", "lesions", "yellowing leaves", "wilting", "growths", "cause", "reason", "origin", "sources", "causes of disease", "poor drainage", "contamination", "infected plants", "overwatering", "damp environment", "fungal growth", "root rot", "early symptoms", "initial signs", "early stages", "roots", "leaves", "stems", "discoloration", "leaf spots", "premature leaf drop", "reduced vigor", "root symptoms", "soft roots", "mushy textures", "foul smell", "brown roots", "black roots", "treat", "treatment", "cure", "remedy", "manage", "control", "solutions", "fungicides", "bactericides", "organic methods", "neem oil", "copper sprays", "beneficial microbes", "removing affected parts", "soil drainage", "aeration", "prevent", "avoid", "protection", "best practices", "tips", "precautions", "crop rotation", "disease-resistant varieties", "spacing", "sanitation", "soil health", "plant immunity", "disease outbreaks", "climate conditions", "spread", "environment", "humidity", "warmth", "fungal spread", "bacterial spread", "drought stress", "disease transmission", "isolation", "prompt management"
]




csv_path = 'datasetFile.csv'  
data = pd.read_csv(csv_path)

X = data[['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6']]
y = data['Parameter 9']
X = X.to_numpy()

model1 = LinearRegression()
model1.fit(X, y)


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/')
def home():
    return render_template('Homepage.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        inputs = [float(request.form[field]) for field in ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4', 'Parameter 5', 'Parameter 6']]
        prediction = model1.predict([inputs])
        print(prediction)
        output = "Argument 1" if prediction[0] >= 0.5 else "Argument 2"
        return render_template('predictor.html', prediction_text=f'{output}')
    return render_template('predictor.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')


@app.route('/coding')
def coding():
    return render_template('coding.html')

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


# Route to display the .py file
@app.route('/chatbotScript')
def chatbotScript():
    with open('static/chatbot.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Python Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content

@app.route('/cropScript')
def cropScript():
    with open('static/crop-yield.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Python Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>Highlighted Python Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content

@app.route('/cvScript')
def cvScript():
    with open('static/cv.py', 'r') as f:
        code = f.read()
    lexer = PythonLexer()
    formatter = HtmlFormatter(full=True, linenos=True, style='friendly')
    highlighted_code = highlight(code, lexer, formatter)
    html_content = f"""
    <html>
    <head>
        <title>Python Script</title>
        <style>{formatter.get_style_defs('.highlight')}</style>
    </head>
    <body>
        <h1>CV Script</h1>
        <div class="highlight">{highlighted_code}</div>
    </body>
    </html>
    """
    return html_content

@app.route('/cvProject/upload' , methods=['GET','POST'])
def cvPage():
    if request.method == 'POST':
        if 'file' not in request.file:
            print("file is not there")
            return redirect(request.url)
        file = request.fiels['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            model = cropInfectDetect()
            resultImage = predict_image(file_path , model)
            return redirect(url_for('cvResult',filename=os.path.basename(resultImage)))
    return render_template('cvPage.html')

@app.route('/cvProject/result/<filename>')
def cvResult():
    return render_template('cvResult.html', filename=filename)



if __name=="__main__":
    app.run(port=23)
