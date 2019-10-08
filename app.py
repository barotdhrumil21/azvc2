from flask import Flask, render_template,request,redirect,url_for, send_from_directory
from werkzeug import secure_filename
import numpy as np
import cv2
import os
import sys
from keras.models import model_from_json, Model
import pandas as pd
from deal_xl import process_xl

app = Flask(__name__)

loaded_model = 0
fname = ''
label_dict = {0: 'bottom_men_Activewear',
 1: 'bottom_men_Innerwear & Sleepwear',
 2: 'bottom_men_Jeans',
 3: 'bottom_men_Pants',
 4: 'bottom_men_Shorts',
 5: 'bottom_women_Activewear',
 6: 'bottom_women_Bottomwear',
 7: 'bottom_women_Ethnic wear',
 8: 'bottom_women_Lingerie and Sleepwear',
 9: 'men_Jackets',
 10: 'men_Outerwear',
 11: 'men_Shirts',
 12: 'men_Suits',
 13: 'men_Tshirts',
 14: 'top_men_Activewear',
 15: 'top_men_Ethnic wear',
 16: 'top_men_Innerwear & Sleepwear',
 17: 'top_women_Activewear',
 18: 'top_women_Dresses',
 19: 'top_women_Ethnic wear',
 20: 'top_women_Lingerie and Sleepwear',
 21: 'women_Dungarees',
 22: 'women_Jeans',
 23: 'women_Jumpsuits & Playsuits',
 24: 'women_Outerwear',
 25: 'women_Skirts',
 26: 'women_Tops'}

@app.route('/')
def index():
    global loaded_model
    print("rendered")
    json_file = open('./model/category-model-6165.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/category-model-6165.h5")
    print("Loaded model from disk")
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global fname, label_dict, loaded_model
    if request.method == 'POST':
        f = request.files['xl']
        fname = secure_filename(f.filename)
        f.save("./xl/" + fname)
        process_xl("./xl/", fname, loaded_model, label_dict)
        render_template("wait.html")
        return send_from_directory("./xl/", "solved-" +  fname, as_attachment = True)



if __name__ == '__main__':
    print("started")
    port = int(os.environ.get('PORT',8000))
    app.run(host='0.0.0.0', port=port, debug=True)
