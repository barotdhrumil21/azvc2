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
label_dict = {0: 'bottom_men_Activewear', 1: 'bottom_men_Ethnic wear', 2: 'bottom_men_Innerwear & Sleepwear',
3: 'bottom_men_Jeans', 4: 'bottom_men_Pants', 5: 'bottom_men_Plus Size', 6: 'bottom_men_Shorts',
7: 'bottom_men_Swimwear', 8: 'bottom_women_Activewear', 9: 'bottom_women_Bottomwear', 10: 'bottom_women_Ethnic wear',
11: 'bottom_women_Lingerie and Sleepwear', 12: 'bottom_women_Swimwear', 13: 'men_Jackets', 14: 'men_Outerwear',
15: 'men_Shirts', 16: 'men_Suits', 17: 'men_Tshirts', 18: 'top_men_Activewear', 19: 'top_men_Ethnic wear', 20: 'top_men_Innerwear & Sleepwear',
21: 'top_men_Plus Size', 22: 'top_women_Activewear', 23: 'top_women_Dresses', 24: 'top_women_Ethnic wear', 25: 'top_women_Lingerie and Sleepwear',
26: 'top_women_Swimwear', 27: 'women_Dungarees', 28: 'women_Jeans', 29: 'women_Jeggings', 30: 'women_Jumpsuits & Playsuits',
31: 'women_Outerwear', 32: 'women_Skirts', 33: 'women_Tops'}

@app.route('/')
def index():
    global loaded_model
    print("rendered")
    json_file = open('./model/category-model-55.json', 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/category-model-55.h5")
    print("Loaded model from disk")
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    global fname, label_dict, loaded_model
    if request.method == 'POST':
        f = request.files['xl']
        fname = secure_filename(f.filename)
        if fname[-5:] != ".xlsx":
            return 0
        f.save("./xl/" + fname)
        process_xl("./xl/", fname, loaded_model, label_dict)
        render_template("wait.html")
        return send_from_directory("./xl/", "solved-" +  fname, as_attachment = True)



if __name__ == '__main__':
    print("started")
    port = int(os.environ.get('PORT',8000))
    app.run(host='0.0.0.0', processes = 5, threaded = True, port=port, debug=True)
