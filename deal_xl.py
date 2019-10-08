import os
import pandas as pd
from PIL import Image
from multiprocessing.pool import ThreadPool
import random
import requests
from keras.preprocessing.image import ImageDataGenerator
import cv2
import wget
import numpy as np

def convert_428x428(rootDir):
    lostlist=[]
    for dirName, subdirList, fileList in os.walk(rootDir):
        print("inside : ", dirName, "\n")
        for fname in fileList:
            try:
                im = Image.open(dirName + "\\" +fname)
                if im.size != (428,428):
                    im = im.resize((428,428))
                    im.save(dirName + "\\" +fname)
                else:
                    pass
            except:
                lostlist.append(fname)
                #print(fname, len(lostlist))

    return lostlist

def re_download(rdir, lostlist, df):
    while len(lostlist) != 0:
        popped = []
        for file in os.listdir(rdir):
            for i in lostlist:
                if i in file:
                    try:
                        imglink = list(df[df['UID'].str.contains(i[:-4])].to_dict()["IMAGE LINK"].items())[0][1]
                        os.remove("./temp_images/" + i)
                        wget.download(imglink,"./temp_images/" + list(df[df['UID'].str.contains(i[:-4])].to_dict()["UID"].items())[0][1] + '.jpg')
                        lostlist.remove(i)
                    except IndexError:
                        break
    convert_428x428(rdir)

def url_response(urls):
    path, url = urls
    r = requests.get(url, stream = True)
    with open(path, 'wb') as f:
        for ch in r:
            f.write(ch)


def download_imgs(file_df, fname):
    urls = []
    try:
        os.makedirs("./temp_images/")
    except:
        pass
    uid_list = [ ''.join((random.choice('abcdefghijklmnopqrs1234567890') for i in range(50))) for i in range(len(file_df))]
    file_df["UID"] = uid_list
    print('downloading images for ' + fname)
    for i in range(len(file_df)):
        url = file_df['IMAGE LINK'].iloc[i]
        try:
            image_path = str( "./temp_images/" + '/' + ''.join(file_df['UID'].iloc[i]) +'.jpg')
        except:
            pass
        urls.append(tuple([image_path,url]))
    ThreadPool(15).imap_unordered(url_response, urls)
    return file_df

def finished(num, rdir):
    if len(os.listdir(rdir)) != num:
        return False
        len(os.listdir(rdir), ':', num, '\n')
    else:
        return True

def process_xl(xl_path, fname, model, label_dict):
    df = pd.read_excel(xl_path + fname)
    #df['CONFIDENCE'] = [-1 for i in range(len(df))]
    #df['CATEGORY'] = ['' for i in range(len(df))]
    df = download_imgs(df, fname)
    while not finished(len(df), "./temp_images/"):
        pass#print(len(os.listdir("./temp_images/")))
    print("finished")
    lost_imgs = convert_428x428("./temp_images/")
    re_download("./temp_images/", lost_imgs, df)
    print("starting model work")
    answer = []
    #confs = []
    for i in os.listdir("./temp_images/"):
        img = cv2.imread("./temp_images/" + i, 0)
        img = cv2.resize(img, (64,64))
        img = np.reshape(img, (1,64,64,1))
        x = img/255
        preds = model.predict(x)
        answer.append(label_dict[np.argmax(preds)])#.split('_')[-1])
        #confs.append(np.amax(preds))
        os.remove("./temp_images/"+i)
    df.insert(2, 'CATEGORY', answer, True)
    #df.insert(2, 'CONFIDENCE', confs, True)
    df.to_excel("./xl/solved-"+fname, index = False)
