from flask import Flask, request
from flask_cors import CORS
# from nltk.corpus import stopwords
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree, ParentedTree
import os
import json
from six.moves import urllib
import zipfile
import sys
import time
from flask import send_file
import cv2, pickle
import numpy as np
import tensorflow as tf
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Download zip file from https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip and extract in stanford-parser-full-2015-04-20 folder in higher directory
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR,
                                             'stanford-parser-full-2018-10-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    

def is_parser_jar_file_present():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    return os.path.exists(stanford_parser_zip_file_path)


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    url = "stanford-parser-full-2018-10-17.zip"
    urllib.request.urlretrieve(url, stanford_parser_zip_file_path, reporthook)

def extract_parser_jar_file():
    stanford_parser_zip_file_path = os.environ.get('CLASSPATH') + ".jar"
    try:
        with zipfile.ZipFile(stanford_parser_zip_file_path) as z:
            z.extractall(path=BASE_DIR)
    except Exception:
        os.remove(stanford_parser_zip_file_path)
        download_parser_jar_file()
        extract_parser_jar_file()


def extract_models_jar_file():
    stanford_models_zip_file_path = os.path.join(os.environ.get('CLASSPATH'), 'stanford-parser-3.9.2-models.jar')
    stanford_models_dir = os.environ.get('CLASSPATH')
    with zipfile.ZipFile(stanford_models_zip_file_path) as z:
        z.extractall(path=stanford_models_dir)


def download_required_packages():
    if not os.path.exists(os.environ.get('CLASSPATH')):
        if is_parser_jar_file_present():
           pass
        else:
            download_parser_jar_file()
        extract_parser_jar_file()

    if not os.path.exists(os.environ.get('STANFORD_MODELS')):
        extract_models_jar_file()


def filter_stop_words(words):
    stopwords_set = set(['a', 'an', 'the', 'is'])
    # stopwords_set = set(stopwords.words("english"))
    words = list(filter(lambda x: x not in stopwords_set, words))
    return words


def lemmatize_tokens(token_list):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for token in token_list:
        lemmatized_words.append(lemmatizer.lemmatize(token))

    return lemmatized_words


def label_parse_subtrees(parent_tree):
    tree_traversal_flag = {}

    for sub_tree in parent_tree.subtrees():
        tree_traversal_flag[sub_tree.treeposition()] = 0
    return tree_traversal_flag


def handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    # if clause is Noun clause and not traversed then insert them in new tree first
    if tree_traversal_flag[sub_tree.treeposition()] == 0 and tree_traversal_flag[sub_tree.parent().treeposition()] == 0:
        tree_traversal_flag[sub_tree.treeposition()] = 1
        modified_parse_tree.insert(i, sub_tree)
        i = i + 1
    return i, modified_parse_tree


def handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree):
    # if clause is Verb clause or Proportion clause recursively check for Noun clause
    for child_sub_tree in sub_tree.subtrees():
        if child_sub_tree.label() == "NP" or child_sub_tree.label() == 'PRP':
            if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                tree_traversal_flag[child_sub_tree.treeposition()] = 1
                modified_parse_tree.insert(i, child_sub_tree)
                i = i + 1
    return i, modified_parse_tree


def modify_tree_structure(parent_tree):
    # Mark all subtrees position as 0
    tree_traversal_flag = label_parse_subtrees(parent_tree)
    # Initialize new parse tree
    modified_parse_tree = Tree('ROOT', [])
    i = 0
    for sub_tree in parent_tree.subtrees():
        if sub_tree.label() == "NP":
            i, modified_parse_tree = handle_noun_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)
        if sub_tree.label() == "VP" or sub_tree.label() == "PRP":
            i, modified_parse_tree = handle_verb_prop_clause(i, tree_traversal_flag, modified_parse_tree, sub_tree)

    # recursively check for omitted clauses to be inserted in tree
    for sub_tree in parent_tree.subtrees():
        for child_sub_tree in sub_tree.subtrees():
            if len(child_sub_tree.leaves()) == 1:  #check if subtree leads to some word
                if tree_traversal_flag[child_sub_tree.treeposition()] == 0 and tree_traversal_flag[child_sub_tree.parent().treeposition()] == 0:
                    tree_traversal_flag[child_sub_tree.treeposition()] = 1
                    modified_parse_tree.insert(i, child_sub_tree)
                    i = i + 1

    return modified_parse_tree


def convert_eng_to_isl(input_string):
    # get all required packages
    download_required_packages()

    if len(list(input_string.split(' '))) is 1:
        return list(input_string.split(' '))

    # Initializing stanford parser
    parser = StanfordParser()

    # Generates all possible parse trees sort by probability for the sentence
    possible_parse_tree_list = [tree for tree in parser.parse(input_string.split())]

    # Get most probable parse tree
    parse_tree = possible_parse_tree_list[0]
    print(parse_tree)
    # output = '(ROOT
    #               (S
    #                   (PP (IN As) (NP (DT an) (NN accountant)))
    #                   (NP (PRP I))
    #                   (VP (VBP want) (S (VP (TO to) (VP (VB make) (NP (DT a) (NN payment))))))
    #                )
    #             )'

    # Convert into tree data structure
    parent_tree = ParentedTree.convert(parse_tree)

    modified_parse_tree = modify_tree_structure(parent_tree)

    parsed_sent = modified_parse_tree.leaves()
    return parsed_sent


def pre_process(sentence):
    words = list(sentence.split())
    f = open('words.txt', 'r')
    eligible_words = f.read()
    f.close()
    final_string = ""

    for word in words:
        if word not in eligible_words:
            for letter in word:
                final_string += " " + letter
        else:
            final_string += " " + word

    return final_string

def build_squares(img):
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
			#print(imgCrop.shape)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
			x+=w+d
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) 
		imgCrop = None
		x = 420
		y+=h+d
	return crop

@app.route('/parser', methods=['GET', 'POST'])
def parseit():
    if request.method == "POST":
        input_string = request.form['text']
    else:
        input_string = request.args.get('speech')

    # print("input_string: " + input_string)
    input_string = input_string.capitalize()
    # input_string = input_string.lower()
    isl_parsed_token_list = convert_eng_to_isl(input_string)
    # print("isl_parsed_token_list: " + ' '.join(isl_parsed_token_list))

    # lemmatize tokens
    lemmatized_isl_token_list = lemmatize_tokens(isl_parsed_token_list)
    # print("lemmatized_isl_token_list: " + ' '.join(lemmatized_isl_token_list))

    # remove stop words
    filtered_isl_token_list = filter_stop_words(lemmatized_isl_token_list)
    # print("filtered_isl_token_list: " + ' '.join(filtered_isl_token_list))

    isl_text_string = ""

    for token in filtered_isl_token_list:
        isl_text_string += token
        isl_text_string += " "

    isl_text_string = isl_text_string.lower()

    data = {
        'isl_text_string': isl_text_string,
        'pre_process_string': pre_process(isl_text_string)
    }
    return json.dumps(data)

@app.route('/sethist')
def get_hand_hist():
    cam = cv2.VideoCapture(0)
    if cam.read()[0]==False: # cam.read returns a bool value and the image frame
	    cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
        key = cv2.waitKey(1)
        if key == ord('c'):		
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif key == ord('s'):
            flagPressedS = True	
            break
        if flagPressedC:	
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
			#cv2.imshow("Gaussian Blur",blur)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            cv2.imshow("Threshold", thresh)
        if not flagPressedS:
            imgCrop = build_squares(img)
		#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand histogram", img)
    cam.release()
    cv2.destroyAllWindows()
    #hist0 = cv2.fromarray(hist)
    #print hist.shape()
    #mat = np.matrix(hist)
    #with open('hist0.txt', 'w') as f:
        #for line in mat:
            #np.savetxt(f, line, fmt='%.2f')
    with open("hist", "wb") as f:
        pickle.dump(hist, f)
    return "background set successfully"

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f, encoding='bytes')
    f.close()
    return hist

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

def keras_process_image(img):
    image_x, image_y = get_image_size()
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT, (0, 0, 0))
    model = load_model('model.h5')
    model._make_predict_function()
    keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab*100 > 70:
        text = get_pred_text_from_db(pred_class)
    return text

def get_img_contour_thresh(img):
    x, y, w, h = 300, 100, 300, 300
    hist = get_hand_hist()
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh

def say_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()

def text_mode(cam):
    text = ""
    word = ""
    x, y, w, h = 300, 100, 300, 300
    count_same_frame = 0
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (720, 640))
        img, contours, thresh = get_img_contour_thresh(img)
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0
                if count_same_frame > 20:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text, )).start()
                    word = word + text
                    if word.startswith('I/Me '):
                        word = word.replace('I/Me ', 'I ')
                    elif word.endswith('I/Me '):
                        word = word.replace('I/Me ', 'me ')
                    count_same_frame = 0

            elif cv2.contourArea(contour) < 1000:
                if word != '':
					#print('yolo')
					#say_text(text)
                    Thread(target=say_text, args=(word, )).start()
                text = ""
                word = ""
        else:
            if word != '':
				#print('yolo1')
				#say_text(text)
                Thread(target=say_text, args=(word, )).start()
            text = ""
            word = ""
        blackboard = np.zeros((640, 720, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text to Speech", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255,255))
        cv2.putText(blackboard, "Predicted text- " + text, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0),2)
        cv2.putText(blackboard, word, (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("threshold", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('c'):
            cam.release()
            cv2.destroyAllWindows()
            break
    if keypress == ord('c'):
        return 2
    else:
        return 0

@app.route('/recognize-gesture')
def recognize():
    cam = cv2.VideoCapture(0)
    if cam.read()[0]==False:
        cam = cv2.VideoCapture(0)
    keypress = 1
    while True:
        if keypress == 1:
            keypress = text_mode(cam)
        else:            
            cam.release()
            cv2.destroyAllWindows()
            break
    return "window opened"

#@app.route('/return-files')
#def return_files_tut():
	#try:
		#return send_file('histo.txt', attachment_filename='histo.txt')
	#except Exception as e:
		#return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
