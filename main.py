#!/usr/bin/env python
# coding: utf-8
# import firebase_admin
# from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
# from firebase_admin import credentials
# from firebase_admin import firestore
# import re
# import shutil
# from kivymd.uix.button import MDRaisedButton
# from kivy.properties import ObjectProperty ,ColorProperty
from kivy.clock import mainthread
from kivy.core.window import Window
import socket
from sys import exit
#from pathlib import Path
import time
import cv2
import Json_managt as jm
import requests
import multiprocessing, time
from multiprocessing import Process
multiprocessing.freeze_support()
import threading
from kivy.clock import Clock
import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request
import logging
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.video import Video
from kivy.config import Config
from kivy.core.window import Window

Window.hide()
stop_threads = False
os.environ["KIVY_VIDEO"] = "ffpyplayer"
logging.basicConfig(level=logging.DEBUG)
plus = 0.1

Config.set('graphics', 'resizable', '1')
Config.write()
Window.size = (1920, 1080)

## getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
ip_address = socket.gethostbyname(hostname)

url = f"http://{ip_address}:5000/predict"

app = Flask(__name__)

model = tf.keras.models.load_model('Best-retinal.h5')

logging.basicConfig(level=logging.DEBUG)


# load model

# In[2]


# prepare images

# In[3]:


def prepare_image(img):
    """
    prepares the image for the api call
    """
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


# prediction

# In[4]:


def predict_result(img):
    """predicts the result"""
    return np.argmax(model.predict(img)[0])


# initialize flask object

# In[5]:


app = Flask(__name__)


# setting up routes and their functions

# In[6]:


@app.route('/predict', methods=['POST'])
def infer_image():
    logging.info(str(request.files))

    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return str(predict_result(img))


@app.route('/', methods=['GET'])
def index():
    return 'Retinal OCT prediction API'


def start_app():
    global server
    print("Starting Flask app...")
    server = Process(target=app.run(debug=True, host='0.0.0.0', use_reloader=False))
    server.start()


help_str = '''

ScreenManager:
    Initiate_screen:
    WelcomeScreen:
    Single_screen:
    Multi_screen:

    
<Initiate_screen>:
    name : 'initiatescreen'
    spinner1 : spinner1
    spinner2 : spinner2
    spinner3 : spinner3
    spinner4 : spinner4
    spinner5 : spinner5
    spinner6 : spinner6
    spinner7 : spinner7
    
    label : label


    Videos:
        id:video
        source :"Datas/ImageKivyMD/TEST.jpg"
        state:"pause"
        options : {'eos':'loop'}
        allow_stretch : True
        keep_ratio : True
    
    MDSpinner:
        id : spinner1
        size_hint: None, None
        size: dp(507), dp(507)
        pos_hint: {'center_x': .5, 'center_y': .495}
        determinate: True
        determinate_time : 5
        active : False
        line_width : dp(4)
    
    MDSpinner:
        id : spinner2
        size_hint: None, None
        size: dp(637), dp(637)
        pos_hint: {'center_x': .50, 'center_y': .493}
        determinate: True
        determinate_time : 3
        active : False
        line_width : dp(4)
        
        
    MDSpinner:
        id : spinner3
        size_hint: None, None
        size: dp(854), dp(854)
        pos_hint: {'center_x': .500, 'center_y': .493}
        determinate: True
        determinate_time : 3
        active : False
        line_width : dp(4)
        
        
    MDLabel:
        id : label
        text : ""
        pos_hint: {'center_x': 0.5, 'center_y': .65}
        theme_text_color: "Custom"
        text_color: "white"
        halign: "center"
        multiline : True
        font_style:'Button'
        font_size : 18
        
        
        
    MDIconButton :
        id : icon1
        icon : "Datas/ImageKivyMD/algo.png"
        pos_hint: {'center_y':0.46,'center_x':0.5}
        icon_size : "300dp"
        opacity : 1
        on_press:
            app.start_animate()
    
    MDSpinner:
        id : spinner4
        size_hint: None, None
        size: dp(140), dp(140)
        pos_hint: {'center_x': 0.109, 'center_y': .243}
        active : False
        line_width : dp(3)
    
    MDSpinner:
        id : spinner5
        size_hint: None, None
        size: dp(140), dp(140)
        pos_hint: {'center_x': 0.888, 'center_y': .714}
        active : False
        line_width : dp(3)
        
    MDSpinner:
        id : spinner6
        size_hint: None, None
        size: dp(96), dp(96)
        pos_hint: {'center_x': 0.878, 'center_y': .256}
        active : False
            
    MDSpinner:
        id : spinner7
        size_hint: None, None
        size: dp(96), dp(96)
        pos_hint: {'center_x': 0.1195, 'center_y': .7022}
        active : False

<WelcomeScreen>:
    name : 'welcomescreen'
    titre_single_prediction : titre_single_prediction
    titre_multi_prediction : titre_multi_prediction
   
    Image:
        id:image_welcome_screen
        source :"Datas/ImageKivyMD/connexion.jpg"
        allow_stretch : True
        keep_ratio : False
        
    
    
    MDCard :
        id : card_title_opacity
        size_hint: None, None
        size: "650dp", "100dp"
        md_bg_color : 0,0,0,1
        opacity : 0.5
        pos_hint : {'center_y':0.85,'center_x':0.51}
    
    MDLabel:
        id : Titre_page_welcom
        text : "Retinal recognition of eyes disease" 
        pos_hint: {'center_x': 0.51, 'center_y': .85}
        theme_text_color: "Custom"
        text_color: "white"
        halign: "center"
        multiline : True
        font_style:'Button'
        font_size : 30
    
    
    MDCard :
        id : card_title_opacity_single
        size_hint: None, None
        size: "350dp", "80dp"
        md_bg_color : 0,0,0,1
        opacity : 0.5
        pos_hint: {'center_x': 0.63, 'center_y': .65}
    
    MDLabel:
        id : titre_single_prediction
        text : ''
        pos_hint: {'center_x': 0.63, 'center_y': .65}
        theme_text_color: "Custom"
        text_color: "white"
        halign: "center"
        multiline : True
        font_style:'Button'
        font_size : 20
    
    MDCard :
        id : card_title_opacity_multi
        size_hint: None, None
        size: "350dp", "80dp"
        md_bg_color : 0,0,0,1
        opacity : 0.5
        pos_hint: {'center_x': 0.39, 'center_y': .65}
    
    MDLabel:
        id : titre_multi_prediction
        text : ''
        pos_hint: {'center_x': 0.39, 'center_y': .65}
        theme_text_color: "Custom"
        text_color: "white"
        halign: "center"
        multiline : True
        font_style:'Button'
        font_size : 20
    
   
        
    MDIconButton :
        id : _bt_detect_multi_image
        icon : "Datas/ImageKivyMD/cercle_multi.png"
        pos_hint: {'center_y':0.50,'center_x':0.39}
        icon_size : "200dp"
        opacity : 1
        on_press : 
            root.manager.current = 'multiscreen'
            root.manager.transition.direction = 'left'
            
    MDIconButton :
        id : _bt_detect_single_image
        icon : "Datas/ImageKivyMD/cercle_single.png"
        pos_hint: {'center_y':0.50,'center_x':0.63}
        icon_size : "200dp"
        opacity : 1
        on_press : 
            root.manager.current = 'singlescreen'
            root.manager.transition.direction = 'left'
        
        
<Multi_screen>:
    name : "multiscreen"
    filechooser_multi_screen : filechooser_multi_screen
    image_multi_detect : image_multi_detect
    image_multi_detect2 : image_multi_detect2
    image_multi_detect3 : image_multi_detect3
    image_multi_detect4 : image_multi_detect4
    
    info_label_multi : info_label_multi
    info_label_multi2 : info_label_multi2
    info_label_multi3 : info_label_multi3
    info_label_multi4 : info_label_multi4
    
     
    Image:
        id:image_welcome_screen
        source :"Datas/ImageKivyMD/multi_single.jpg"
        allow_stretch : True
        keep_ratio : False
        
    MDIconButton :
        id : left
        icon : "Datas/ImageKivyMD/left.png"
        pos_hint: {'center_y':0.90,'center_x':0.05}
        icon_size : "100dp"
        opacity : 0.8
        on_press : 
            root.manager.current = 'welcomescreen'
            root.manager.transition.direction = 'right'
            app.clear_multi_screen()
            
    FileChooserIconView:
        canvas.before:
            Color:
                rgba:[62/255,60/255,55/255,0.45]
            RoundedRectangle:
                pos: 0,0
                size:600,225
        id:filechooser_multi_screen
        multiselect: True
        on_selection: app.display_image_multi(filechooser_multi_screen.selection)
        color : (0.8, 0.8, 0.8, 1.0)
        on_subentry_to_entry:False
        size_hint: None, None
        size:600,225
        pos_hint:{'x': 0.36,'center_y':0.17}
        rootpath : "./Datas/Image_test/"
        
    Image:
        id:image_multi_detect
        source : ""
        allow_stretch : False
        keep_ratio : True
        size_hint: None,None
        size : 600 , 300 
        pos_hint:{'center_y': 0.80,'center_x':0.30}
        
    MDCard:
        size_hint: None, None
        size: "500dp", "40dp"
        pos_hint:{'center_y': 0.97,'center_x':0.30}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.30]
        
    MDLabel:
        id : info_label_multi
        pos_hint:{'center_y': 0.97,'center_x':0.30}
        multiline : True
        halign : "center"
        text : ""
        color : [255/255,255/255,255/255,1]
        
    Image:
        id:image_multi_detect2
        source : ""
        allow_stretch : False
        keep_ratio : True
        size_hint: None,None
        size : 600 , 300 
        pos_hint:{'center_y': 0.80,'center_x':0.70}
        
    MDCard:
        size_hint: None, None
        size: "500dp", "40dp"
        pos_hint:{'center_y': 0.97,'center_x':0.70}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.30]
        
    MDLabel:
        id : info_label_multi2
        pos_hint:{'center_y': 0.97,'center_x':0.70}
        multiline : True
        halign : "center"
        text : ""
        color : [255/255,255/255,255/255,1]
        
    Image:
        id:image_multi_detect3
        source : ""
        allow_stretch : False
        keep_ratio : True
        size_hint: None,None
        size : 600 , 300 
        pos_hint:{'center_y': 0.45,'center_x':0.30}
        
    MDCard:
        size_hint: None, None
        size: "500dp", "40dp"
        pos_hint:{'center_y': 0.62,'center_x':0.30}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.30]
        
    MDLabel:
        id : info_label_multi3
        pos_hint:{'center_y': 0.62,'center_x':0.30}
        multiline : True
        halign : "center"
        text : ""
        color : [255/255,255/255,255/255,1]
        
    Image:
        id:image_multi_detect4
        source : ""
        allow_stretch : False
        keep_ratio : True
        size_hint: None,None
        size : 600 , 300 
        pos_hint:{'center_y': 0.45,'center_x':0.70}
        
    MDCard:
        size_hint: None, None
        size: "500dp", "40dp"
        pos_hint:{'center_y': 0.62,'center_x':0.70}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.30]
        
    MDLabel:
        id : info_label_multi4
        pos_hint:{'center_y': 0.62,'center_x':0.70}
        multiline : True
        halign : "center"
        text : ""
        color : [255/255,255/255,255/255,1]
        
    Button:
        id:_ti_cfg_test_ia_detect_multi
        text:'Multi prédictions sur images'
        size_hint : None,None
        size : 605,40
        pos_hint:{'center_y': 0.02,'center_x':0.518}
        background_color: [62/255,60/255,55/255,0.5]
        on_press: app.multi_predict()
    
    
<Single_screen>:
    name : "singlescreen" 
    image_single_detect : image_single_detect
    filechooser_single_screen: filechooser_single_screen
    text_field_crop : text_field_crop
    text_field_crop2 : text_field_crop2
    text_field_crop3 : text_field_crop3
    text_field_crop4 : text_field_crop4
    info_single_screen : info_single_screen
    
     
    Image:
        id:image_welcome_screen
        source :"Datas/ImageKivyMD/multi_single.jpg"
        allow_stretch : True
        keep_ratio : False
        
    MDIconButton :
        id : _bt_right
        icon : "Datas/ImageKivyMD/left.png"
        pos_hint: {'center_y':0.90,'center_x':0.05}
        icon_size : "100dp"
        opacity : 0.8
        on_press : 
            root.manager.current = 'welcomescreen'
            root.manager.transition.direction = 'right'
            app.clear_when_change_screen()
    
    FileChooserIconView:
        canvas.before:
            Color:
                rgba:[62/255,60/255,55/255,0.45]
            RoundedRectangle:
                pos: 0,0
                size:600,225
        id:filechooser_single_screen
        on_selection: app.display_image_single(filechooser_single_screen.selection)
        color : (0.8, 0.8, 0.8, 1.0)
        on_subentry_to_entry:False
        size_hint: None, None
        size:600,225
        pos_hint:{'x': 0.36,'center_y':0.25}
        rootpath : "./Datas/Image_test/"
        
    MDCard:
        size_hint: None, None
        size: "150dp", "80dp"
        pos_hint:{'center_y': 0.322,'center_x':0.30}
        opacity : 0.8
        md_bg_color :  [62/255,60/255,55/255,0.45]
        
        
    MDTextField:
        id: text_field_crop 
        hint_text: "y"
        mode: "rectangle"
        on_text_validate:app.crop_img()
        size_hint : None ,None
        pos_hint:{'center_y': 0.322,'center_x':0.30}
        size : 100,50
        line_color_focus : (255,255,255,1)
        text_color_focus : (109,7,26,1)
        hint_text_color_normal : (255,255,255,1)
        hint_text_color_focus : (255,255,255,1)
        
        
    MDCard:
        size_hint: None, None
        size: "150dp", "80dp"
        pos_hint:{'center_y': 0.3222,'center_x':0.73}
        opacity : 0.8
        md_bg_color :  [62/255,60/255,55/255,0.45]
        
    MDTextField:
        id: text_field_crop2
        hint_text: "y + height"
        mode: "rectangle"
        on_text_validate:app.crop_img()
        size_hint : None ,None
        pos_hint:{'center_y': 0.3222,'center_x':0.73}
        size : 100,50
        line_color_focus : (255,255,255,1)
        text_color_focus : (109,7,26,1)
        hint_text_color_normal : (255,255,255,1)
        hint_text_color_focus : (255,255,255,1)
    
    MDCard:
        size_hint: None, None
        size: "150dp", "80dp"
        pos_hint:{'center_y': 0.178,'center_x':0.30}
        opacity : 0.8
        md_bg_color : [62/255,60/255,55/255,0.45]
        
    MDTextField:
        id: text_field_crop3
        hint_text: "x"
        mode: "rectangle"
        on_text_validate:app.crop_img()
        size_hint : None ,None
        pos_hint:{'center_y': 0.178,'center_x':0.30}
        size : 100,50
        line_color_focus : (255,255,255,1)
        text_color_focus : (109,7,26,1)
        hint_text_color_normal : (255,255,255,1)
        hint_text_color_focus : (255,255,255,1)
        
    MDCard:
        size_hint: None, None
        size: "150dp", "80dp"
        pos_hint:{'center_y': 0.178,'center_x':0.73}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.45]
        
    MDTextField:
        id: text_field_crop4
        hint_text: "x + width"
        mode: "rectangle"
        on_text_validate:app.crop_img()
        size_hint : None ,None
        pos_hint:{'center_y': 0.178,'center_x':0.73}
        size : 100,150
        line_color_focus : (255,255,255,1)
        text_color_focus : (109,7,26,1)
        hint_text_color_normal : (255,255,255,1)
        hint_text_color_focus : (255,255,255,1)
        
    Image:
        id:image_single_detect
        source : ""
        allow_stretch : False
        keep_ratio : True
        size_hint: None,None
        size : 800 ,  500 
        pos_hint:{'center_y': 0.65,'center_x':0.52}
        
    Button:
        id:_ti_cfg_test_ia__detect
        text:'Prédiction sur image'
        size_hint : None,None
        size : 600,40
        pos_hint:{'center_y': 0.10,'center_x':0.515}
        background_color: [62/255,60/255,55/255,0.5]
        on_press: app.single_image_detect()
        
        
    MDCard:
        size_hint: None, None
        size: "1000dp", "50dp"
        pos_hint:{'center_y': 0.95,'center_x':0.52}
        opacity : 0.8
        md_bg_color :   [62/255,60/255,55/255,0.45]
        
    MDLabel:
        id : info_single_screen
        pos_hint:{'center_y': 0.95,'center_x':0.52}
        multiline : True
        halign : "center"
        text : ""
        color : [255/255,255/255,255/255,1]
        
            
    
'''


class WelcomeScreen(Screen):
    pass


class Initiate_screen(Screen):
    pass


class Single_screen(Screen):
    pass


class Multi_screen(Screen):
    pass


class Videos(Video):
    pass


sm = ScreenManager()
sm.add_widget(WelcomeScreen(name='welcomescreen'))
sm.add_widget(Initiate_screen(name='initiatescreen'))
sm.add_widget(Single_screen(name='singlescreen'))
sm.add_widget(Multi_screen(name='multiscreen'))


class LoginApp(MDApp):

    def build(self):
        self.strng = Builder.load_string(help_str)
        # self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Indigo"
        # self.theme_cls.accent_palette = "Red"
        return self.strng

    ###########################
    # Fonctions d'animation  #
    ###########################

    def annimate(self, param):
        self.strng.get_screen('initiatescreen').spinner1.active = True
        self.strng.get_screen('initiatescreen').spinner1.color = (255, 255, 255, 1)
        self.strng.get_screen('initiatescreen').label.text = f"Lecture des fichiers systéme en cours "

    def annimate2(self, param):
        self.strng.get_screen('initiatescreen').spinner2.active = True
        self.strng.get_screen('initiatescreen').spinner2.color = (255, 255, 255, 1)
        self.strng.get_screen('initiatescreen').label.text = f"Initialisation des variables \n d'environnment"

    def annimate3(self, param):
        self.strng.get_screen('initiatescreen').spinner3.active = True
        self.strng.get_screen('initiatescreen').spinner3.color = (255, 255, 255, 1)
        self.strng.get_screen('initiatescreen').label.text = f"Connexion au serveur Flask"

    def little_spiner_one(self, param):
        self.strng.get_screen('initiatescreen').spinner6.active = True
        self.strng.get_screen('initiatescreen').spinner6.color = (255, 255, 255, 1)

    def little_spiner_two(self, param):
        self.strng.get_screen('initiatescreen').spinner7.active = True
        self.strng.get_screen('initiatescreen').spinner7.color = (255, 255, 255, 1)

    def medium_spiner_one(self, param):
        self.strng.get_screen('initiatescreen').spinner4.active = True
        self.strng.get_screen('initiatescreen').spinner4.color = (255, 255, 255, 1)

    def medium_spiner_two(self, param):
        self.strng.get_screen('initiatescreen').spinner5.active = True
        self.strng.get_screen('initiatescreen').spinner5.color = (255, 255, 255, 1)

    def change_screen(self, param):
        self.strng.get_screen('initiatescreen').manager.current = 'welcomescreen'
        self.strng.transition.direction = 'left'

    def change_text_welcom(self, param):
        self.strng.get_screen('initiatescreen').label.font_size = 18
        self.strng.get_screen('initiatescreen').label.text = f"Bienvenue"
        self.strng.get_screen('welcomescreen').titre_single_prediction.text = "Prédictions sur une image \n et rognage"
        self.strng.get_screen('welcomescreen').titre_multi_prediction.text = "Prédictions sur plusieurs image"


    def start_animate(self):
        Clock.schedule_once(self.little_spiner_one, 0.1)
        Clock.schedule_once(self.little_spiner_two, 0.2)
        Clock.schedule_once(self.medium_spiner_one, 0.3)
        Clock.schedule_once(self.medium_spiner_two, 0.4)
        Clock.schedule_once(self.annimate, 0.5)
        Clock.schedule_once(self.annimate2, 5.5)
        Clock.schedule_once(self.annimate3, 9)
        Clock.schedule_once(self.change_text_welcom,13)
        Clock.schedule_once(self.change_screen, 14)
        Clock.schedule_once(self.change_screen, 15)

    def on_stop(self):
        p1.terminate()  # terminate Flask by pressing on cancel
        exit(1)

    ############################
    # Fonctions predict class  #
    ############################

    def multi_predict(self):


        payload = {}
        dicts = {}


        if res[0] != [] and len(res[0]) <= 4:

            for i in range(len(res[0])):
                files = [('file', (f'{res[0][i]}', open(f'{res[0][i]}', 'rb'), 'image/jpeg'))]
                response = requests.request("POST", url, data=payload, files=files)
                values = "Class  " + response.text
                keys = F"ID{i}"

                for i in range(1):
                    dicts[keys] = values
            print(dicts)

            if res[0] == []:
                self.strng.get_screen('multiscreen').info_label_multi.text = ""
                self.strng.get_screen('multiscreen').info_label_multi2.text = ""
                self.strng.get_screen('multiscreen').info_label_multi3.text = ""
                self.strng.get_screen('multiscreen').info_label_multi4.text = ""


            if len(res[0]) == 1 :
                self.strng.get_screen('multiscreen').info_label_multi.text = f"{dicts['ID0']}"
                self.strng.get_screen('multiscreen').info_label_multi2.text = ""

            if len(res[0]) == 2 :
                self.strng.get_screen('multiscreen').info_label_multi.text = f"{dicts['ID0']}"
                self.strng.get_screen('multiscreen').info_label_multi2.text = f"{dicts['ID1']}"
                self.strng.get_screen('multiscreen').info_label_multi3.text = ""

            if len(res[0]) == 3 :
                self.strng.get_screen('multiscreen').info_label_multi.text = f"{dicts['ID0']}"
                self.strng.get_screen('multiscreen').info_label_multi2.text = f"{dicts['ID1']}"
                self.strng.get_screen('multiscreen').info_label_multi3.text = f"{dicts['ID2']}"
                self.strng.get_screen('multiscreen').info_label_multi4.text = ""

            if len(res[0]) == 4 :
                self.strng.get_screen('multiscreen').info_label_multi.text = f"{dicts['ID0']}"
                self.strng.get_screen('multiscreen').info_label_multi2.text = f"{dicts['ID1']}"
                self.strng.get_screen('multiscreen').info_label_multi3.text = f"{dicts['ID2']}"
                self.strng.get_screen('multiscreen').info_label_multi4.text = f"{dicts['ID3']}"


        else:
            self.strng.get_screen('multiscreen').info_label_multi.text = "Aucune image sélectionner ou trop d'image sélectionner (max 4 ) "

    def single_image_detect(self):
        payload = {}
        dicts = {}
        path1 = donnees["path_single_image"]
        path2 = donnees["path_single_cropped"]

        if donnees["path_single_image"] != []:

            if donnees["active_crop"] == 'False':
                path = path1[-1]
            else:
                path = path2[-1]

            print(path)

            files = [('file', (f'{path}', open(f'{path}', 'rb'), 'image/jpeg'))]

            response = requests.request("POST", url, data=payload, files=files)
            keys = "Class  " + response.text
            values = path

            for i in range(1):
                dicts[keys] = values

            self.strng.get_screen(
                'singlescreen').info_single_screen.text = f"Cette image appartient à la classe : {response.text}"


        else:
            self.strng.get_screen('singlescreen').info_single_screen.text = f"Vous n'avez pas sélectionner d'images"

    ###########################
    # Fonctions partie single #
    ###########################

    # Affiche l'image sur l'écran lors de la sélection de l'image dans le filechooser
    def display_image_single(self, path_image):

        try:
            self.strng.get_screen('singlescreen').image_single_detect.source = path_image[0]
            donnees["active_crop"] = 'False'

            donnees["path_single_image"].append(path_image[0])

            # donnees["Image_normal_path"].append(path_image[0])
            # donnees["Image_crop_path"].append(path_image[0])

            donnees["height_image_single"].append(self.strng.get_screen('singlescreen').image_single_detect.width)
            donnees["width_image_single"].append(self.strng.get_screen('singlescreen').image_single_detect.height)

            self.strng.get_screen('singlescreen').text_field_crop.text = ''
            self.strng.get_screen('singlescreen').text_field_crop2.text = ''
            self.strng.get_screen('singlescreen').text_field_crop3.text = ''
            self.strng.get_screen('singlescreen').text_field_crop4.text = ''

            # donnees["width_image_crop"].clear()
            # donnees["height_image_crop"].clear()
            # donnees["x_image_crop"].clear()
            # donnees["y_image_crop"].clear()

        except:
            pass

    def clear_when_change_screen(self):
        self.strng.get_screen('singlescreen').image_single_detect.source = ""

        self.strng.get_screen('singlescreen').text_field_crop.text = ''
        self.strng.get_screen('singlescreen').text_field_crop2.text = ''
        self.strng.get_screen('singlescreen').text_field_crop3.text = ''
        self.strng.get_screen('singlescreen').text_field_crop4.text = ''

        donnees["path_single_image"].clear()

        self.strng.get_screen('singlescreen').info_single_screen.text = f""

    # Permets de cropper l'image
    def crop_img(self):
        global cropped

        if donnees["path_single_image"] != []:
            img = cv2.imread(donnees["path_single_image"][-1])

            donnees["height_image_single"].append(int(img.shape[0]))
            donnees["width_image_single"].append(int(img.shape[1]))
            donnees["y_image_crop"].append(0)
            donnees["x_image_crop"].append(0)

            try:
                donnees["y_image_crop"].append(int(self.strng.get_screen('singlescreen').text_field_crop.text))

            except:
                pass

            try:
                donnees["x_image_crop"].append(int(self.strng.get_screen('singlescreen').text_field_crop3.text))
            except:
                pass

            try:
                donnees["height_image_single"].append(int(self.strng.get_screen('singlescreen').text_field_crop2.text))
            except:
                pass

            try:
                donnees["width_image_single"].append(int(self.strng.get_screen('singlescreen').text_field_crop4.text))
            except:
                pass

            if donnees["height_image_single"][-1] == donnees["y_image_crop"][-1] or donnees["height_image_single"][
                -1] == 0 or donnees["width_image_single"][-1] == donnees["x_image_crop"][-1] or \
                    donnees["width_image_single"][-1] == 0 or donnees["y_image_crop"][-1] > \
                    donnees["height_image_single"][-1] or donnees["height_image_single"][-1] < donnees["y_image_crop"][
                -1] or donnees["x_image_crop"][-1] > donnees["width_image_single"][-1] or donnees["width_image_single"][
                -1] < donnees["x_image_crop"][-1]:

                if donnees["x_image_crop"][-1] > donnees["width_image_single"][-1]:
                    self.strng.get_screen(
                        'singlescreen').info_single_screen.text = f"Valeur incompatible | Diminuer la valeur de x de {donnees['x_image_crop'][-1] - donnees['width_image_single'][-1]} "

                else:
                    self.strng.get_screen(
                        'singlescreen').info_single_screen.text = f"Valeur incompatible | Diminuer la valeur de y de {donnees['y_image_crop'][-1] - donnees['height_image_single'][-1]} "


            elif not donnees["y_image_crop"][-1] > donnees["height_image_single"][-1] or donnees["x_image_crop"][-1] > \
                    donnees["width_image_single"][-1]:

                print(donnees["y_image_crop"][-1], donnees["height_image_single"][-1], donnees["x_image_crop"][-1],
                      donnees["width_image_single"][-1])

                donnees["active_crop"] = 'True'
                cropped = img[donnees["y_image_crop"][-1]:donnees["height_image_single"][-1],
                          donnees["x_image_crop"][-1]:donnees["width_image_single"][-1]]

                cv2.imwrite('./Datas/Image_test/cropped.jpeg', cropped)

                donnees["path_single_cropped"].append("./Datas/Image_test/cropped.jpeg")

                self.strng.get_screen('singlescreen').image_single_detect.source = './Datas/Image_test/cropped.jpeg'
                self.strng.get_screen('singlescreen').image_single_detect.reload()

                donnees["height_image_single"].clear()
                donnees["width_image_single"].clear()
                donnees["y_image_crop"].clear()
                donnees["x_image_crop"].clear()
                self.strng.get_screen('singlescreen').info_single_screen.text = ""

            else:
                self.strng.get_screen(
                    'singlescreen').info_single_screen.text = f"Valeur incompatible | Diminuer la valeur de y de {donnees['y_image_crop'][-1] - donnees['height_image_single'][-1]} et la valeur de x de {donnees['x_image_crop'][-1] - donnees['width_image_single'][-1]} "
        else:
            self.strng.get_screen('singlescreen').info_single_screen.text = "Aucune image sélectionner"

    ###########################
    # Fonctions partie multi  #
    ###########################

    def display_image_multi(self, path_image):
        global res

        donnees["path_multi_image"].append(path_image)

        res = []
        [res.append(x) for x in donnees["path_multi_image"] if x not in res]

        if res[0] == []:
            self.strng.get_screen('multiscreen').image_multi_detect.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect2.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect3.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect4.source = ""

            self.strng.get_screen('multiscreen').info_label_multi.text = ""
            self.strng.get_screen('multiscreen').info_label_multi2.text = ""
            self.strng.get_screen('multiscreen').info_label_multi3.text = ""
            self.strng.get_screen('multiscreen').info_label_multi4.text = ""

            self.strng.get_screen('multiscreen').image_multi_detect.reload()
            self.strng.get_screen('multiscreen').image_multi_detect2.reload()
            self.strng.get_screen('multiscreen').image_multi_detect3.reload()
            self.strng.get_screen('multiscreen').image_multi_detect4.reload()

        if len(res[0]) == 1:
            self.strng.get_screen('multiscreen').image_multi_detect2.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect2.reload()
            self.strng.get_screen('multiscreen').image_multi_detect.source = res[0][0]
            self.strng.get_screen('multiscreen').image_multi_detect2.source = ""
            print(res[0])

        if len(res[0]) == 2:
            self.strng.get_screen('multiscreen').image_multi_detect3.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect3.reload()
            self.strng.get_screen('multiscreen').image_multi_detect.source = res[0][0]
            self.strng.get_screen('multiscreen').image_multi_detect2.source = res[0][1]
            self.strng.get_screen('multiscreen').info_label_multi3.text = ""
            print(res[0])

        if len(res[0]) == 3:
            self.strng.get_screen('multiscreen').image_multi_detect4.source = ""
            self.strng.get_screen('multiscreen').image_multi_detect4.reload()
            self.strng.get_screen('multiscreen').image_multi_detect.source = res[0][0]
            self.strng.get_screen('multiscreen').image_multi_detect2.source = res[0][1]
            self.strng.get_screen('multiscreen').image_multi_detect3.source = res[0][2]
            self.strng.get_screen('multiscreen').info_label_multi4.text = ""
            print(res[0])

        if len(res[0]) == 4:
            self.strng.get_screen('multiscreen').image_multi_detect.source = res[0][0]
            self.strng.get_screen('multiscreen').image_multi_detect2.source = res[0][1]
            self.strng.get_screen('multiscreen').image_multi_detect3.source = res[0][2]
            self.strng.get_screen('multiscreen').image_multi_detect4.source = res[0][3]
            print(res[0])

        if len(res[0]) > 4 :
            del res[0][-1]
            print(res[0])


    def clear_multi_screen(self):

        self.strng.get_screen('multiscreen').image_multi_detect.source = ""
        self.strng.get_screen('multiscreen').image_multi_detect2.source = ""
        self.strng.get_screen('multiscreen').image_multi_detect3.source = ""
        self.strng.get_screen('multiscreen').image_multi_detect4.source = ""

        self.strng.get_screen('multiscreen').image_multi_detect.reload()
        self.strng.get_screen('multiscreen').image_multi_detect2.reload()
        self.strng.get_screen('multiscreen').image_multi_detect3.reload()
        self.strng.get_screen('multiscreen').image_multi_detect4.reload()

        self.strng.get_screen('multiscreen').info_label_multi.text = ""
        self.strng.get_screen('multiscreen').info_label_multi2.text = ""
        self.strng.get_screen('multiscreen').info_label_multi3.text = ""
        self.strng.get_screen('multiscreen').info_label_multi4.text = ""

        donnees["path_multi_image"].clear

        '''self.strng.get_screen('multiscreen').image_multi_detect.source =

        self.strng.get_screen('multiscreen').image_multi_detect2.source =

        self.strng.get_screen('multiscreen').image_multi_detect3.source =

        self.strng.get_screen('multiscreen').image_multi_detect4.source ='''

    ''' image_multi_detect: image_multi_detect
        image_multi_detect2: image_multi_detect2
        image_multi_detect3: image_multi_detect3
        image_multi_detect4: image_multi_detect4'''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if os.environ.get("WERKZEUG_RUN_MAIN") != 'true':

       p1 = Process(target=start_app) # assign Flask to a process
       p1.start()

    print('Lecture base de donnees original')
    donnees = jm.read_json('data_safe.json')
    print('Sauvegarde base de donnees de travail')
    jm.write_json('data.json', donnees)
    time.sleep(2)
    Window.show()
    LoginApp().run()
