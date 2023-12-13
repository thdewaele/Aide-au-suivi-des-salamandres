
import json
import os
import sys
import tempfile

import psycopg2
from flask import Flask, jsonify, request, Blueprint, Response

from flask import Flask, render_template, request, flash,send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import ARRAY
from geoalchemy2 import Geometry
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import salamandre
from salamandre.accueil import accueil
from salamandre.exif_data import get_exif_data
from salamandre.img import get_gpsinfo
from salamandre.index import index
from datetime import datetime


db = SQLAlchemy()
size = 0

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True, static_folder="static", static_url_path="/static")

    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:doisneau@localhost/Salamandre_webapp'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = 'secret string'

    db.init_app(app)



    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    @app.route('/')
    def init():
        return accueil()

    @app.route('/index.html')
    def index():
        return salamandre.index()

    @app.route('/donnee.html')
    def donnee():
        pictures = Pictures.query.all()
        return render_template('donnee.html', salamandres=pictures)

    bp = Blueprint('donnee', __name__, url_prefix='./donnee.html')


    @app.route('/addpict', methods=['POST'])
    def add_pictures():
        dataset = request.files['photo']
        filename = secure_filename(dataset.name)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            dataset.save(temp.name)
            file_content = temp.read()
        file = temp.name

        latitude, longitude = get_gpsinfo(file)
        date_exif = get_exif_data(file)



        data = Pictures(dataset.name, file_content, longitude, latitude, 0, date_exif,size,None, None)

        db.session.add(data)
        db.session.commit()

        answer = {
            'lat': latitude,
            'long': longitude,
            'focal' : 0,
            'filename': filename
        }
        return Response(json.dumps(answer), mimetype='application/json')


    @app.route ('/addtaille', methods = ['POST'])
    def add_taille():
        dataset = request.get_json()
        taille = dataset.get('size')
        filename = dataset.get('filename')
        derniere_ligne = Pictures.query.order_by(Pictures.id.desc()).first()
        derniere_ligne.size = taille
        derniere_ligne.filename=filename
        db.session.commit()
        size = float(taille)
        answer ={
            'size': taille
        }
        return Response(json.dumps(answer), mimetype='application/json')

    @app.route('/getlast', methods=['GET'])
    def get_data():
        data = Pictures.query.order_by(Pictures.id.desc()).first()
        longitude = data.longitude
        latitude = data.latitude

        data_send = {'latitude': latitude, 'longitude': longitude}
        return jsonify(data_send)

    @app.route('/identification', methods=['POST'])
    def add_identification():

        dataset = request.get_json()

        tab = dataset.get('tableau')
        pourc = dataset.get('pourc')
        print("Pourcentage : ", pourc)
        maxlength = 0
        length = len(tab)

        for i in range(length):
            long = len(tab[i])
            line = tab[i]
            if (long > maxlength):
                maxlength = long
            for j in range(long):
                if (line[j]== None):
                    line[j]=0
        tab2 = [[0]*maxlength for _ in range(length)]
        for i in range(length):
            long = len(tab[i])
            line = tab[i]
            for j in range(long):
                if (line[j] == "x" or line[j]=="X"):
                    tab2[i][j]= 1000
                elif (line[j] != 0):
                    if (line[j] == ""):
                        tab[i][j] = 0
                    else:
                        tab2[i][j] = line[j]


        ligne_dos = []
        sum_left =0;
        sum_mid= 0;
        sum_rigth= 0;
        for i in range(5):
            sum_left+=int(tab2[6+i][3])
            sum_mid+=int(tab2[6+i][4])
            sum_rigth+=int(tab2[6+i][5])

        ligne_dos.append(sum_left)
        ligne_dos.append(sum_mid)
        ligne_dos.append(sum_rigth)

        tab3 = [[0]*maxlength for _ in range(9)]
        for i in range(length):
            if (i+5>=length):
                break
            long = len(tab[i])
            line = tab2[i]
            if (i<6):
                for j in range(long):
                    if (line[j] != 0):
                        tab3[i][j] = line[j]
            elif (i==6):
                tab3[i][3]=ligne_dos[0]
                tab3[i][4] = ligne_dos[1]
                tab3[i][5] = ligne_dos[2]
            else:
                line = tab2[i+4]
                for j in range(long):
                    if (line[j] != 0):
                        tab3[i][j] = line[j]



        data = Pictures.query.order_by(Pictures.id.desc()).first()
        data.identification = tab3
        db.session.commit()

        pictures = Pictures.query.all()
        identique = 1
        data_send =  {'latitude': 0, 'longitude': 0 , 'date':0, 'pourcentage':0, 'index': 0}
        min_compt = 10000000
        index_min_compt = 1000
        for index, picture in enumerate(pictures):
            print(index)
            identique = 1
            compt = 0
            if index < len(pictures) - 1:
                tabcurrent = picture.identification

                if (tabcurrent != None):
                    if (len(tab3) != len(tabcurrent) and tabcurrent != None):
                        print("je suis là")
                        print("tab2: ",len(tab2))
                        print("tab: ",len(tabcurrent))
                        continue
                    for i in range (len(tabcurrent)):
                        for j in range (len(tabcurrent[i])):
                            if (tabcurrent[i][j] != tab3[i][j]):
                                compt += 1
                                identique = 0
                    print(compt)

                    if (identique == 1):
                        print("Identique")
                        break;
                        #if (identique ==0):
                         #   break
                    if (compt < min_compt):
                        print("hello")
                        min_compt = compt
                        index_min_compt = index

                    #Modif pour récuperer l'identification de avec la plus grande similitude.



        salamandre_ajoutee = Pictures.query.order_by(Pictures.id.desc()).first()



        data_sal = Salamandre.query.order_by(Salamandre.id.desc()).first()

        if (data_sal is not None):
            last_id = data_sal.salamandre_id
            print(min_compt)
            nombre_tab = len(tab2)*len(tab2)
            #print(1-(min_compt/nombre_tab))*100
            pourcentage = round((1-(min_compt/nombre_tab))*100,2)
            print(pourcentage)
            if (pourcentage >= pourc):
                for  index, picture in enumerate(pictures):
                    if (index == index_min_compt):

                        longitude = picture.longitude
                        latitude = picture.latitude
                        date = picture.date
                        id = picture.salamandre_id

                        if (date != None):
                            date = date.isoformat()
                        if (id == None):

                            picture.salamandre_id = last_id + 1
                            salamandre_ajoutee.salamandre_id = last_id + 1
                            db.session.commit()
                            date1 = salamandre_ajoutee.date
                            lat1 = salamandre_ajoutee.latitude
                            long1 = salamandre_ajoutee.longitude
                            data = Salamandre(picture.salamandre_id,date1,lat1, long1, 2)

                            db.session.add(data)
                            db.session.commit()
                            element = Salamandre.query.order_by(Salamandre.id.desc()).first()
                            element.last_lat = salamandre_ajoutee.latitude
                            element.last_long = salamandre_ajoutee.longitude
                            element.last_obs = salamandre_ajoutee.date
                            db.session.commit()
                        else:

                            salamandre_ajoutee.salamandre_id = id
                            element = db.session.query(Salamandre).filter(Salamandre.salamandre_id== id).first()
                            element.last_lat = salamandre_ajoutee.latitude
                            element.last_long = salamandre_ajoutee.longitude
                            element.last_obs = salamandre_ajoutee.date
                            element.nbre_obs +=1
                            db.session.commit()

                        print("Indice envoyé: ", index)
                        data_send = {'latitude': latitude, 'longitude': longitude, 'date': date, 'pourcentage': pourcentage, 'index' : index}

                        break
            else:
                salamandre_ajoutee.salamandre_id= last_id+1
                date1 = salamandre_ajoutee.date

                lat1 = salamandre_ajoutee.latitude
                long1 = salamandre_ajoutee.longitude
                data = Salamandre(last_id+1, date1,lat1, long1, 1)
                db.session.add(data)
                db.session.commit()
                element =Salamandre.query.order_by(Salamandre.id.desc()).first()
                element.last_lat = salamandre_ajoutee.latitude
                element.last_long = salamandre_ajoutee.longitude
                element.last_obs = salamandre_ajoutee.date
                db.session.commit()
                print("sal ajouté +1")
        else:

            salamandre_ajoutee.salamandre_id = 1

            date1 = salamandre_ajoutee.date
            lat1 = salamandre_ajoutee.latitude
            long1 = salamandre_ajoutee.longitude
            data = Salamandre(1, date1, lat1, long1, 1)

            db.session.add(data)
            db.session.commit()
            element = Salamandre.query.order_by(Salamandre.id.desc()).first()
            element.last_lat = salamandre_ajoutee.latitude
            element.last_long = salamandre_ajoutee.longitude
            element.last_obs = salamandre_ajoutee.date
            db.session.commit()



        return Response(json.dumps(data_send), mimetype='application/json')

    @app.route('/get_image', methods=['GET'])
    def get_image():
        dataset = request.args.get("index")
        #indice = dataset.get('index')
        indice = int(dataset)
        print("Indice reçu: ", indice)
        if (indice>0):
            pictures = Pictures.query.all()
            for index, picture in enumerate(pictures):
                if (index == indice):
                    print(type(picture.file))
                    file1 = picture.file.decode("ascii")
                    print("file: ",file1)
                    with open(picture.file, 'rb') as file:
                        image_data = file.read()
                    with open("photo.jpg", "wb") as file:
                        file.write(image_data)

                    #image = Image.open(BytesIO(image_data))

            return send_file(file, mimetype='image/jpeg')
        return




    """
    @app.route('/getdata', methods=['GET'])
    def getalldata():
        pictures = Pictures.query.all()
        data = [{'id':salamandre.id, 'filename': salamandre.filename, 'file': salamandre.file, 'latitude': salamandre.latitude, 'longitude': salamandre.longitude, 'focal': salamandre.focal, 'date':salamandre.date, 'size': salamandre.size} for salamandre in pictures]
        return jsonify(data, default = custom_json_serializer)

"""


    from . import img
    app.register_blueprint(img.bp)

    #db.create_all()
    return app


def custom_json_serializer(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')


class Salamandre(db.Model):
    __tablename__ = "salamandre"
    id = db.Column(db.Integer, primary_key = True, autoincrement=True)
    salamandre_id = db.Column(db.Integer) #, db.ForeignKey('Pictures.salamandre_id')
    last_obs = db.Column(db.Date)
    last_lat = db.Column(db.Float)
    last_long = db.Column(db.Float)
    nbre_obs = db.Column(db.Integer)

    def __init__(self, id, date, lat, long, nbr_obs):
        self.salamandre_id = id
        self.date = date
        self.lat = lat
        self.long = long
        self.nbre_obs = nbr_obs

class Pictures (db.Model):
    __tablename__ = 'pictures'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(50))
    file = db.Column(db.LargeBinary)
    longitude = db.Column(db.Float)
    latitude = db.Column(db.Float)
    geo = db.Column(Geometry(geometry_type="POINT"))
    focal = db.Column(db.Integer)
    date = db.Column(db.Date)
    size = db.Column(db.Float)
    identification = db.Column(ARRAY(db.Integer))
    salamandre_id = db.Column(db.Integer) #a verifier + creer colonne db

    def __init__(self, filename, file, longitude, latitude, focal,date, size, identification, salamandre_id):
        self.filename = filename
        self.file = file
        self.longitude =longitude
        self.latitude = latitude
        self.focal = focal
        self.date = date
        self.geo = 'POINT ({} {})'.format(longitude,latitude)
        self.size = size
        self.identification = identification
        self.salamandre_id =salamandre_id




application = create_app()