
import json
import os
import sys
import tempfile
import io

import psycopg2
from flask import Flask, jsonify, request, Blueprint, Response
from flask_cors import CORS
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
from salamandre.Segmentation import segmentation
from salamandre.image_moments import get_table

from datetime import datetime


db = SQLAlchemy()
size = 0

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True, static_folder="static", static_url_path="/static")
    cors = CORS(app, resources={r"/get_image": {"origins": "http://127.0.0.1:5000/"}})
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

        #with open (dataset, 'rb') as f:
        file = dataset.read()
        dataset.seek(0)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            dataset.save(temp.name)
            temp.seek(0)
            file_content = temp.read()
        file2 = temp.name

        latitude, longitude = get_gpsinfo(file2)
        date_exif = get_exif_data(file2)
        table = get_table(file2)



        data = Pictures(dataset.name, file, longitude, latitude, 0, date_exif,size,table, None)

        db.session.add(data)
        db.session.commit()

        answer = {
            'lat': latitude,
            'long': longitude,
            'focal' : 0,
            'filename': filename
        }
        return Response(json.dumps(answer), mimetype='application/json')

    @app.route("/getTable", methods=['GET'])
    def gettable():
        data = Pictures.query.order_by(Pictures.id.desc()).first()
        table = data.identification

        answer ={
            'table': table
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
    #Verification identification identique
    @app.route('/identification', methods=['POST'])
    def add_identification():

        dataset = request.get_json()

        tab = dataset.get('tableau')
        pourc = dataset.get('pourc')
        if pourc == None:
            pourc = 95

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
        data_send =  {'latitude': -1, 'longitude': -1 , 'date':0, 'pourcentage':0, 'index': -1}
        min_compt = 10000000
        index_min_compt = 1000
        index = -1
        for picture in pictures:
            index += 1
            identique = 1
            compt = 0
            if index < len(pictures) - 1:
                tabcurrent = picture.identification

                if (tabcurrent != None):
                    if (len(tab3) != len(tabcurrent) and tabcurrent != None):

                        continue
                    for i in range (len(tabcurrent)):
                        for j in range (len(tabcurrent[i])):
                            if (tabcurrent[i][j] != tab3[i][j]):
                                compt += 1
                                identique = 0


                    if (identique == 1):

                        break;
                        #if (identique ==0):
                         #   break
                    if (compt < min_compt):

                        min_compt = compt
                        index_min_compt = index

                    #Modif pour récuperer l'identification de avec la plus grande similitude.



        salamandre_ajoutee = Pictures.query.order_by(Pictures.id.desc()).first()



        data_sal = Salamandre.query.order_by(Salamandre.id.desc()).first()

        if (data_sal is not None):
            last_id = data_sal.salamandre_id

            nombre_tab = len(tab2)*len(tab2) - 111

            pourcentage = round((1-(min_compt/nombre_tab))*100,2)

            if (pourcentage >= pourc):
                index = 0
                for   picture in pictures:
                    index += 1
                    if (index == index_min_compt):

                        longitude = picture.longitude
                        latitude = picture.latitude
                        date = picture.date
                        id = picture.salamandre_id
                 
                        if (date != None):
                            date = date.isoformat()
                        data_send = {'latitude': latitude, 'longitude': longitude, 'date': date, 'pourcentage': pourcentage, 'index' : index}

                        break

     

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

    @app.route('/changeid',methods=['POST'])
    def changeid():
        dataset = request.get_json()
        index = dataset['indice']
        data_sal = Salamandre.query.order_by(Salamandre.id.desc()).first()
        salamandre_ajoutee = Pictures.query.order_by(Pictures.id.desc()).first()
        pictures = Pictures.query.all()
        last_id = data_sal.salamandre_id
        data_send = {'id':0}
        print("indice: ",index)
        if (index == -1):
            salamandre_ajoutee.salamandre_id = last_id + 1
            date1 = salamandre_ajoutee.date
            id = last_id+1
            data_send = {'id':id }
            lat1 = salamandre_ajoutee.latitude
            long1 = salamandre_ajoutee.longitude
            data = Salamandre(last_id + 1, date1, lat1, long1, 1)
            db.session.add(data)
            db.session.commit()
            element = Salamandre.query.order_by(Salamandre.id.desc()).first()
            element.last_lat = salamandre_ajoutee.latitude
            element.last_long = salamandre_ajoutee.longitude
            element.last_obs = salamandre_ajoutee.date
            db.session.commit()
        else:

            ind = 0
            for picture in pictures:
                ind += 1
                if (ind == index):
                    longitude = picture.longitude
                    latitude = picture.latitude
                    date = picture.date
                    id = picture.salamandre_id
                    print("id sal: ",id)
                    data_send = {'id': id}

                    if (id == None):
                        picture.salamandre_id = last_id + 1
                        salamandre_ajoutee.salamandre_id = last_id + 1
                        db.session.commit()
                        date1 = salamandre_ajoutee.date
                        lat1 = salamandre_ajoutee.latitude
                        long1 = salamandre_ajoutee.longitude
                        data = Salamandre(picture.salamandre_id, date1, lat1, long1, 2)

                        db.session.add(data)
                        db.session.commit()
                        element = Salamandre.query.order_by(Salamandre.id.desc()).first()
                        element.last_lat = salamandre_ajoutee.latitude
                        element.last_long = salamandre_ajoutee.longitude
                        element.last_obs = salamandre_ajoutee.date
                        db.session.commit()
                    else:

                        salamandre_ajoutee.salamandre_id = id
                        element = db.session.query(Salamandre).filter(Salamandre.salamandre_id == id).first()
                        element.last_lat = salamandre_ajoutee.latitude
                        element.last_long = salamandre_ajoutee.longitude
                        element.last_obs = salamandre_ajoutee.date
                        element.nbre_obs += 1
                        db.session.commit()



        return Response(json.dumps(data_send), mimetype='application/json')

    @app.route('/get_image', methods=['GET'])
    def get_image():
        dataset = request.args.get("index")
        #indice = dataset.get('index')
        indice = int(dataset)


        pictures = Pictures.query.all()
        for index, picture in enumerate(pictures):
            if (index == indice):

                file = picture.file
                break
                #with open(file,'rb') as f:

        image = Image.open(io.BytesIO(file))
        #image.show()
        #return Response(file, mimetype= 'image/jpeg')

        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='JPEG')
        img_byte_array.seek(0)

        # Retourner les données binaires de l'image en tant que réponse avec le type MIME 'image/jpeg'
        return send_file(img_byte_array, mimetype='image/jpeg')


    # image = Image.open(BytesIO(image_data))


    @app.route('/getdata', methods=['GET'])
    def getalldata():
        pictures = Pictures.query.all()
        data = [{'id':salamandre.id, 'filename': salamandre.filename, 'latitude': salamandre.latitude, 'longitude': salamandre.longitude, 'focal': salamandre.focal, 'date':salamandre.date, 'size': salamandre.size, 'salamandre_id': salamandre.salamandre_id} for salamandre in pictures]
        serialized_data = json.dumps(data, default=custom_json_serializer)
        return serialized_data




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