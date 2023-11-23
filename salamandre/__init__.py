import json
import os
import sys
import tempfile

import psycopg2
from flask import Flask, jsonify, request, Blueprint, Response

from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import ARRAY
from geoalchemy2 import Geometry
from werkzeug.utils import secure_filename

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


        latitude, longitude = get_gpsinfo(temp.name)
        date_exif = get_exif_data(temp.name)
        file = temp.name

        data = Pictures(dataset.name, file.encode('ascii'), longitude, latitude, 0, date_exif,size,None)

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
        maxlength = 0
        length = len(tab)
        print(tab)
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

        print(tab2)
        ligne_dos = []
        sum_left =0;
        sum_mid= 0;
        sum_rigth= 0;
        for i in range(5):
            sum_left+=tab[6+i][3]
            sum_mid+=tab[6+i][4]
            sum_rigth+=tab[6+i][5]

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
        data_send =  {'latitude': 0, 'longitude': 0 , 'date':0}
        for index, picture in enumerate(pictures):
            identique = 1
            if index < len(pictures) - 1:
                tabcurrent = picture.identification

                if (tabcurrent != None):
                    if (len(tab2) != len(tabcurrent) and tabcurrent != None):
                        continue
                    for i in range (len(tabcurrent)):
                        for j in range (len(tabcurrent[i])):
                            if (tabcurrent[i][j] != tab2[i][j]):
                                identique = 0
                                break

                        if (identique ==0):
                            break
                    if (identique == 1):
                        longitude = picture.longitude
                        latitude = picture.latitude
                        date = picture.date
                        if (date != None):
                            date= date.isoformat()
                        data_send = {'latitude': latitude, 'longitude': longitude, 'date': date}
                        break


        return Response(json.dumps(data_send), mimetype='application/json')


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

    def __init__(self, filename, file, longitude, latitude, focal,date, size, identification):
        self.filename = filename
        self.file = file
        self.longitude =longitude
        self.latitude = latitude
        self.focal = focal
        self.date = date
        self.geo = 'POINT ({} {})'.format(longitude,latitude)
        self.size = size
        self.identification = identification




application = create_app()