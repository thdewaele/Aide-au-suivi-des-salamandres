import json
import os
import tempfile

import psycopg2
from flask import Flask, jsonify, request, Blueprint, Response

from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry
from werkzeug.utils import secure_filename

from salamandre.img import get_gpsinfo
from salamandre.index import index


db = SQLAlchemy()
size = 0

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

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
        return index()


    @app.route('/addpict', methods=['POST'])
    def add_pictures():
        dataset = request.files['photo']
        filename = secure_filename(dataset.name)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            dataset.save(temp.name)


        latitude, longitude = get_gpsinfo(temp.name)
        file = temp.name
        data = Pictures(dataset.name, file.encode('ascii'), longitude, latitude, 0, 0,size)
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
        print(dataset)
        taille = dataset.get('size')
        derniere_ligne = Pictures.query.order_by(Pictures.id.desc()).first()
        derniere_ligne.size = taille
        db.session.commit()
        size = float(taille)
        print(taille)
        answer ={
            'size': taille
        }
        return Response(json.dumps(answer), mimetype='application/json')





    from . import img
    app.register_blueprint(img.bp)

    #db.create_all()
    return app






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

    def __init__(self, filename, file, longitude, latitude, focal,date, size):
        self.filename = filename
        self.file = file
        self.longitude =longitude
        self.latitude = latitude
        self.focal = focal
        self.data = date
        self.geo = 'POINT ({} {})'.format(longitude,latitude)
        self.size = size






"""

def picturesadd(filename,file, longitude, latitude,focal,date):

    geo = 'POINT({} {})'.format(longitude, latitude)
    data = Pictures(filename=filename,
                    file=file,
                    longitude=longitude,
                    latitude=latitude,
                    geo=geo,
                    focal=focal,
                    date=date)

    db.session.add(data)
    db.session.commit()
    return render_template("index.html")

class Pictures (db.Model):
    __tablename__ = 'Pictures'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(50))
    file = db.Column(db.LargeBinary)
    longitude = db.Column(db.Float)
    latitude = db.Column(db.Float)
    geo = db.Column(Geometry(geometry_type="POINT"))
    focal = db.Column(db.Integer)
    date = db.Column(db.Date)

    @classmethod
    def add_pictures(cls, filename, file, longitude, latitude, focal, date):


        geo = 'POINT({} {})'.format(longitude, latitude)
        data = Pictures(filename=filename,
                        file = file,
                           longitude=longitude,
                           latitude=latitude,
                          geo=geo,
                        focal = focal,
                        date = date)

        db.session.add(data)
        db.session.commit()

    @classmethod
    def update_geometries(cls):

"""
"""
Using each city's longitude and latitude, add geometry data to db."""
"""
        pict = Pictures.query.all()

        for p in pict:
            point = 'POINT({} {})'.format(p.longitude, p.latitude)
            p.geo = point

        db.session.commit()
        
"""