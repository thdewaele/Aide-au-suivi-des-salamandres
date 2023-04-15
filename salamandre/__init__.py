import os

import psycopg2
from flask import Flask, jsonify, request

from flask import Flask, render_template, request, flash
from flask_sqlalchemy import SQLAlchemy
from geoalchemy2 import Geometry

from salamandre.index import index


db = SQLAlchemy()
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:19992bbL@localhost/Salamandre_webapp'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = 'secret string'

    db = SQLAlchemy(app)


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


    from . import img
    app.register_blueprint(img.bp)

    return app

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
        """Put a new city in the database."""

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
        """Using each city's longitude and latitude, add geometry data to db."""

        pict = Pictures.query.all()

        for p in pict:
            point = 'POINT({} {})'.format(p.longitude, p.latitude)
            p.geo = point

        db.session.commit()