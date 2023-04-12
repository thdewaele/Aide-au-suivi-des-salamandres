import os

import psycopg2
from flask import Flask, jsonify, request

from salamandre.index import index
from salamandre.img import getimgfromjs


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'salamandre.sqlite')
    )

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
