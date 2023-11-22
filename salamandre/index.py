from flask import render_template, Blueprint


def index():
    return render_template('index.html')




bp = Blueprint('index', __name__, url_prefix='./index.html')
