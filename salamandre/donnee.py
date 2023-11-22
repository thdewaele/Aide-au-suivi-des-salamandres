from flask import render_template, Blueprint

from salamandre import Pictures


def donnee():
    pictures = Pictures.query.all()
    return render_template('donnee.html', salamandres = Pictures)




bp = Blueprint('donnee', __name__, url_prefix='./donnee.html')
