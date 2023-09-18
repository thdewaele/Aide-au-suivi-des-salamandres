from flask import render_template, Blueprint


def accueil():
    return render_template('Accueil.html')




bp = Blueprint('Accueil', __name__, url_prefix='./Accueil')