
from geoalchemy2 import Geometry
from flask import Flask, Blueprint
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from geoalchemy2 import Geometry
from salamandre.img import getimgfromjs
from flask import current_app, g
bp = Blueprint('db', __name__, url_prefix='')
def init_db():
    db = get_db()

def get_db():
    if 'db' not in g:
        g.db = SQLAlchemy(current_app)
        current_app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:19992bbL@localhost/Salamandre_webapp'
        current_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        current_app.secret_key = 'secret string'




"""
#see on https://www.digitalocean.com/community/tutorials/how-to-use-a-postgresql-database-in-a-flask-application on 30/03/23

import os
import psycopg2

conn = psycopg2.connect(
        host="localhost",
        database="salamandre_db",
        user=os.environ['admin_db'],
        password=os.environ['salamandre'])

# Open a cursor to perform database operations
cur = conn.cursor()

# Execute a command: this creates a new table
cur.execute('DROP TABLE IF EXISTS Pictures;')
cur.execute('CREATE TABLE Pictures (id serial PRIMARY KEY,'
                                    'filename VARCHAR(150) NOT NULL ,'
                                 'picture BLOB,'
                                 'date_added date DEFAULT CURRENT_TIMESTAMP);'
                                 )





conn.commit()

cur.close()
conn.close()
"""

"""
cur.execute('INSERT INTO books (title, author, pages_num, review)'
            'VALUES (%s, %s, %s, %s)',
            ('A Tale of Two Cities',
             'Charles Dickens',
             489,
             'A great classic!')
            )


cur.execute('INSERT INTO books (title, author, pages_num, review)'
            'VALUES (%s, %s, %s, %s)',
            ('Anna Karenina',
             'Leo Tolstoy',
             864,
             'Another great classic!')
            )
"""
