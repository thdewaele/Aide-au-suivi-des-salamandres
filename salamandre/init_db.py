
from geoalchemy2 import Geometry
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from geoalchemy2 import Geometry
from salamandre.img import getimgfromjs

db = SQLAlchemy()




class Pictures (db.Model):
    __tablename__ = 'Pictures'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(50))
    file = db.Column(db.bytea)
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
