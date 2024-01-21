import json
import tempfile
from flask import request, Blueprint, Response
from GPSPhoto import gpsphoto
from werkzeug.utils import secure_filename

#Récuperation position gps

def get_gpsinfo(img):
    data = gpsphoto.getGPSData(img)
    if (data):
        latitude = data['Latitude']
        longitude = data['Longitude']
    else :
        latitude = 0
        longitude = 0
    return latitude, longitude



