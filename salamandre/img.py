import json
import tempfile
from flask import request, Blueprint, Response
from GPSPhoto import gpsphoto
from werkzeug.utils import secure_filename

"""
bp = Blueprint('img', __name__, url_prefix='')
"""

def get_gpsinfo(img):
    data = gpsphoto.getGPSData(img)
    if (data):
        latitude = data['Latitude']
        longitude = data['Longitude']
    else :
        latitude = 0
        longitude = 0
    return latitude, longitude

bp = Blueprint('img', __name__, url_prefix='')




@bp.route('/img', methods=['POST'])
def getimgfromjs():
    dataset = request.files['photo']
    filename = secure_filename(dataset.name)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
        dataset.save(temp.name)

        # Process the image using the temporary file
        lat,long = get_gpsinfo(temp.name)
    answer = {
        'lat' :lat,
        'long' : long,
        'filename': filename
    }

    return Response(json.dumps(answer), mimetype='application/json')

"""
@bp.route('/img', methods=['POST'])
def getimgfromjs():
    dataset = request.files['photo']
    filename = secure_filename(dataset.name)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
        dataset.save(temp.name)

        # Process the image using the temporary file
        lat,long = get_gpsinfo(temp.name)
    print(lat,long)
    answer = {
        'lat' :lat,
        'long' : long,
        'filename': filename
    }
    add_pictures(dataset.name, temp,long,lat,0)
    return Response(json.dumps(answer), mimetype='application/json')
    """

"""
     if 'file' in request.files:
       photo = request.files['file']
       if photo.filename != '':
          image = request.files['file']
          image_string = base64.b64encode(image.read())
          image_string = image_string.decode('utf-8')
          # use this to remove b'...' to get raw string
          return render_template('img.html', filestring=image_string)
    print('no file')
 
 
 
     dataset = request.files.to_dict()
    print(dataset)
    list1 = list(dataset.items())
    filename = list1[0]
    print(filename)
    lat, long = Exif_data.get_gpsinfo(filename)
    print(lat, long)
    answer = {
       'filename': filename
    }
    return Response(json.dumps(answer), mimetype='application/json')
 
    """
"""
     dataset = request.files
     list1 = list(dataset.items())
     filename = str(list1[0])
     answer ={
        'filename': filename
     }
     return Response(json.dumps(answer), mimetype='application/json')
     """

"""
    data = request.get_json(silent=True)
    filename = data.get('filename')
    print(filename)
    lat, long = Exif_data.get_gpsinfo(filename)
    return render_template('img', latitude = lat, longitude = long)
    """