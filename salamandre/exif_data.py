
from datetime import datetime

import image
from GPSPhoto import gpsphoto
from PIL import Image
from PIL.ExifTags import TAGS


# read on 31/01/2023 on https://pypi.rg/project/gpsphoto/
def get_gpsinfo(img):
    print("Image: ", img)
    data = gpsphoto.getGPSData(img)
    print(data)
    latitude = data['Latitude']
    longitude = data['Longitude']
    return latitude, longitude

def get_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == "DateTimeOriginal":
                        date_part, heure_part = value.split(' ')
                        date_sql = datetime.strptime(date_part, "%Y:%m:%d").date()
                        return date_sql
            return None
    except (AttributeError, KeyError, IndexError):
        return None


if __name__ == "__main__":
    date = get_exif_data("IMG_0983.jpg")
    #lat, long = (get_gpsinfo("IMG_0983.jpg"))
    print('date: ', date)

    #print("latitude : ", lat)
    #print("longitude : ", long)

