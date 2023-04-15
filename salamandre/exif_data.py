import image
from GPSPhoto import gpsphoto


# read on 31/01/2023 on https://pypi.org/project/gpsphoto/
def get_gpsinfo(img):
    print("Image: ", img)
    data = gpsphoto.getGPSData(img)
    print(data)
    latitude = data['Latitude']
    longitude = data['Longitude']
    return latitude, longitude


if __name__ == "__main__":
    lat, long = (get_gpsinfo("IMG_0983.jpg"))
    print("latitude : ", lat)
    print("longitude : ", long)


"""
def get_loc(img):
    exif_table = {}
    image = Image.open(img)
    info = image.getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        exif_table[decoded] = value

    print(exif_table)
    gps_info = {}
    for key in exif_table['GPSInfo'].keys():
        decode = GPSTAGS.get(key, key)
        gps_info[decode] = exif_table['GPSInfo'][key]
    return gps_info



def get_field(exif, field):
    for (k,v) in exif.items():
        if TAGS.get(k) == field:
            return v
"""




"""
def get_exif(img):
    exif_table = {}
    image = Image.open(img)
    info = image.getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
        if decoded == "GPSInfo":
            gps_data = {}
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = value[t]
            exif_table[decoded] = gps_data
        else:
            exif_table[decoded] = value
    else:
        print("No info")
    return exif_table


def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None


def get_lat_lon(exif_data):
    gps_latitude = 0
    gps_longitude = 0
    gps_latitude_ref = 0
    gps_longitude_ref = 0
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, "GPSLatitudeRef")
        gps_longitude = _get_if_exist(gps_info, "GPSLongitude")
        gps_longitude_ref = _get_if_exist(gps_info, "GPSLongitudeRef")
    return gps_latitude, gps_latitude_ref,gps_longitude,gps_longitude_ref


"""