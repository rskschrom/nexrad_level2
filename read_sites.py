import numpy as np
from fastkml import kml

# get nexrad site location
def nexrad_loc(radsite):
    # read nexrad data
    with open('Weather_Radar_Stations.kml', 'rt', encoding="utf-8") as myfile:
        doc = myfile.read()

    # parse kml file
    k = kml.KML()
    k.from_string(bytes(doc, encoding='utf8'))
    features = list(k.features())
    f2 = list(features[0].features())
    f3 = list(f2[0].features())
    site_name = []
    site_lon = []
    site_lat = []

    for fsub in f3:
        site_name.append(fsub.extended_data.elements[0].data[1]['value'])
        lon, lat = fsub.geometry.xy
        site_lon.append(lon[0])
        site_lat.append(lat[0])

    # get data at radar site
    radind = site_name.index(radsite)
    radlon = site_lon[radind]
    radlat = site_lat[radind]
    return radlon, radlat
