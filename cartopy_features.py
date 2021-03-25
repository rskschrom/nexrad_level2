import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from metpy.plots import USCOUNTIES
from read_sites import nexrad_loc
from netCDF4 import Dataset

# function to convert x,y to lon,lat
def lonlat2xy(lon, lat, lon0, lat0):
    km2deg = 110.62
    y = (lat-lat0)*km2deg
    x = (lon-lon0)*(km2deg*np.cos(np.pi*lat0/180.))
    return x, y

# create county vertices for radar site
def radsite_counties(site):
    radlon, radlat = nexrad_loc(site)

    # read all vertices
    uscnt = USCOUNTIES.with_scale('20m').geometries()
    cntx = []
    cnty = []
    for usc in uscnt:
        for poly in usc:
            lon, lat = poly.exterior.xy

            # convert to radar-relative x-y coordinates
            x, y = lonlat2xy(np.array(lon), np.array(lat), radlon, radlat)
            cntx.append(x)
            cnty.append(y)

    # subset county polygons close enough to radar site
    ncnt = len(cntx)
    cntx_sub = []
    cnty_sub = []
    maxdist = 450.

    for i in range(ncnt):
        dist = np.sqrt(cntx[i]**2.+cnty[i]**2.)
        if np.min(dist)<maxdist:
            cntx_sub.append(cntx[i])
            cnty_sub.append(cnty[i])

    # create matplotlib path arrays
    npoly_cnt = len(cntx_sub)
    verts = np.vstack((cntx_sub[0],cnty_sub[0])).T
    codes = np.full(len(cntx_sub[0]), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    for i in range(npoly_cnt-1):
        verts = np.concatenate((verts, np.vstack((cntx_sub[i+1],cnty_sub[i+1])).T))
        codes_new = np.full(len(cntx_sub[i+1]), Path.LINETO)
        codes_new[0] = Path.MOVETO
        codes_new[-1] = Path.CLOSEPOLY
        codes = np.concatenate((codes, codes_new))

    # write to netcdf file
    nvert = len(codes)
    cfile = Dataset(f'site_geom/{site}_counties.nc', 'w')
    vind = cfile.createDimension('vert_ind', nvert)
    cdim = cfile.createDimension('coor_dim', 2)
    vert = cfile.createVariable('vertex', 'f4', ('vert_ind','coor_dim'))
    vertc = cfile.createVariable('vertex_code', 'i4', ('vert_ind'))

    vert.units = 'km'
    vertc.units = ''
    vert.description = 'distances of vertex from radar site'
    vertc.description = 'matplotlib path code for vertex'
    vert[:] = verts
    vertc[:] = codes
    cfile.close()

# create state vertices for radar site
def radsite_states(site):
    radlon, radlat = nexrad_loc(site)

    # read all vertices
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='10m',
                                         category='cultural', name=shapename)
    usst = shpreader.Reader(states_shp).geometries()
    stx = []
    sty = []
    for ust in usst:
        for poly in ust:
            lon, lat = poly.exterior.xy
            x, y = lonlat2xy(np.array(lon), np.array(lat), radlon, radlat)
            stx.append(x)
            sty.append(y)

    # subset county polygons close enough to radar site
    nst = len(stx)
    stx_sub = []
    sty_sub = []
    maxdist = 450.

    for i in range(nst):
        dist = np.sqrt(stx[i]**2.+sty[i]**2.)
        if np.min(dist)<maxdist:
            stx_sub.append(stx[i])
            sty_sub.append(sty[i])

    # create matplotlib path arrays
    npoly_st = len(stx_sub)
    verts = np.vstack((stx_sub[0],sty_sub[0])).T
    codes = np.full(len(stx_sub[0]), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    for i in range(npoly_st-1):
        verts = np.concatenate((verts, np.vstack((stx_sub[i+1],sty_sub[i+1])).T))
        codes_new = np.full(len(stx_sub[i+1]), Path.LINETO)
        codes_new[0] = Path.MOVETO
        codes_new[-1] = Path.CLOSEPOLY
        codes = np.concatenate((codes, codes_new))

    # write to netcdf file
    nvert = len(codes)
    sfile = Dataset(f'site_geom/{site}_states.nc', 'w')
    vind = sfile.createDimension('vert_ind', nvert)
    cdim = sfile.createDimension('coor_dim', 2)
    vert = sfile.createVariable('vertex', 'f4', ('vert_ind','coor_dim'))
    vertc = sfile.createVariable('vertex_code', 'i4', ('vert_ind'))

    vert.units = 'km'
    vertc.units = ''
    vert.description = 'distances of vertex from radar site'
    vertc.description = 'matplotlib path code for vertex'
    vert[:] = verts
    vertc[:] = codes
    sfile.close()

