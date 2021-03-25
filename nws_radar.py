import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from pyart.io.nexrad_archive import read_nexrad_archive
from numpy import genfromtxt
import os
import glob
import time
from datetime import datetime, timedelta, timezone
from kdp_estimation import calc_kdp
from netCDF4 import Dataset

# function to de-alias phidp
#----------------------------------
def dealiasPhiDP(phiDP):
    deal_phi = np.ma.empty([phiDP.shape[0], phiDP.shape[1]])
    deal_phi[phiDP<0.] = 180.+phiDP[phiDP<0.] 
    deal_phi[phiDP>=0.] = phiDP[phiDP>=0.]
    return deal_phi   

# function for creating color map
#----------------------------------
def createCmap(mapname):
    fil = open('/home/robert/research/nws/'+mapname+'.rgb')
    cdata = genfromtxt(fil,skip_header=2)
    cdata = cdata/256
    cmap = cm.ListedColormap(cdata, mapname)
    fil.close()
    return cmap

# function to convert x,y to lon,lat
#-----------------------------------
def xy2latlon(x, y, lat0, lon0):
    km2deg = 110.62
    lat = y/km2deg+lat0
    lon = x/(km2deg*np.cos(np.pi*lat0/180.))+lon0
    return lat, lon

# function to convert x,y to lon,lat
#-----------------------------------
def lonlat2xy(lon, lat, lat0, lon0):
    km2deg = 110.62
    y = (lat-lat0)*km2deg
    x = (lon-lon0)*(km2deg*np.cos(np.pi*lat0/180.))
    return x, y

# extend azimuthal grid
#-----------------------------------
def extend_azi(field):
    dims = field.shape
    nrad = dims[0]
    ngat = dims[1]

    field_ext = np.empty([nrad+1, ngat])
    field_ext[0:nrad,:] = field
    field_ext[nrad,:] = field[0,:]

    return field_ext

# plot single variable from sweep
#-----------------------------------
def plot_single(site, fpath, ds_set=150., xcen_set=150., ycen_set=70., sw_ang=0.5, range_rings=False, mode='severe'):
    t1 = time.time()
    # open radar file
    radar = read_nexrad_archive(fpath)
    radlat = radar.latitude['data'][0]
    radlon = radar.longitude['data'][0]
    nyvel = radar.get_nyquist_vel(1)
    t2 = time.time()
    print(f'opening time, {t2-t1:.4f}')

    # plot stuff
    mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    mpl.rc('text', usetex=True)

    facecolor = 'white'
    edgecolor = 'black'

    mpl.rcParams['axes.facecolor'] = facecolor
    mpl.rcParams['savefig.facecolor'] = facecolor

    mpl.rcParams['axes.edgecolor'] = edgecolor
    mpl.rcParams['figure.edgecolor'] = edgecolor
    mpl.rcParams['savefig.edgecolor'] = edgecolor

    mpl.rcParams['text.color'] = edgecolor
    mpl.rcParams['axes.labelcolor'] = edgecolor
    mpl.rcParams['xtick.color'] = edgecolor
    mpl.rcParams['ytick.color'] = edgecolor
    figcnt = 0

    # get lat lon corners of data domain
    ds = ds_set
    xcen = xcen_set
    ycen = ycen_set

    #minlat, minlon = xy2latlon(xcen-ds, ycen-ds, radlat, radlon)
    #maxlat, maxlon = xy2latlon(xcen+ds, ycen+ds, radlat, radlon)
    minx = xcen-ds
    maxx = xcen+ds
    miny = ycen-ds
    maxy = ycen+ds

    xlabel = 'X-distance (km)'
    ylabel = 'Y-distance (km)'

    # colormaps
    zh_map = createCmap('zh2_map')
    zdr_map = createCmap('zdr_map')
    vel_map = createCmap('vel2_map')
    phv_map = createCmap('phv_map')
    kdp_map = createCmap('kdp_map')

    # set ranges of colormaps based on mode
    if mode=='winter':
        zh_min = -10.
        zh_max = 60.
        zdr_min = -2.4
        zdr_max = 6.9
        kdp_min = -1.6
        kdp_max = 4.3
        #rhohv_min = 0.695
        #rhohv_max = 1.045
        rhohv_min = 0.81
        rhohv_max = 1.01
    if mode=='severe':
        zh_min = 10.
        zh_max = 80.
        zdr_min = -2.4
        zdr_max = 6.9
        kdp_min = -3.2
        kdp_max = 8.6
        #rhohv_min = 0.695
        #rhohv_max = 1.045
        rhohv_min = 0.6
        rhohv_max = 1.0

    # set common font sizes
    cblb_fsize = 18
    cbti_fsize = 16
    axtl_fsize = 20
    axlb_fsize = 20
    axti_fsize = 18

    # height contours
    clevs = [3.,6.,9.,12.]
    fmt_dict = {}
    for cl in clevs:
        fmt_dict[cl] = f'{cl:.0f} km'

    # cartopy features
    t2 = time.time()
    cfile = Dataset(f'site_geom/{site}_counties.nc', 'r')
    cnt_verts = cfile.variables['vertex'][:]
    cnt_codes = cfile.variables['vertex_code'][:]

    sfile = Dataset(f'site_geom/{site}_states.nc', 'r')
    st_verts = sfile.variables['vertex'][:]
    st_codes = sfile.variables['vertex_code'][:]

    cnt_path = Path(cnt_verts, cnt_codes)
    st_path = Path(st_verts, st_codes)
    t3 = time.time()
    print(f'cartopy features time, {t3-t2:.4f}')

    # get sweeps
    fixed_angles = radar.fixed_angle['data']
    print(fixed_angles)
    nang = len(fixed_angles)
    sw05 = np.arange(nang)[np.abs(fixed_angles-sw_ang)<0.2]

    # seperate out z and mdv sweeps
    sw_inds = sw05[::2]
    swv_inds = sw05[1::2]
    #sw_inds = [sw_inds[0]]

    # loop over sweeps
    for sw in sw_inds:
        t2 = time.time()
        rd1 = time.time()
        azi = 90.-radar.get_azimuth(sw)
        elev = radar.get_elevation(sw) 
        ran = radar.range['data']

        sweep = radar.extract_sweeps([sw])
        fixed_angle = fixed_angles[sw]
        if fixed_angle<2.:
            sweep_v = radar.extract_sweeps([sw+1])
        else:
            sweep_v = sweep

        # calculate sweep time
        vol_time = sweep.time['units']
        sw_toffs = sweep.time['data'][0]
        sw_time = datetime.strptime(vol_time, 'seconds since %Y-%m-%dT%H:%M:%SZ')
        sw_time = sw_time+timedelta(seconds=sw_toffs)
        sw_time = datetime(year=sw_time.year, month=sw_time.month,
                           day=sw_time.day, hour=sw_time.hour,
                           minute=sw_time.minute, second=sw_time.second, tzinfo=timezone.utc)

        # get time strings
        yyyy = '{:04d}'.format(sw_time.year)
        mm = '{:02d}'.format(sw_time.month)
        dd = '{:02d}'.format(sw_time.day)
        hh = '{:02d}'.format(sw_time.hour)
        mn = '{:02d}'.format(sw_time.minute)
        ss = '{:02d}'.format(sw_time.second)

        print(yyyy, mm, dd, hh, mn, ss)

        ref = sweep.fields['reflectivity']['data']
        zdr = sweep.fields['differential_reflectivity']['data']
        rhohv = sweep.fields['cross_correlation_ratio']['data']
        vel = sweep_v.fields['velocity']['data']
        phidp = sweep.fields['differential_phase']['data']
        swd = sweep_v.fields['spectrum_width']['data']

        if fixed_angle<2.:
            azi_v = 90.-radar.get_azimuth(sw+1)
            elev_v = radar.get_elevation(sw+1)
        else:
            azi_v = 90.-radar.get_azimuth(sw)
            elev_v = radar.get_elevation(sw)

        dims = ref.shape
        numradials = dims[0]
        numgates = dims[1]

        angle = np.mean(elev)
        angle_v = np.mean(elev_v)

        # extend azimuth by one
        azi_ext = np.empty([numradials+1])
        azi_ext[0:numradials] = azi
        azi_ext[numradials] = azi[0]
        azi = azi_ext[:]

        azi_ext = np.empty([numradials+1])
        azi_ext[0:numradials] = azi_v
        azi_ext[numradials] = azi_v[0]
        azi_v = azi_ext[:]

        ref = extend_azi(ref)
        zdr = extend_azi(zdr)
        phidp = extend_azi(phidp)
        rhohv = extend_azi(rhohv)
        swd = extend_azi(swd)
        vel = extend_azi(vel)

        # mask data by rhohv and threshold
        #-----------------------------------------------
        ref = np.ma.masked_where(rhohv<0.4, ref)
        zdr = np.ma.masked_where(rhohv<0.4, zdr)
        phidp = np.ma.masked_where(rhohv<0.4, phidp)
        vel = np.ma.masked_where(rhohv<0.4, vel)
        swd = np.ma.masked_where(rhohv<0.4, swd)
        rhohv = np.ma.masked_where(rhohv<0.4, rhohv)

        zdr = np.ma.masked_where(ref<zh_min, zdr)
        phidp = np.ma.masked_where(ref<zh_min, phidp)
        rhohv = np.ma.masked_where(ref<zh_min, rhohv)
        vel = np.ma.masked_where(ref<zh_min, vel)
        swd = np.ma.masked_where(ref<zh_min, swd)
        ref = np.ma.masked_where(ref<zh_min, ref)
        rd2 = time.time()
        print('read data time', rd2-rd1)

        # calculate kdp
        #-----------------------------------------------
        print('Calculating KDP...')
        kd1 = time.time()
        phidp = dealiasPhiDP(phidp)
        kdp, delta, phidp_alt = calc_kdp(phidp, 0.25)
        kdp = np.ma.masked_where(ref<-15., kdp)
        kd2 = time.time()
        print('kdp calc time', kd2-kd1)

        # calculate x and y coordinates (wrt beampath) for plotting
        #-----------------------------------------------------------
        ct1 = time.time()
        ran_2d = np.tile(ran,(numradials+1,1))
        azi.shape = (azi.shape[0], 1)
        azi_v.shape = (azi_v.shape[0], 1)
        azi_2d = np.tile(azi,(1,numgates))
        azi_v_2d = np.tile(azi_v,(1,numgates))

        radz = 10.
        erad = np.pi*angle/180.
        erad_v = np.pi*angle_v/180.

        ke = 4./3.
        a = 6378137.

        # beam height and beam distance
        zcor = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad))-ke*a+radz
        scor = ke*a*np.arcsin(ran_2d*np.cos(erad)/(ke*a+zcor))/1000.

        xcor = ran_2d*np.cos(np.pi*azi_2d/180.)/1.e3
        ycor = ran_2d*np.sin(np.pi*azi_2d/180.)/1.e3

        # for velocity
        zcor_v = np.sqrt(ran_2d**2.+(ke*a)**2.+2.*ran_2d*ke*a*np.sin(erad_v))-ke*a+radz
        scor_v = ke*a*np.arcsin(ran_2d*np.cos(erad_v)/(ke*a+zcor))/1000.

        xcor_v = scor_v*np.cos(np.pi*azi_v_2d/180.)
        ycor_v = scor_v*np.sin(np.pi*azi_v_2d/180.)

        # convert to lon,lat for map plotting
        lat, lon = xy2latlon(xcor, ycor, radlat, radlon)
        lat_v, lon_v = xy2latlon(xcor_v, ycor_v, radlat, radlon)
        ct2 = time.time()
        print('make coordinates time', ct2-ct1)
        t3 = time.time()
        print(f'data prep time, {t3-t2:.4f}')

        # plot
        #---------------------
        print('Plotting...')
        fig = plt.figure(figcnt)
        figcnt = figcnt+1

        # ZH plot
        #------------------------------
        t2 = time.time()
        ax = fig.add_subplot(2,2,1)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #plt.pcolormesh(xcor, ycor, ref, cmap=zh_map, vmin=-10., vmax=80.)
        plt.pcolormesh(xcor, ycor, ref, cmap=zh_map, vmin=zh_min, vmax=zh_max)
        cb = plt.colorbar(format='%.0f', fraction=0.045)

        if range_rings:
            cs = plt.contour(xcor, ycor, zcor/1.e3, levels=clevs, linewidths=2., colors='k', linestyles='--')
            ax.clabel(cs, inline=1, fmt=fmt_dict, fontsize=axti_fsize)

        ax.set_title('Z$\sf{_H}$ (dBZ)', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)
        cnt_patch = patches.PathPatch(cnt_path, edgecolor=edgecolor, fill=False, lw=0.3)
        st_patch = patches.PathPatch(st_path, edgecolor=edgecolor, fill=False, lw=0.6)
        ax.add_patch(cnt_patch)
        ax.add_patch(st_patch)
        ax.set_aspect(1.)
        ax.set_facecolor(facecolor)
        ax.set_xlim([minx,maxx])
        ax.set_ylim([miny,maxy])
        t3 = time.time()
        print(f'zh time, {t3-t2:.4f}')

        # ZDR plot
        #------------------------------
        t2 = time.time()
        ax = fig.add_subplot(2,2,2)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #plt.pcolormesh(xcor, ycor, zdr, cmap=zdr_map, vmin=-2.4, vmax=6.9)
        plt.pcolormesh(xcor, ycor, zdr, cmap=zdr_map, vmin=zdr_min, vmax=zdr_max)
        cb = plt.colorbar(format='%.1f', fraction=0.045)

        if range_rings:
            cs = plt.contour(xcor, ycor, zcor/1.e3, levels=clevs, linewidths=2., colors='k', linestyles='--')
            ax.clabel(cs, inline=1, fmt=fmt_dict, fontsize=axti_fsize)

        ax.set_title('Z$\sf{_{DR}}$ (dB)', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)
        cnt_patch = patches.PathPatch(cnt_path, edgecolor=edgecolor, fill=False, lw=0.3)
        st_patch = patches.PathPatch(st_path, edgecolor=edgecolor, fill=False, lw=0.6)
        ax.add_patch(cnt_patch)
        ax.add_patch(st_patch)
        ax.set_aspect(1.)
        ax.set_facecolor(facecolor)
        ax.set_xlim([minx,maxx])
        ax.set_ylim([miny,maxy])
        t3 = time.time()
        print(f'zdr time, {t3-t2:.4f}')

        # KDP plot
        #------------------------------
        t2 = time.time()
        ax = plt.subplot(2,2,3)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #plt.pcolormesh(xcor, ycor, kdp, cmap=kdp_map, vmin=-1.6, vmax=4.3)
        plt.pcolormesh(xcor, ycor, kdp, cmap=kdp_map, vmin=kdp_min, vmax=kdp_max)
        cb = plt.colorbar(format='%.1f', fraction=0.045)

        if range_rings:
            cs = plt.contour(xcor, ycor, zcor/1.e3, levels=clevs, linewidths=2., colors='k', linestyles='--')
            ax.clabel(cs, inline=1, fmt=fmt_dict, fontsize=axti_fsize)

        ax.set_title('$\sf{K_{DP}}$ ($^{\circ}$ km$^{-1}$)', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)
        cnt_patch = patches.PathPatch(cnt_path, edgecolor=edgecolor, fill=False, lw=0.3)
        st_patch = patches.PathPatch(st_path, edgecolor=edgecolor, fill=False, lw=0.6)
        ax.add_patch(cnt_patch)
        ax.add_patch(st_patch)
        ax.set_aspect(1.)
        ax.set_facecolor(facecolor)
        ax.set_xlim([minx,maxx])
        ax.set_ylim([miny,maxy])
        t3 = time.time()
        print(f'kdp time, {t3-t2:.4f}')

        
        # RhoHV plot
        #------------------------------
        t2 = time.time()
        ax = plt.subplot(2,2,4)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #plt.pcolormesh(xcor, ycor, rhohv, cmap=phv_map, vmin=0.695, vmax=1.045)
        #plt.pcolormesh(xcor, ycor, rhohv, cmap=phv_map, vmin=rhohv_min, vmax=rhohv_max)
        plt.pcolormesh(xcor, ycor, rhohv, cmap='Spectral_r', vmin=rhohv_min, vmax=rhohv_max)
        cb = plt.colorbar(format='%.2f', fraction=0.045)

        if range_rings:
            cs = plt.contour(xcor, ycor, zcor/1.e3, levels=clevs, linewidths=2., colors='k', linestyles='--')
            ax.clabel(cs, inline=1, fmt=fmt_dict, fontsize=axti_fsize)

        ax.set_title('$\\rho\sf{_{HV}}$', x=0.0, y=1.02, horizontalalignment='left',
                      fontsize=axtl_fsize)
        cnt_patch = patches.PathPatch(cnt_path, edgecolor=edgecolor, fill=False, lw=0.3)
        st_patch = patches.PathPatch(st_path, edgecolor=edgecolor, fill=False, lw=0.6)
        ax.add_patch(cnt_patch)
        ax.add_patch(st_patch)
        ax.set_aspect(1.)
        ax.set_facecolor(facecolor)
        ax.set_xlim([minx,maxx])
        ax.set_ylim([miny,maxy])
        t3 = time.time()
        print(f'rhohv time, {t3-t2:.4f}')

        # save image as .png
        #-------------------------------
        title = '{} - {}/{}/{} - {}:{} UTC - {:.1f} deg. PPI'.format(site, yyyy, mm, dd,
                                                                     hh, mn, float(angle))
        plt.suptitle(title, fontsize=24, y=1.06)
        plt.subplots_adjust(top=0.96, hspace=0.2, wspace=0.2)
        imgname = yyyy+mm+dd+'_'+hh+mn+'_'+site.lower()+'.png'
        t2 = time.time()
        plt.savefig(imgname, format='png', dpi=90, bbox_inches='tight')
        plt.close()
        t3 = time.time()
        print(f'render time, {t3-t2:.4f}')
