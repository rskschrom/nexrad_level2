import boto3
from botocore.handlers import disable_signing
from numpy import array, argmin, abs
from datetime import datetime, timedelta
import os
#from nws_ppi_times import plot_low_sweeps, plot_single
from nws_radar import plot_single, kdp_compare
from nws_qvp import single_qvp
import cartopy_features as cf
import glob

# function to convert string time 'HHMMSS' into number of seconds past midnight
def dtstring2secs(dtstring):
    yyyy = int(dtstring[0:4])
    mm = int(dtstring[4:6])
    dd = int(dtstring[6:8])

    h = int(dtstring[9:11])  
    m = int(dtstring[11:13])
    s = int(dtstring[13:15])
    ts = datetime(yyyy, mm, dd, h, m, s).timestamp()
    return ts

# set up access to aws server
s3 = boto3.resource('s3')
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
bucket = s3.Bucket('noaa-nexrad-level2')

# set radar site and set time as now
radsite = 'KLWX'
'''
year = 2020
month = 8
day = 27
hour = 5
minute = 0
end = datetime(year, month, day, hour, minute)
'''
code_min = 1
end = datetime.utcnow()+timedelta(minutes=code_min)

# get beginning loop time
loop_min = 1
loop_len = timedelta(minutes=loop_min)
start = end-loop_len

# merge all files between start and end days
syyyy = start.year
smm = start.month
sdd = start.day
sday_dt = datetime(syyyy, smm, sdd, 0, 0)

eyyyy = end.year
emm = end.month
edd = end.day
eday_dt = datetime(eyyyy, emm, edd, 0, 0)

# loop over days between start and end times
nday = (eday_dt-sday_dt).days
keys = []
fnames = []
times = []

for i in range(nday+1):
    now = sday_dt+timedelta(days=i)
    yyyy = now.year
    mm = now.month
    dd = now.day
    print(yyyy, mm, dd)

    # read files from server
    prefix = '{:04d}/{:02d}/{:02d}/{}'.format(yyyy, mm, dd, radsite)
    objs = bucket.objects.filter(Prefix=prefix)
    keys_new = [o.key for o in objs]
    fnames_new = [k.split('/')[-1] for k in keys_new]
    times_new = [f.replace(radsite, '').replace('_V06', '') for f in fnames_new]

    # append to arrays for all days
    keys = keys + keys_new
    fnames = fnames + fnames_new
    times = times + times_new

# loop over times and download files
print(fnames)
nexrad_files = []

for i in range(loop_min):
    now = start+timedelta(minutes=i)
    yyyy = now.year
    mm = now.month
    dd = now.day
    hh = now.hour
    mn = now.minute
    ss = now.second

    # get file with closest time to want time
    secs = [dtstring2secs(t) for t in times]
    want_time = f'{yyyy:04d}{mm:02d}{dd:02d}_{hh:02d}{mn:02d}{ss:02d}'
    want_secs = dtstring2secs(want_time)
    secs_arr = array(secs)
    closeind = argmin(abs(secs_arr-want_secs))

    # download file
    dkey = keys[closeind]
    dfile = fnames[closeind]
    if not os.path.isfile(dfile):
        s3_client = boto3.client('s3')
        s3_client.meta.events.register('choose-signer.s3.*', disable_signing)
        s3_client.download_file('noaa-nexrad-level2', dkey, dfile)
        os.system('gunzip {}'.format(dfile))

    # add filename to list of ones to plot
    if not dfile in nexrad_files:
        nexrad_files.append(dfile)
        print(nexrad_files)


# check if geometry files have been created
if not os.path.isfile(f'site_geom/{radsite}_counties.nc'):
    print('making county outlines...')
    cf.radsite_counties(radsite)
if not os.path.isfile(f'site_geom/{radsite}_states.nc'):
    print('making state outlines...')
    cf.radsite_states(radsite)

# plot ppis
for nf in nexrad_files:
    plot_single(radsite, nf, ds_set=40., xcen_set=45., ycen_set=0., sw_ang=0.5, range_rings=False, mode='severe', npanel=4)
    #single_qvp(radsite, nf, sw_ang=5.)
    #kdp_compare(radsite, nf, ds_set=150., xcen_set=0., ycen_set=0., sw_ang=0.5, range_rings=False, mode='severe')
