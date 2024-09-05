# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:45:34 2023

@author: 19855
"""

import xarray as xr
import numpy as np
import openpyxl as xlx
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings("ignore")

dir2 = '/home/gj/data2/rrtm/mie/merra2table/'    # directory of aerosol optical property file
dir3 = '/home/gj/data3/monthly/result/lw_alllev/'

IOUT = 0  # 99 for 17 intercals, 0 for all band, 16-29 for seperate band
ICLD = 0  # 0 for no cloud, 1-5 for different overlap assumption (2-5 need IMCA=1)

lenp = 37
D = 12  # cloud diameter, um
kernel_method = 1   # 0 for 10% increase, 1 for specific increase 

######## aer file ########
# mid point of each of the 16 spextral bands
mdpn = np.array([180,425,565,665,760,900,1030,1130,1285,1435,1640,1940,2165,2315,2490,5850]) # unit: cm-1
mdpl = 1/mdpn*1e-2  # unit:m
# calculate aod, ssa, g under different RH 
def match_rh(faer,name_m2,name_opt,rd,rh,thick,dens):
    q = faer[name_m2].data   # aerosol mixing ratio [kg/kg]
    ds = xr.open_dataset(dir2+'%s'%name_opt)  # input MERRA-2 optical properties
    ds = ds.rename({'lambda': 'lamb'})  # escape error related to lambda
    
    # get mass extinction efficiency beasd on different radius [m2/g]
    if rd < 1:  # sulfate, find out the corresponding radius
        kext0 = ds['bext'].sel(radius=rd,method='nearest').interp(lamb=mdpl)  # unit: m2/kg
        ssa0 = (ds['qsca']/ds['qext']).sel(radius=rd,method='nearest').interp(lamb=mdpl)
    else:  # different bins of aerosols
        kext0 = ds['bext'][rd-1].interp(lamb=mdpl)  # at 1um
        ssa0 = (ds['qsca']/ds['qext'])[rd-1].interp(lamb=mdpl)
    # find out the nearest rh
    rh0 = ds['rh']
    kext1 = np.empty((16,lenp-9))
    ssa1 = np.empty((16,lenp-9))
    for i in range(lenp-9):
        rh1 = np.abs(rh0-rh[i])
        plc = np.where(rh1==rh1.min())[0][0]
        kext1[:,i] = kext0.data[plc]
        ssa1[:,i] = ssa0.data[plc]
    aod1 = q*dens*1e3 * thick * kext1*1e-3
    aod2 = aod1*(1-ssa1)   # isolate absorption aod
    aod3 = np.nan_to_num(aod2)   # set nan value to 0
    return aod3 

def get_aer_opt(faer):
    dens = faer['AIRDENS'].data   # kg/m3
    delp = faer['DELP'].data  # pa  
    rh = np.nan_to_num(faer['RH'])
    # pressure (Pa), density of a gas (kg/m3), gravity acceleration (9.8 m/s2) --> height of a column of gas (m)
    thick = delp/(dens*9.8)
    
    # AOD  # constant: mass extinction coefficient [m2/g]
    aod_du1 = match_rh(faer,'DU001','optics_DU.v15_3.nc',1,rh,thick,dens)  # dust bin1
    aod_du2 = match_rh(faer,'DU002','optics_DU.v15_3.nc',2,rh,thick,dens)  # bin2
    aod_du3 = match_rh(faer,'DU003','optics_DU.v15_3.nc',3,rh,thick,dens)  # bin3
    aod_du4 = match_rh(faer,'DU004','optics_DU.v15_3.nc',4,rh,thick,dens)  # bin4
    aod_du5 = match_rh(faer,'DU005','optics_DU.v15_3.nc',5,rh,thick,dens)  # bin5
    aod_sa = match_rh(faer,'SO4', 'optics_SU.v1_3.nc', 0.16*1e-6, rh,thick,dens)
    aod_bco = match_rh(faer,'BCPHOBIC', 'optics_BC.v1_3.nc', 1, rh,thick,dens)  # hydrophobic bc
    aod_bci = match_rh(faer,'BCPHILIC', 'optics_BC.v1_3.nc', 2, rh,thick,dens)  # hydrophilic
    aod_oco = match_rh(faer,'OCPHOBIC', 'optics_OC.v1_3.nc', 1, rh,thick,dens)  # hydrophobic
    aod_oci = match_rh(faer,'OCPHILIC', 'optics_OC.v1_3.nc', 2, rh,thick,dens)  # hydrophilic

    # total
    aod = np.array([aod_du1,aod_du2,aod_du3,aod_du4,aod_du5,aod_bco,aod_bci,aod_oco,aod_oci,aod_sa])
    aodt = aod.sum(axis=0)
    return aodt

#### save to file ####
def save_aer_file(aodt,ptb):
    AOD0 = np.zeros((16,lenp-9))   
    # create AOD
    for pul in range(lenp-9):
        if pul == ptb:   # perturbed level
            if kernel_method == 0:    # ratio increase
                AOD0[:,pul] = aodt[:,pul] * dx
            elif kernel_method == 1:    # specific increase
                AOD0[:,pul] = aodt[:,pul] + dx[pul]
        else:
            AOD0[:,pul] = aodt[:,pul]
    
    aid = open('IN_AER_RRTM','w')

    ### 1 ###
    # number of aerosol species
    line = '    1\n'
    aid.write(line)
    ### 2 ###
    # number of layer, way to input AOD
    line = '   %2d    1\n'% (lenp-9)
    aid.write(line)
    ### 3 ###
    # AOD at each layer
    num = np.arange(10,lenp+1)    # only in uts
    for l in range(28):    
        line = '  %3d'%num[l]
        for n in range(16):
            line += '%7.5f'%AOD0[n,l]         # have samiliar order as atm file
            if n == 15:
                line += '\n'
        aid.write(line)
    return AOD0[:,::-1]     # from top to bottom
        
def save_cld_file(aodt,ptb):
    COD0 = np.zeros((16,lenp))   
    # create COD
    for pul in range(lenp):
        if pul == ptb:   # perturbed level
            if kernel_method == 0:    # ratio increase
                COD0[:,pul] = aodt[:,pul]*dx
            elif kernel_method == 1:    # specific increase
                COD0[:,pul] = aodt[:,pul] + dx[pul]
        else:
            COD0[:,pul] = aodt[:,pul]
            
    aid = open('IN_AER_RRTM','w')

    ### 1 ###
    # number of aerosol species
    line = '    1\n'
    aid.write(line)
    ### 2 ###
    # number of layer, way to input AOD
    line = '   %2d    1\n'% (lenp)
    aid.write(line)
    ### 3 ###
    # AOD at each layer
    num = np.arange(1,lenp+1)    # only in uts
    for l in range(lenp):    
        line = '  %3d'%num[l]
        for n in range(16):
            line += '%7.5f'%COD0[n,l]         # have samiliar order as atm file
            if n == 15:
                line += '\n'
        aid.write(line)
    return COD0[:,::-1]

######## cloud file ########
# longwave band limt from FU 1998
ffu = xlx.load_workbook('Fu-lw-cld.xlsx')  
st_f = ffu['Sheet1']
row1=[item.value for item in list(st_f.columns)[0]]  
wave_fu = np.array([float(i) for i in row1[1:37]])*1e-6  # convert unit from um to m
row2=[item.value for item in list(st_f.columns)[2]]  
a0 = np.array([float(i) for i in row2[1:37]])
row3=[item.value for item in list(st_f.columns)[3]]  
a1 = np.array([float(i) for i in row3[1:37]])
row4=[item.value for item in list(st_f.columns)[4]]  
a2 = np.array([float(i) for i in row4[1:37]])
row5=[item.value for item in list(st_f.columns)[6]]  
b0 = np.array([float(i) for i in row5[1:37]])
row6=[item.value for item in list(st_f.columns)[7]]  
b1 = np.array([float(i) for i in row6[1:37]])
row7=[item.value for item in list(st_f.columns)[8]]  
b2 = np.array([float(i) for i in row7[1:37]])
row8=[item.value for item in list(st_f.columns)[9]]  
b3 = np.array([float(i) for i in row8[1:37]])

extc0 = (a0+a1/D+a2/(D**2))
# absorption, = 1-ssa
abc0 = (b0+b1*D+b2*(D**2)+b3*(D**3))/(a0*D+a1+a2/D)

rlowb = 1/np.array([10,350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2380,2600])*1e-2
rhighb = 1/np.array([350,500,630,700,820,980,1080,1180,1390,1480,1800,2080,2250,2380,2600,3250])*1e-2

def get_cld_opt(fcld,fair):
    dens = fair['AIRDENS'].data   # kg/m3, bottom to top
    delp = fair['DELP'].data  # pa 
    # pressure (Pa), density of a gas (kg/m3), gravity acceleration (9.8 m/s2) --> height of a column of gas (m)
    thick = delp/(dens*9.8)
    iwc = fcld['IWC'].data*1e-3  # mg/m3 to g/m3, bottom to top
    extc1 = iwc*extc0[:,np.newaxis]  #(fu band, lenp)
    extc = np.zeros((16,lenp))
    abc = np.zeros(16)
    for k in range(16):
        band = (wave_fu > rhighb[k]) & (wave_fu <= rlowb[k])
        extc[k] = extc1[band].mean(axis=0)
        abc[k] = abc0[band].mean()
    cod = np.nan_to_num(thick*extc*abc[:,np.newaxis])
    cod[cod<0] = 0
    return cod   # from top to bottom
'''
def get_save_cloud(frad,faer,fatm):
    cldf = frad['CLOUD'].data[::-1]  # cloud fraction
    
    dens = faer['AIRDENS'].data   # kg/m3
    delp = faer['DELP'].data  # pa  
    # pressure (Pa), density of a gas (kg/m3), gravity acceleration (9.8 m/s2) --> height of a column of gas (m)
    thick = delp/(dens*9.8)
    
    ciw = fatm['QI'].data  # mass fraction of cloud ice water
    iwc = ciw*dens*1e3   # g/m3
    extc1 = iwc*extc0[:,np.newaxis]  #(fu band, lenp)
    abs_extc = extc1*abc[:,np.newaxis]
    extc = abs_extc.mean(axis=0)
    cod = np.nan_to_num(thick*extc)
    # save cloud file
    cid = open('IN_CLD_RRTM','w')

    ### 1 ###
    # flag of cloud
    line = '    0    2    1\n'
    cid.write(line)
    ### 2 ###
    # data at each layer
    num = np.arange(1,lenp+1)
    for j in range(lenp): 
        line = '  %3d%10.5f%10.5f\n'%(num[j],cldf[j],cod[j])              
        cid.write(line)
    ### 3 ###
    # end
    line = '%'
    cid.write(line)
    cid.close()
'''    
######## prepare atm data ########
def get_atm_data(fatm,frag,fts):  # in MERRA-2
    k =  1.380650e-23    # Boltzmann constant, J/K
    h = fatm['z'].data/9.8/1000     # convert unit from m2/s2 to km, height above surface
    p = frag['lev'].data   # unit:hpa
    t = fatm['t'].data
    ts = fts['skt'].data
    o3 = np.nan_to_num(frag['O3'].data)   # unit: ppmv, -> from bottom to top
    h2o = np.nan_to_num(frag['H2O'].data)   # unit: ppmv
    # calculate top t, top p and column density
    top_p = np.zeros(lenp+1)
    top_p[0] = p[0] - np.abs(p[0]-p[1])/2
    top_p[-1] = p[-1] - np.abs(p[-1]-p[-2])/2
    delt_z = np.zeros(lenp)
    delt_z[0] = np.abs(h[1] - h[0])*1e5
    for i in range(lenp-1):
        top_p[i+1] = (p[i]+p[i+1])/2
        delt_z[i+1] = np.abs(h[i+1] - h[i])*1e5  # convert unit from km to cm
    lineart = interp1d(np.log(p[1:]),t[1:], fill_value='extrapolate')
    top_t = lineart(np.log(top_p))
    top_t[0] = ts
    dens = p*1e2/(k*t)*1e-6    # unit:cm-3
    col_dens = dens*delt_z
    return p, t, ts, top_p, top_t, o3, h2o, col_dens

co2 = 525*29/44*1e-6*np.ones(lenp)   # constant co2 for RCP4.5

# those don't include in MERRA-2 should be get from rrtmg
f0 = xlx.load_workbook('rrtmg_sw_atm.xlsx')  
st1_f = f0['Sheet1']
row1=[item.value for item in list(st1_f.columns)[0]]  # pressure, hpa
pa = np.array([float(i) for i in row1[1:61]])
def interp(n,p):
    row2=[item.value for item in list(st1_f.columns)[n]]  # co2, ppv
    x = np.array([float(i) for i in row2[1:61]])
    linear1 = interp1d(np.log(pa),x,fill_value='extrapolate')
    x = linear1(np.log(p))
    return x

######## extract output ########
def get_output(band):
    oid = open('OUTPUT_RRTM','r')
    if band == 99:
        out = np.zeros((17,6,lenp+1))
        for i in range(17):
            for j in range(lenp+5):  # in one band range
                line = oid.readline()
                if j < 3 or j == lenp+4:  # first 5 line or final line
                    pass
                else:
                    out0 = line.strip().split(' ')
                    out1 = list(filter(None,out0))
                    for p in range(6):
                        try:
                            out[i,p,j-3] = float(out1[p])
                        except(ValueError):
                            out[i,p,j-3] = 999
                    #out[i,:,j-3] = [float(n) for n in list(filter(None,out0))]
    else:    
        out = np.zeros((6,lenp+1))
        for j in range(lenp+5):  # in one band range
            line = oid.readline()
            if j < 3 or j == lenp+4:  # first 5 line or final line
                pass
            else:
                out0 = line.strip().split(' ')
                out1 = list(filter(None,out0))
                for p in range(6):
                    try:
                        out[p,j-3] = float(out1[p])
                    except(ValueError):
                        out[p,j-3] = 999
                #out[:,j-3] = [float(n) for n in list(filter(None,out0))]
    oid.close()
    return out


####### loop over all grid and all time #######
def run_rrtm(ti,i):
    # time
    time = str(time7[ti].data)
    lat = latt[i]
    print(time,lat)
    
    faer = dsaer.sel(lat=lat).isel(time=ti)
    fatm = dsatm.sel(lat=lat).isel(time=ti)
    frag = dsrag.sel(lat=lat).isel(time=ti)
    fcld = dscld.sel(lat=lat).isel(time=ti)
    fts = dsts.sel(lat=lat).isel(time=ti)
    fair = dsair.sel(lat=lat).isel(time=ti%12)   # only one year

    # atm data
    p,t,ts,top_p,top_t,o3,h2o,col_dens = get_atm_data(fatm,frag,fts)  
    #co2 = interp(1,p)
    o2 = interp(5,p)
    n2o = interp(2,p)
    ch4 = interp(4,p)
    co = interp(3,p)
    
    # emissivity
    alb = fts['fal'].data
    emis = 1-alb
    
    print('finish preparing file')
    
    ### create input data ###
    fid = open('./LW_TEMPLATE', 'r')
    wid = open('./INPUT_RRTM', 'w')
    
    ### 1 ###
    for ii in range(3):
        line = fid.readline()
        wid.write(line)
    ### 2 ###
    # Set flag for aerosol(A), output(B), and Cloud(K).
    line = fid.readline()
    line = line.replace('AA', '%2d' % IAER)
    line = line.replace('BB', '%2d' % IOUT)
    line = line.replace('K', str(ICLD))
    wid.write(line)
    ### 3 ###
    # set surface temperature and emissivity.
    line = fid.readline()
    line = line.replace('SurfaceTem', '%10.3f' % ts)
    line = line.replace('EMISS', '%5.3f'% emis)
    wid.write(line)
    ### 4 ###
    # Set number format, number of layers, number of molecules
    line = fid.readline()
    line = line.replace('NLY', '%3d' % (lenp))
    wid.write(line)
    ### 5 ###
    # set atm data
    for l in range(lenp): 
        # pressure and temperature
        if l == 0:  # set pressures of RRTM SW calculation layer boundaries
            line1 = '%15.7e%10.4f                       %8.3f%7.2f       %8.3f%7.2f\n'\
                    %(p[0],t[0],top_p[0],top_t[0],top_p[1],top_t[1])                 
        else:
            line1 = '%15.7e%10.4f                                             %8.3f%7.2f\n'\
                    %(p[l],t[l],top_p[l+1],top_t[l+1])
        wid.write(line1)
        # input atms
        line2 = '%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%11.7f    %15.7e\n'\
                %(h2o[l],co2[l],o3[l],n2o[l],co[l],ch4[l],o2[l],col_dens[l])
        wid.write(line2)
    ### 6 ####
    wid.write('%%%%%')
    ### close files ###
    fid.close()
    wid.close()
    
    # aer data
    if IAER == 10:
        if aer_type == 3:   # cloud
            odt = get_cld_opt(fcld,fair)
            for ptb in range(17):   # from bottom to top
                od[ti,i,ptb,1:,1:] = save_cld_file(odt,ptb) 
                ######## run model ########
                #os.system('./rrtmg_lw_v19.1.3.304_linux_intel')
                #os.system('./rrtmg_lw_v19.1.1.217_linux_intel')
                os.system('./rrtmg_lw_v14.0.2_linux_intel')
                print('finish running')
                
                ### extract output ###
                output[ti,i,ptb] = get_output(IOUT)
        else:
            odt = get_aer_opt(faer)
            for ptb in range(17):   # from bottom to top
                od[ti,i,ptb,1:,1:lenp-8] = save_aer_file(odt,ptb)   # from top to bottom
                ######## run model ########
                #os.system('./rrtmg_lw_v19.1.3.304_linux_intel')
                #os.system('./rrtmg_lw_v19.1.1.217_linux_intel')
                os.system('./rrtmg_lw_v14.0.2_linux_intel')
                print('finish running')
                
                ### extract output ###
                output[ti,i,ptb] = get_output(IOUT)
    else:
        pass
    
######## set input parameters ########
dir1 = '/home/gj/data3/'
atm_input = dir1+'era5/era5_ztcld_2011-2020_monthly_zm.mlslat.nc'
rag_input = dir1+'mls/merge/MLS-gas_alllev_mon_2011-2020.nc'
aer_input = dir1+'monthly/aer_np/m2aer.uts.2011-2020.zm.mlslat.nc'
cld_input = dir1+'mls/merge/MLS-iwc_alllev_mon_2011-2020.nc'
ts_input = dir1+'era5/era5_albts_2011-2020_monthly_zm_mlslat.nc'
air_input = dir1+'monthly/air/m2_air_np_mon_2020.zm.mlslat.nc'

# interpolate to get same time
dsrag = xr.open_dataset(rag_input).sel(lat=slice(-30,30),lev=slice(1000,1))
lev0 = dsrag['lev'].data  # from top to bottom
latt = dsrag['lat'].data
time7 = dsrag['time']
dsatm = xr.open_dataset(atm_input).interp(level=lev0).sel(lat=slice(-30,30)).squeeze()
dsaer = xr.open_dataset(aer_input).interp(plev=lev0[9:37]*100).sel(lat=slice(-30,30)).squeeze()  # only in uts  # from bottom to top
dscld = xr.open_dataset(cld_input).sel(lat=slice(-30,30)).interp(lev=lev0)
dsts = xr.open_dataset(ts_input).interp(lat=latt).squeeze()
dsair = xr.open_dataset(air_input).interp(plev=lev0*100).sel(lat=slice(-30,30)).squeeze()

# calculate pressure output
p = dsrag['lev'].data[::-1]   # unit:hpa, top to surface
top_p = np.zeros(lenp+1)
top_p[0] = p[0] - np.abs(p[0]-p[1])/2
top_p[-1] = p[-1] - np.abs(p[-1]-p[-2])/2
for i in range(lenp-1):
    top_p[i+1] = (p[i]+p[i+1])/2
print(top_p)

lev = xr.DataArray(top_p,coords=[np.arange(lenp+1)],dims=['lev'])
ptb_lev = xr.DataArray(top_p[::-1][9:26],coords=[np.arange(17)],dims=['perturbed_lev'])

# get specific od increase
dsod = np.load('mean_vert_od10_2011-2020.npz')

for rr in range(2):
    # ratio increase
    if kernel_method == 0:
        if rr == 0:
            aer_type = 0
            dx = 1.1    # multiple of aod
            IAER = 10
            output_file = dir3+'rrtm_lw_2011-2020_monthly_zm_alllev_eachlev_aod10.nc'
        if rr == 1:
            aer_type = 3
            dx = 1.1    
            IAER = 10
            output_file = dir3+'rrtm_lw_2011-2020_monthly_zm_alllev_eachlev_cod10.nc'
    # specific increase
    elif kernel_method == 1:
        if rr == 0:
            aer_type = 0
            dx = np.ones(lenp)*1e-5  # too small that can't use 10% mean
            print(dx)
            IAER = 10
            output_file = dir3+'rrtm_lw_2011-2020_monthly_zm_alllev_add_eachlev_aod10.nc'
        if rr == 1:
            aer_type = 3
            dx = dsod['cod'][::-1] * 0.1  
            IAER = 10
            output_file = dir3+'rrtm_lw_2011-2020_monthly_zm_alllev_add_eachlev_cod10.nc'
    
    timlen = len(time7.data)
    latlen = len(latt.data)
    
    od = np.zeros((timlen,latlen,17,17,lenp+1))
    #aod = np.zeros((timlen,17,lenp+1))
    if IOUT == 99:
        output = np.zeros((timlen,latlen,17,17,6,lenp+1))
    else:
        output = np.zeros((timlen,latlen,17,6,lenp+1))
    for ti in range(timlen):
        for i in range(latlen):
            run_rrtm(ti,i)
            
    #### save file ####
    if IOUT == 99:  # all band range
        output1 = output.transpose((4,3,0,2,5,1))
        od1 = od.transpose((3,0,2,4,1))
        bandnum = np.arange(0,17)
        band = xr.DataArray(bandnum,coords=[bandnum],dims=['band'])
        ds = xr.Dataset({'p':(['time','lev'],output1[1,0,:,0,:,0]),
                         'up_flx':(['band','time','perturbed_lev','lev','lat'],output1[2]),
                         'dn_flx':(['band','time','perturbed_lev','lev','lat'],output1[3]),
                         'net_flx':(['band','time','perturbed_lev','lev','lat'],output1[4]),
                         'heat_rate':(['band','time','perturbed_lev','lev','lat'],output1[5]),
                         'od':(['band','time','perturbed_lev','lev','lat'],od1)},
                            coords={'lev':lev,'band':band,'time':time7,'lat':latt,'perturbed_lev':ptb_lev})
        
    else:
        output1 = output.transpose((3,0,2,4,1))
        od1 = od.transpose((3,0,2,4,1))[7]  # 10um
        ds = xr.Dataset({'p':(['time','lev'],output1[1,:,0,:,0]),
                         'up_flx':(['time','perturbed_lev','lev','lat'],output1[2]),
                         'dn_flx':(['time','perturbed_lev','lev','lat'],output1[3]),
                         'net_flx':(['time','perturbed_lev','lev','lat'],output1[4]),
                         'heat_rate':(['time','perturbed_lev','lev','lat'],output1[5]),
                         'od10':(['time','perturbed_lev','lev','lat'],od1)},
                            coords={'lev':lev,'time':time7,'lat':latt,'perturbed_lev':ptb_lev})
    ds.to_netcdf(output_file)
    print('finish saving')
