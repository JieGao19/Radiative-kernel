# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:02:19 2023

@author: 19855
"""
# insert ssa and g at each level
import xarray as xr
import numpy as np
import openpyxl as xlx
from scipy.interpolate import interp1d
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

dir2 = '/home/gj/data2/rrtm/mie/merra2table/'    # directory of aerosol optical property file
dir3 = '/home/gj/data3/monthly/result/sw_alllev/'

######## set input parameters ########
IOUT = 0  # 98 for 15 intervals, 0 for all band, 16-29 for seperate band
ICLD = 0  # 0 for no cloud, 1-5 for different overlap assumption (2-5 need IMCA=1)
    
lenp = 37
D = 12  # cloud diameter, um
kernel_method = 1   # 0 for 10% increase, 1 for specific increase 

#### calculate juldat and zenith angle ####
def rrtm_sw_tinfo(year, month, day, hour, lat):
    '''
    Calculate Julian day(nth day of the year), and solar zenith angle in degree.
    '''
    # the day of year
    t = datetime.datetime(year, month, day)
    N = t.timetuple().tm_yday

    # calculate solar zenith angle.
    # three needed angles
    phi = lat # latitude in deg 
    delta = -np.arcsin(0.39779 * np.cos(np.deg2rad(0.98565 * (N + 9) + 1.914 * np.sin(np.deg2rad(0.98565 * (N - 3))))))
    h = 15 * (hour - 12)     # hour angle in deg
    # deg 2 rad
    phi = np.deg2rad(phi)  # degree to radian
    h = np.deg2rad(h)
    # calculate 
    theta = np.rad2deg(np.arccos(np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(h))) # SZA in deg
    if theta > 87:  # will raise error for sza>87
        theta = 87

    ddd = N
    zenith_angle = theta
    #print("Solar Zenith Angle is: %s" % theta)
    return ddd, zenith_angle

######## aer file ########
# mid point of each of the 14 spextral bands
mdpn = np.array([2925,3625,4325,4900,5650,6925,7875,10450,14425,19325,25825,33500,44000,1710]) # unit: cm-1
mdpl = 1/mdpn*1e-2  # unit:m
def match_rh(faer,name_m2,name_opt,rd,rh,thick,dens):
    q = faer[name_m2].data
    ds = xr.open_dataset(dir2+'%s'%name_opt)  # not allowed to use lambda for index
    ds = ds.rename({'lambda': 'lamb'})
    if rd < 1:  # sulfate
        kext0 = ds['bext'].sel(radius=rd,method='nearest').interp(lamb=mdpl)  # unit: m2/kg
        ssa0 = (ds['qsca']/ds['qext']).sel(radius=rd,method='nearest').interp(lamb=mdpl)
        g0 = ds['g'].sel(radius=rd,method='nearest').interp(lamb=mdpl)
    else:  # easier to distinguish sulfate and others
        kext0 = ds['bext'][rd-1].interp(lamb=mdpl)  # at 1um
        ssa0 = (ds['qsca']/ds['qext'])[rd-1].interp(lamb=mdpl)
        g0 = ds['g'][rd-1].interp(lamb=mdpl)
    # find out the nearest rh
    rh0 = ds['rh']
    kext1 = np.empty((14,lenp-9))
    ssa1 = np.empty((14,lenp-9))
    g1 = np.empty((14,lenp-9))
    for i in range(lenp-9):
        rh1 = np.abs(rh0-rh[i])
        plc = np.where(rh1==rh1.min())[0][0]
        kext1[:,i] = kext0.data[plc]
        ssa1[:,i] = ssa0.data[plc]
        g1[:,i] = g0.data[plc]
    aod1 = q*dens*1e3 * thick * kext1*1e-3
    aod2 = np.nan_to_num(aod1)
    ssa2 = np.nan_to_num(ssa1)
    g2 = np.nan_to_num(g1)
    if aer_type == 0:   # all absorbing
        aod3 = aod2*(1-ssa2)
        ssa3 = np.zeros((ssa2.shape))
    elif aer_type == 1:  # all scattering
        aod3 = aod2*ssa2
        ssa3 = np.ones(ssa2.shape)
    return aod3, ssa3, g2
   
def get_opt(faer):
    dens = faer['AIRDENS'].data  # kg/m3
    delp = faer['DELP'].data  # pa  
    rh = np.nan_to_num(faer['RH'].data)
    # pressure (Pa), density of a gas (kg/m3), gravity acceleration (9.8 m/s2) --> height of a column of gas (m)
    thick = delp/(dens*9.8)
    
    # aerosol
    aod_du1,ssa_du1,g_du1 = match_rh(faer,'DU001','optics_DU.v15_3.nc',1,rh,thick,dens)
    aod_du2,ssa_du2,g_du2 = match_rh(faer,'DU002','optics_DU.v15_3.nc',2,rh,thick,dens)
    aod_du3,ssa_du3,g_du3 = match_rh(faer,'DU003','optics_DU.v15_3.nc',3,rh,thick,dens)
    aod_du4,ssa_du4,g_du4 = match_rh(faer,'DU004','optics_DU.v15_3.nc',4,rh,thick,dens)
    aod_du5,ssa_du5,g_du5 = match_rh(faer,'DU005','optics_DU.v15_3.nc',5,rh,thick,dens)
    aod_sa,ssa_sa,g_sa = match_rh(faer,'SO4', 'optics_SU.v1_3.nc', 0.16*1e-6, rh,thick,dens)
    aod_bco,ssa_bco,g_bco = match_rh(faer,'BCPHOBIC', 'optics_BC.v1_3.nc', 1, rh,thick,dens)
    aod_bci,ssa_bci,g_bci = match_rh(faer,'BCPHILIC', 'optics_BC.v1_3.nc', 2, rh,thick,dens)
    aod_oco,ssa_oco,g_oco = match_rh(faer,'OCPHOBIC', 'optics_OC.v1_3.nc', 1, rh,thick,dens)
    aod_oci,ssa_oci,g_oci = match_rh(faer,'OCPHILIC', 'optics_OC.v1_3.nc', 2, rh,thick,dens)

    # total
    aod = np.array([aod_du1,aod_du2,aod_du3,aod_du4,aod_du5,aod_bco,aod_bci,aod_oco,aod_oci,aod_sa])
    ssa = np.array([ssa_du1,ssa_du2,ssa_du3,ssa_du4,ssa_du5,ssa_bco,ssa_bci,ssa_oco,ssa_oci,ssa_sa])
    g = np.array([g_du1,g_du2,g_du3,g_du4,g_du5,g_bco,g_bci,g_oco,g_oci,g_sa])
    aodt = np.nan_to_num(aod.sum(axis=0))
    ssat = np.nan_to_num((ssa*aod).sum(axis=0)/aod.sum(axis=0))
    gt = np.nan_to_num((g*ssa*aod).sum(axis=0)/(ssa*aod).sum(axis=0))
    
    return aodt, ssat, gt

#### save to file ####    
def save_aer_file(aodt,ssat,gt,ptb):   # input ssa and g at each layer
    AOD0 = np.zeros((14,lenp-9))   
    # create AOD
    for pul in range(lenp-9):
        if pul == ptb:   # perturbed level
            if kernel_method == 0:    # ratio increase
                AOD0[:,pul] = aodt[:,pul]*dx
            elif kernel_method == 1:    # specific increase
                AOD0[:,pul] = aodt[:,pul] + dx[pul]
        else:
            AOD0[:,pul] = aodt[:,pul]
        
    aid = open('IN_AER_RRTM','w')

    ### 1 ###
    # number of aerosol species
    line = '   %s\n'%(lenp-9)
    aid.write(line)
    ### 2 ###
    # number of layer, way to input AOD, SSA and g
    num = np.arange(10,lenp+1)
    for l in range(28):
        NLAY = 1
        IAOD = 1
        line = '   %2d    %d    %d    %d\n'%(NLAY,IAOD,IAOD,IAOD)
        aid.write(line)
        ### 3 ###
        # AOD at each layer 
        line = '  %3d'%num[l]
        for n in range(14):
            line += '%7.5f'%AOD0[n,l]     
            if n == 13:
                line += '\n'
        aid.write(line)
        ### 4 ###
        # SSA
        line = ''
        if AOD0[9,l] < 0.00005:
            for n in range(14):
                line += '%5.2f'%1     
                if n == 13:
                    line += '\n'
        else:
            for n in range(14):
                if ssat[n,l]>1:
                    line += '%5.2f'%1
                else:
                    line += '%5.2f'%ssat[n,l]   
                if n == 13:
                    line += '\n'
        aid.write(line)
        ### 5 ###
        # phase
        line = ''
        if AOD0[9,l] < 0.00005:
            for n in range(14):
                line += '%5.2f'%0     
                if n == 13:
                    line += '\n'
        else:
            for n in range(14):
                line += '%5.2f'%gt[n,l]     
                if n == 13:
                    line += '\n'
        aid.write(line)
    aid.close()
    return AOD0[:,::-1]    # from top to bottom
    
def save_cld_file(aodt,ssat,gt,ptb):   # input ssa and g at each layer
    COD0 = np.zeros((14,lenp))   
    # create AOD
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
    line = '   %s\n'%(lenp)
    aid.write(line)
    ### 2 ###
    # number of layer, way to input AOD, SSA and g
    num = np.arange(1,lenp+1)
    for l in range(lenp):
        NLAY = 1
        IAOD = 1
        line = '   %2d    %d    %d    %d\n'%(NLAY,IAOD,IAOD,IAOD)
        aid.write(line)
        ### 3 ###
        # AOD at each layer 
        line = '  %3d'%num[l]
        for n in range(14):
            line += '%7.5f'%COD0[n,l]     
            if n == 13:
                line += '\n'
        aid.write(line)
        ### 4 ###
        # SSA
        line = ''
        if COD0[9,l] < 0.00005:
            for n in range(14):
                line += '%5.2f'%1     
                if n == 13:
                    line += '\n'
        else:
            for n in range(14):
                if ssat[n,l]>1:
                    line += '%5.2f'%1
                else:
                    line += '%5.2f'%ssat[n,l]   
                if n == 13:
                    line += '\n'
        aid.write(line)
        ### 5 ###
        # phase
        line = ''
        if COD0[9,l] < 0.00005:
            for n in range(14):
                line += '%5.2f'%0     
                if n == 13:
                    line += '\n'
        else:
            for n in range(14):
                line += '%5.2f'%gt[n,l]     
                if n == 13:
                    line += '\n'
        aid.write(line)
    aid.close()
    return COD0[:,::-1]
    
######## cloud file ########
# shortwave band limt from FU 1996
ffu = xlx.load_workbook('Fu-sw-cld.xlsx')  
st_f = ffu['Sheet1']
row1=[item.value for item in list(st_f.columns)[2]]  
wave_fu = np.array([float(i) for i in row1[1:26]])*1e-6  # convert unit from um to m
row2=[item.value for item in list(st_f.columns)[4]]  
a0 = np.array([float(i) for i in row2[1:26]])
row3=[item.value for item in list(st_f.columns)[5]]  
a1 = np.array([float(i) for i in row3[1:26]])
row4=[item.value for item in list(st_f.columns)[7]]  
b0 = np.array([float(i) for i in row4[1:26]])
row5=[item.value for item in list(st_f.columns)[8]]  
b1 = np.array([float(i) for i in row5[1:26]])
row6=[item.value for item in list(st_f.columns)[9]]  
b2 = np.array([float(i) for i in row6[1:26]])
row7=[item.value for item in list(st_f.columns)[10]]  
b3 = np.array([float(i) for i in row7[1:26]])
row8=[item.value for item in list(st_f.columns)[12]]  
c0 = np.array([float(i) for i in row8[1:26]])
row9=[item.value for item in list(st_f.columns)[13]]  
c1 = np.array([float(i) for i in row9[1:26]])
row10=[item.value for item in list(st_f.columns)[14]]  
c2 = np.array([float(i) for i in row10[1:26]])
row11=[item.value for item in list(st_f.columns)[15]]  
c3 = np.array([float(i) for i in row11[1:26]])

extc0 = a0+a1/D
ssa0 = 1-(b0+b1*D+b2*(D**2)+b3*(D**3))
g0 = c0+c1*D+c2*(D**2)+c3*(D**3)

rlowb = 1/np.array([2600,3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,820])*1e-2
rhighb = 1/np.array([3250,4000,4650,5150,6150,7700,8050,12850,16000,22650,29000,38000,50000,2600])*1e-2

def get_cld_opt(fcld,fair):
    dens = fair['AIRDENS'].data  # kg/m3
    delp = fair['DELP'].data  # pa 
    # pressure (Pa), density of a gas (kg/m3), gravity acceleration (9.8 m/s2) --> height of a column of gas (m)
    thick = delp/(dens*9.8)
    iwc = fcld['IWC'].data*1e-3  # mg/m3 to g/m3
    extc1 = iwc*extc0[:,np.newaxis]  #(fu band, lenp)
    extc = np.zeros((14,lenp))
    ssa = np.zeros(14)
    g = np.zeros(14)
    for k in range(14):
        band = (wave_fu > rhighb[k]) & (wave_fu <= rlowb[k])
        extc[k] = extc1[band].mean(axis=0)
        ssa[k] = ssa0[band].mean()
        g[k] = g0[band].mean()
    cod = np.nan_to_num(thick*extc)
    cod[cod<0] = 0
    ssat = np.nan_to_num(np.ones((14,lenp))*ssa[:,np.newaxis])
    ssat[ssat<0] = 0
    gt =np.nan_to_num(np.ones((14,lenp))*g[:,np.newaxis])
    gt[gt<0] = 0
    return cod,ssat,gt   # from top to bottom
 
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
    return p, t, top_p, top_t, o3, h2o, col_dens

co2 = 525*29/44*1e-6*np.ones(lenp)   # constant co2 for RCP4.5

# those don't include in MERRA-2 should be get from rrtmg
# use xlsx file
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
    if band == 98:
        out = np.zeros((15,8,lenp+1))
        for i in range(15):
            for j in range(lenp+7):  # in one band range
                line = oid.readline()
                if j < 5 or j == lenp+6:  # first 5 line or final line
                    pass
                else:
                    out0 = line.strip().split(' ')
                    out1 = list(filter(None,out0))
                    for p in range(8):
                        try:
                            out[i,p,j-5] = float(out1[p])
                        except(ValueError):
                            out[i,p,j-5] = 999
                    #out[i,:,j-5] = [float(n) for n in list(filter(None,out0))]
    else:    
        out = np.zeros((8,lenp+1))
        for j in range(lenp+7):  # in one band range
            line = oid.readline()
            if j < 5 or j == lenp+6:  # first 5 line or final line
                pass
            else:
                out0 = line.strip().split(' ')
                out1 = list(filter(None,out0))
                for p in range(8):
                    try:
                        out[p,j-5] = float(out1[p])
                    except(ValueError):
                        out[p,j-5] = 999
                #out[:,j-5] = [float(n) for n in list(filter(None,out0))]
    oid.close()
    out1 = np.nan_to_num(out)   # convert nan to 0
    return out1


#### loop over all grid and all time ####
def run_rrtm(ti,i,th):
    time = str(time7[ti].data)
    lat = latt[i]
    print(lat)
    # juldat and zenith angle
    year = int(time[0:4])
    month = int(time[5:7])
    day = 15
    hour = 3*th   # local time
    juldat,zenith_angle =  rrtm_sw_tinfo(year, month, day, hour, lat)
    print(time,lat,hour)
    
    faer = dsaer.sel(lat=lat).isel(time=ti)
    fatm = dsatm.sel(lat=lat).isel(time=ti)
    frag = dsrag.sel(lat=lat).isel(time=ti)
    fcld = dscld.sel(lat=lat).isel(time=ti)
    fts  = dsts.sel(lat=lat).isel(time=ti*8+th)
    fair = dsair.sel(lat=lat).isel(time=ti%12)   # only one year
    inso = inso0[ti%12,th,i]
    
    # atm data
    p,t,top_p,top_t,o3,h2o,col_dens = get_atm_data(fatm,frag,fts)  
    #co2 = interp(1,p)
    o2 = interp(5,p)
    n2o = interp(2,p)
    ch4 = interp(4,p)
    co = interp(3,p)

    # emissivity
    alb = fts['fal'].data
    emis = 1-alb
    
        #os.system('rm IN_AER_RRTM')   

    print('finish preparing file')
    
    ### create input data ###
    fid = open('./SW_TEMPLATE', 'r')
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
    # set Julian days, and solar zenith angle in degree.
    line = fid.readline()
    line = line.replace('DDD', '%3d' % juldat)
    line = line.replace('ZENITHA', '%7.4f' % zenith_angle)
    line = line.replace('INSOLATION','%10.4f' % inso)
    wid.write(line)
    ### 4 ###
    # set emissivity.
    line = fid.readline()
    line = line.replace('EMISS', '%5.3f'%emis)
    wid.write(line)
    ### 5 ###
    # Set number format, number of layers, number of molecules
    line = fid.readline()
    line = line.replace('NLY', '%3d' % (lenp))
    wid.write(line)
    ### 6 ###
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
    ### 7 ####
    wid.write('%%%%%')
    ### close files ###
    fid.close()
    wid.close()
    
    # aer & cloud data
    if IAER == 10:
        if aer_type == 3:   # cloud
            odt,ssat,gt = get_cld_opt(fcld,fair)
            ssa[ti*8+th,i,1:,1:] = ssat[:,::-1]  # save from toa to surface
            g[ti*8+th,i,1:,1:] = gt[:,::-1]
            for ptb in range(17):   # perturb from surface to top
                od[ti*8+th,i,ptb,1:,1:] = save_cld_file(odt, ssat, gt, ptb)
                ######## run model ########
                os.system('./rrtmg_sw_v14.0.2_linux_intel')
                #os.system('./rrtmg_sw_v19.1.3.304_linux_intel')
                #os.system('./rrtmg_sw_v19.1.1.217_linux_intel')
                print('finish running')
                output[ti*8+th,i,ptb] = get_output(IOUT)
        else:
            odt,ssat,gt = get_opt(faer)
            ssa[ti*8+th,i,1:,1:lenp-8] = ssat[:,::-1]
            g[ti*8+th,i,1:,1:lenp-8] = gt[:,::-1]
            for ptb in range(17):
                od[ti*8+th,i,ptb,1:,1:lenp-8] = save_aer_file(odt, ssat, gt, ptb)
                ######## run model ########
                os.system('./rrtmg_sw_v14.0.2_linux_intel')
                #os.system('./rrtmg_sw_v19.1.3.304_linux_intel')
                #os.system('./rrtmg_sw_v19.1.1.217_linux_intel')
                print('finish running')
                output[ti*8+th,i,ptb] = get_output(IOUT)
    else:
        pass

######## set input parameters ########
dir1 = '/home/gj/data3/'
atm_input = dir1+'era5/era5_ztcld_2011-2020_monthly_zm.mlslat.nc'
rag_input = dir1+'mls/merge/MLS-gas_alllev_mon_2011-2020.nc'
aer_input = dir1+'monthly/aer_np/m2aer.uts.2011-2020.zm.mlslat.nc'
cld_input = dir1+'mls/merge/MLS-iwc_alllev_mon_2011-2020.nc'
ts_input = dir1+'era5/era5_albts_2011-2020_monthly_3h_zm.mlslat.nc'
air_input = dir1+'monthly/air/m2_air_np_mon_2020.zm.mlslat.nc'

# interpolate to get same time
dsrag = xr.open_dataset(rag_input).sel(lat=slice(-30,30),lev=slice(1000,1))
lev0 = dsrag['lev'].data  # from surface to toa
latt = dsrag['lat'].data
time7 = dsrag['time']
dsatm = xr.open_dataset(atm_input).interp(level=lev0).sel(lat=slice(-30,30)).squeeze()
dsaer = xr.open_dataset(aer_input).interp(plev=lev0[9:37]*100,lat=latt).squeeze()
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

# insolation
dsin = np.load(dir1+'ceres/insolation_monthly_3h_2011-2020.mlslat.npz')
inso0 = dsin['inso'][:,:,15:30]   # month, hour, lat(-28~28)

# get specific od increase
dsod = np.load('mean_vert_od550_2011-2020.npz')

for rr in range(2,3):
    # ratio increase
    if kernel_method == 0:   
        if rr == 0:
            dx=1.1
            aer_type = 0    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_eachlev_aaod10.nc'
        elif rr == 1:
            dx=1.1
            aer_type = 1    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_eachlev_saod10.nc'
        elif rr == 2:
            dx=1.1
            aer_type = 3    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_eachlev_cod10.nc'
    # specific increase
    elif kernel_method == 1:   
        if rr == 0:
            dx = np.ones(lenp)*1e-5  # too small that can't use 10% mean
            aer_type = 0    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_add_eachlev_aaod10.nc'
        elif rr == 1:
            # set saod<1e-4 to 1e-4
            saod = dsod['saod']
            x = np.where(saod<1e-4)[0]
            saod[x] = 1e-4
            dx = saod[::-1]*0.1
            aer_type = 1    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_add_eachlev_saod10.nc'
        elif rr == 2:
            dx=dsod['cod'][::-1]*0.1
            aer_type = 3    # 0 for all absorbing, 1 for all scattering
            IAER = 10
            output_file = dir3+'rrtm_sw_2011-2020_monthly_zm_alllev_add_eachlev_cod10.nc'   
    
    timlen = len(time7.data)
    latlen = len(latt.data)
    
    od = np.zeros((timlen*8,latlen,17,15,lenp+1))
    ssa = np.zeros((timlen*8,latlen,15,lenp+1))
    g = np.zeros((timlen*8,latlen,15,lenp+1))
    if IOUT == 98:
        output = np.zeros((timlen*8,latlen,17,15,8,lenp+1))
    else:
        output = np.zeros((timlen*8,latlen,17,8,lenp+1))
    
    time0 = []
    for ti in range(timlen):
        for th in range(8):  # 8 time step per day
            time0.append('Y%4dM%02dT%02d'%(2011+ti//12,ti%12+1,th*3))
            for i in range(latlen):
                run_rrtm(ti,i,th)
                
    time = xr.DataArray(time0,coords=[np.arange(timlen*8)],dims=['time'])
    
    if IOUT == 98:  # all band range
        output1 = output.transpose((4,3,0,2,5,1))
        od1 = od.transpose((3,0,2,4,1))
        ssa1 = ssa.transpose((2,0,3,1))
        g1 = g.transpose((2,0,3,1))
        bandnum = np.concatenate(([0],np.arange(16,30)))
        band = xr.DataArray(bandnum,coords=[bandnum],dims=['band'])   
        ds = xr.Dataset({'p':(['time','lev'],output1[1,0,:,0,:,0]),
                         'up_flx':(['band','time','perturbed_lev','lev','lat'],output1[2]),
                         'difdn_flx':(['band','time','perturbed_lev','lev','lat'],output1[3]),
                         'dirdn_flx':(['band','time','perturbed_lev','lev','lat'],output1[4]),
                         'dn_flx':(['band','time','perturbed_lev','lev','lat'],output1[5]),
                         'net_flx':(['band','time','perturbed_lev','lev','lat'],output1[6]),
                         'heat_rate':(['band','time','perturbed_lev','lev','lat'],output1[7]),
                         'od':(['band','time','perturbed_lev','lev','lat'],od1),
                         'ssa':(['band','time','perturbed_lev','lev','lat'],ssa1),
                         'g':(['band','time','perturbed_lev','lev','lat'],g1)},
                            coords={'lev':lev,'band':band,'time':time,'lat':latt,'perturbed_lev':ptb_lev})
        
    else:
        output1 = output.transpose((3,0,2,4,1))
        od1 = od.transpose((3,0,2,4,1))[10]  # 550nm
        ssa1 = ssa.transpose((2,0,3,1))[10]
        g1 = g.transpose((2,0,3,1))[10]
        ds = xr.Dataset({'p':(['time','lev'],output1[1,:,0,:,0]),
                         'up_flx':(['time','perturbed_lev','lev','lat'],output1[2]),
                         'difdn_flx':(['time','perturbed_lev','lev','lat'],output1[3]),
                         'dirdn_flx':(['time','perturbed_lev','lev','lat'],output1[4]),
                         'dn_flx':(['time','perturbed_lev','lev','lat'],output1[5]),
                         'net_flx':(['time','perturbed_lev','lev','lat'],output1[6]),
                         'heat_rate':(['time','perturbed_lev','lev','lat'],output1[7]),
                         'od550':(['time','perturbed_lev','lev','lat'],od1),
                         'ssa550':(['time','lev','lat'],ssa1),
                         'g550':(['time','lev','lat'],g1)},
                            coords={'lev':lev,'time':time,'lat':latt,'perturbed_lev':ptb_lev})
        
    ds.to_netcdf(output_file)
    print('finish saving')