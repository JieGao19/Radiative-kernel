# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:10:37 2023

@author: 19855
"""

import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

dir1 = '/home/gj/data3/monthly/aer_np/'
dir2 = '/home/gj/data3/monthly/result/lw/'
dir3 = '/home/gj/data3/monthly/result/sw/'
trop_alb = np.array([0.1,0.25,0.45,0.65])

###### calculated SW kernel #########
lev = np.array([ 0.89423615,   1.10576391,   1.33966351,   1.62303936,
         1.96635699,   2.38229609,   2.88621759,   3.49673223,
         4.23638773,   5.13250065,   6.21816683,   7.53348112,
         9.12702084,  11.05763817,  13.39663506,  16.23039436,
        19.6635704 ,  23.8229599 ,  28.86217499,  34.9673233 ,
        42.36387634,  51.32501221,  62.18166733,  75.33480835,
        91.27021027, 110.5763855 , 133.96633911, 162.30392456,
       200.        ])
# aod per 100 hPa
dsaer = xr.open_dataset(dir1+'m2aer.uts.2011-2020avg.zm.mlslat.nc').interp(plev=lev*100).sel(lat=slice(-30,30)).squeeze()
delp0 = dsaer['DELP'].data
delp = np.zeros((96,len(lev),15))   # time, lev, lat
for t in range(12):
    delp[t*8:(t+1)*8] = np.array([delp0[t],delp0[t],delp0[t],delp0[t],delp0[t],delp0[t],delp0[t],delp0[t]])
    
# get dx
# set saod<1e-4 to 1e-4
fod = np.load(dir3+'../../plot/sw/od550_per100hpa_2011-2020.zm.npz')
dsaod = fod['saod'].mean(axis=(0,1,3))[::-1]  # only vertical left, top to surface
x = np.where(dsaod<1e-4)[0]
dsaod[x] = 1e-4
dcod = fod['cod'].mean(axis=(0,1,3))[::-1]

neta_t = np.zeros((4,96,29,15))
hra_t = np.zeros((4,96,29,15))
nets_t = np.zeros((4,96,29,15))
hrs_t = np.zeros((4,96,29,15))
netc_t = np.zeros((4,96,29,15))
hrc_t = np.zeros((4,96,29,15))
neta_ker_t = np.zeros((4,96,17,29,15))
hra_ker_t = np.zeros((4,96,17,29,15))
nets_ker_t = np.zeros((4,96,17,29,15))
hrs_ker_t = np.zeros((4,96,17,29,15))
netc_ker_t = np.zeros((4,96,17,29,15))
hrc_ker_t = np.zeros((4,96,17,29,15))

for i in range(4):
    f00 = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_band_cln.nc'%trop_alb[i])   # (time, lev, lat)
    net00 = f00['net_flx'][0]
    hr00 = f00['heat_rate'][0]
    
    ### AAOD kernel ###
    faaod0 = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_band_aaod.nc'%trop_alb[i])
    # (time, lev, lat)
    neta0 = faaod0['net_flx'][0]
    aaod = faaod0['od'][10].data
    hra0 = faaod0['heat_rate'][0]
    neta_t[i] = neta0 - net00
    hra_t[i] = hra0 - hr00
    
    faaod = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_add_eachlev_band_aaod10.nc'%trop_alb[i])
    # (band, time, ptb, lev)
    neta = faaod['net_flx'][0]
    hra = faaod['heat_rate'][0]
    # net flux
    neta_ker_t[i] = (neta - neta0)/(1e-5/delp[:,np.newaxis,:,:]*10000)
    # heating rate
    hra_ker_t[i] = (hra - hra0)/(1e-5/delp[:,np.newaxis,:,:]*10000) 
    
    ### SAOD kernel ###
    fsaod0 = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_band_saod.nc'%trop_alb[i])
    # (band, time, lev)
    nets0 = fsaod0['net_flx'][0]
    saod = fsaod0['od'][10].data
    hrs0 = fsaod0['heat_rate'][0]
    nets_t[i] = nets0 - net00
    hrs_t[i] = hrs0 - hr00
    
    fsaod = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_add_eachlev_band_saod10.nc'%trop_alb[i])
    # (band, time, lev)
    nets = fsaod['net_flx'][0]
    hrs = fsaod['heat_rate'][0]
    # net flux
    nets_ker_t[i] = (nets - nets0)/(0.1*dsaod[np.newaxis,:17,np.newaxis,np.newaxis]/delp[:,np.newaxis,:,:]*10000)
    # heating rate
    hrs_ker_t[i] = (hrs - hrs0)/(0.1*dsaod[np.newaxis,:17,np.newaxis,np.newaxis]/delp[:,np.newaxis,:,:]*10000)
    
    ### COD kernel ###
    fcod0 = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_band_cod.nc'%trop_alb[i])
    # (band, time, lev)
    netc0 = fcod0['net_flx'][0]
    cod = fcod0['od'][10].data
    hrc0 = fcod0['heat_rate'][0]
    netc_t[i] = netc0 - net00
    hrc_t[i] = hrc0 - hr00
    
    fcod = xr.open_dataset(dir3+'rrtm_sw_11-20_monthly_zm_200srf_alb%s_add_eachlev_band_cod10.nc'%trop_alb[i])
    # (band, time, lev)
    netc = fcod['net_flx'][0]
    hrc = fcod['heat_rate'][0]
    # net flux
    netc_ker_t[i] = (netc - netc0)/(0.1*dcod[np.newaxis,:17,np.newaxis,np.newaxis]/delp[:,np.newaxis,:,:]*10000)
    # heating rate
    hrc_ker_t[i] = (hrc - hrc0)/(0.1*dcod[np.newaxis,:17,np.newaxis,np.newaxis]/delp[:,np.newaxis,:,:]*10000)

lev = f00['lev']
print(lev)
lat = f00['lat']
ptb = faaod['perturbed_lev']
time = f00['time']
alb = xr.DataArray(trop_alb,coords=[trop_alb],dims=['alb'])

    
ds = xr.Dataset({'aaod550':(['time','lev','lat'],aaod/delp*10000),
                 'saod550':(['time','lev','lat'],saod/delp*10000),
                 'cod550':(['time','lev','lat'],cod/delp*10000),
                 'aaod_net_flx':(['alb','time','lev','lat'],neta_t),
                 'aaod_heat_rate':(['alb','time','lev','lat'],hra_t),
                 'saod_net_flx':(['alb','time','lev','lat'],nets_t),
                 'saod_heat_rate':(['alb','time','lev','lat'],hrs_t),
                 'cod_net_flx':(['alb','time','lev','lat'],netc_t),
                 'cod_heat_rate':(['alb','time','lev','lat'],hrc_t),
                 'aaod_net_ker':(['alb','time','perturbed_lev','lev','lat'],neta_ker_t),
                 'aaod_rhr_ker':(['alb','time','perturbed_lev','lev','lat'],hra_ker_t),
                 'saod_net_ker':(['alb','time','perturbed_lev','lev','lat'],nets_ker_t),
                 'saod_rhr_ker':(['alb','time','perturbed_lev','lev','lat'],hrs_ker_t),
                 'cod_net_ker':(['alb','time','perturbed_lev','lev','lat'],netc_ker_t),
                 'cod_rhr_ker':(['alb','time','perturbed_lev','lev','lat'],hrc_ker_t)},
                    coords={'lev':lev,'time':time,'lat':lat,'perturbed_lev':ptb,'alb':alb})
ds.to_netcdf('sw_strat_4alb_kernel.nc')
print('finish sw')

###### calculated LW kernel #########
emist = [269.7,267.4,243.7,222.9]
# get dx
# set saod<1e-4 to 1e-4
fod = np.load(dir3+'../../plot/lw/od10_per100hpa_2011-2020.zm.npz')
dcod = fod['cod'].mean(axis=(0,1,3))[::-1]  # only vertical left, top to surface

neta_t = np.zeros((4,12,29,15))
hra_t = np.zeros((4,12,29,15))
netc_t = np.zeros((4,12,29,15))
hrc_t = np.zeros((4,12,29,15))
neta_ker_t = np.zeros((4,12,17,29,15))
hra_ker_t = np.zeros((4,12,17,29,15))
netc_ker_t = np.zeros((4,12,17,29,15))
hrc_ker_t = np.zeros((4,12,17,29,15))

for i in range(4):
    f00 = xr.open_dataset(dir2+'rrtm_lw_11-20_monthly_zm_200srf_t%s_cln.nc'%emist[i])   # (time, lev, lat)
    net00 = f00['net_flx']
    hr00 = f00['heat_rate']
    
    ### AOD kernel ###
    f0 = xr.open_dataset(dir2+'rrtm_lw_11-20_monthly_zm_200srf_t%s_band_aod.nc'%emist[i])
    # (time, lev, lat)
    neta0 = f0['net_flx'][0]
    aod = f0['od'][7].data
    hra0 = f0['heat_rate'][0]
    neta_t[i] = neta0 - net00
    hra_t[i] = hra0 - hr00
    
    f1 = xr.open_dataset(dir2+'rrtm_lw_11-20_monthly_zm_200srf_add_eachlev_t%s_band_aod10.nc'%emist[i])
    # (time, perturbed_lev, lev, lat)
    neta = f1['net_flx'][0]
    hra = f1['heat_rate'][0]
    # net flux
    neta_ker_t[i] = (neta - neta0)/(1e-5/delp0[:,np.newaxis,:,:]*10000)
    # heating rate
    hra_ker_t[i] = (hra - hra0)/(1e-5/delp0[:,np.newaxis,:,:]*10000)
    
    ### COD kernel ###
    fcod0 = xr.open_dataset(dir2+'rrtm_lw_11-20_monthly_zm_200srf_t%s_band_cod.nc'%emist[i])
    # (band, lev, lat)
    netc0 = fcod0['net_flx'][0]
    cod = fcod0['od'][7].data
    hrc0 = fcod0['heat_rate'][0]
    netc_t[i] = netc0 - net00
    hrc_t[i] = hrc0 - hr00
    
    fcod = xr.open_dataset(dir2+'rrtm_lw_11-20_monthly_zm_200srf_add_eachlev_t%s_band_cod10.nc'%emist[i])
    # (band, lev, lat)
    netc = fcod['net_flx'][0]
    hrc = fcod['heat_rate'][0]
    
    # net flux
    netc_ker_t[i] = (netc - netc0)/(0.1*dcod[np.newaxis,:17,np.newaxis,np.newaxis]/delp0[:,np.newaxis,:,:]*10000)
    # heating rate
    hrc_ker_t[i] = (hrc - hrc0)/(0.1*dcod[np.newaxis,:17,np.newaxis,np.newaxis]/delp0[:,np.newaxis,:,:]*10000)
    
timel = f00['time']
emisT = xr.DataArray(emist,coords=[emist],dims=['emisT'])
ds = xr.Dataset({'aod10':(['time','lev','lat'],aod/delp0*10000),
                 'cod10':(['time','lev','lat'],cod/delp0*10000),
                 'aod_net_flx':(['emisT','time','lev','lat'],neta_t),
                 'aod_heat_rate':(['emisT','time','lev','lat'],hra_t),
                 'cod_net_flx':(['emisT','time','lev','lat'],netc_t),
                 'cod_heat_rate':(['emisT','time','lev','lat'],hrc_t),
                 'aod_net_ker':(['emisT','time','perturbed_lev','lev','lat'],neta_ker_t),
                 'aod_rhr_ker':(['emisT','time','perturbed_lev','lev','lat'],hra_ker_t),
                 'cod_net_ker':(['emisT','time','perturbed_lev','lev','lat'],netc_ker_t),
                 'cod_rhr_ker':(['emisT','time','perturbed_lev','lev','lat'],hrc_ker_t)},
                    coords={'lev':lev,'time':timel,'lat':lat,'perturbed_lev':ptb,'emisT':emisT})
ds.to_netcdf('lw_strat_4emist_kernel.nc')