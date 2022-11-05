import metpy
import datetime
import rioxarray
import os
import glob
import warnings

import geopandas as gpd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.feature as cfeature
import matplotlib.colors as colors
import xesmf as xe

from tqdm.notebook import tqdm
from shapely.geometry import mapping
from dask import delayed
from dask.distributed import Client
from pyorbital.orbital import get_observer_look

xr.set_options(keep_attrs=True)

def open_hour(path):
    keep_noae = ['x', 'y', 't', 'AOD']
    keep_ae = keep_noae + ['AE1']
    drop_these = [
        'y_image','x_image','sunglint_angle','retrieval_local_zenith_angle',
        'quantitative_local_zenith_angle','retrieval_solar_zenith_angle','quantitative_solar_zenith_angle',
        'aod_product_wavelength']
    #        'latitude_bands',
    #    'aod550_retrievals_attempted_land','aod550_good_LZA_retrievals_attempted','aod550_outlier_pixel_count',
    #    'min_aod550_land','max_aod550_land','mean_aod550_land','std_dev_aod550_land',
    # ]

    bad_files = []
    good_datasets = []
    gip = None

    for file in sorted(listdir(path)):
        file_path = os.path.join(path, file)
        try:
            with xr.open_dataset(file_path) as ds:
                ds.load()
        except Exception:
            bad_files.append(file_path)
        else:
            if pd.to_datetime(ds.t.item()).year == 2000:
                bad_files.append(file_path)
            else:
                ds = ds.drop_vars(drop_these)
                if gip is None:
                    gip = ds.goes_imager_projection
                
                if ds.DQF.attrs['flag_values'].size == 4:
                    ds.AOD.data = xr.where(ds.DQF < 2, ds.AOD, np.nan).data
                else:
                    # early data only had two data quality flags
                    ds.AOD.data = xr.where(ds.DQF == 0, ds.AOD, np.nan).data
                ds = ds.drop_vars(['DQF'])
                
                if 'AE1' in ds:
                    ds['AE1'].data = xr.where(ds.AE_DQF < 2, ds.AE1, np.nan).data
                    ds = ds[keep_ae]
                else:
                    ds = ds[keep_noae]
                    
                good_datasets.append(ds)
    ds = xr.concat(good_datasets, 't')
    ds['goes_imager_projection'] = gip
    return convert_to_geostationary(ds, 'AOD'), bad_files

def clip_dataset(ds, proj, north_am):
    a = ds.copy()
    a.rio.write_crs(proj.to_string(), inplace=True)
    na = north_am.copy().to_crs(a.rio.crs)
    return a.rio.clip(na.geometry.apply(mapping))

def add_lat_lon(ds, proj):
    xs, ys = np.meshgrid(ds.x, ds.y)
    transformed = ccrs.PlateCarree().transform_points(src_crs=proj, x=xs, y=ys)
    ds = ds.assign_coords({
            "lat":(["y","x"],transformed[:,:,1]),
            "lon":(["y","x"],transformed[:,:,0])
        })
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    
    return ds

def regrid(ds1, ds2):
    """ Regrid ds2 to match the coordinates of ds1

    """
    lon1 = ds1.goes_imager_projection.longitude_of_projection_origin
    lon2 = ds2.goes_imager_projection.longitude_of_projection_origin
    regrid_path = os.path.join('data', f'regrid_{lon1}_{lon2}.nc')
    if os.path.exists(regrid_path):
        regridder = xe.Regridder(ds2, ds1, "bilinear", unmapped_to_nan=True, weights=regrid_path)
    else:
        regridder = xe.Regridder(ds2, ds1, "bilinear", unmapped_to_nan=True)
        regridder.to_netcdf(regrid_path)
    return regridder(ds2)
        
def combine_compute_aod470_average(ds1, ds2):
    ds = xr.concat([i for i in [ds1, ds2] if i],
       dim='t').sortby('t')
    
    ds['n_snapshots'] = xr.full_like(ds.AOD, fill_value = np.nan, dtype=np.float32)
    ds['n_snapshots'].data = xr.where(ds.AOD.notnull(), 1, np.nan).data
    
    if 'AE1' in ds:
        ds['AOD470'] = ds.AOD * (470 / 550)**(-ds.AE1)
        ds = ds.drop_vars(['AE1'])
        ds.AOD470.attrs = {'long_name': 'derived Aerosol Optical Depth at 470 nm from the 5-minute AE at the 0.47 / 0.86 micron wavelength pair',
         'units': '1',
         'grid_mapping': 'goes_imager_projection'
        }
    
    av = ds.resample(t='1h').mean()
    av['n_snapshots'] = ds['n_snapshots'].resample(t='1h').sum(skipna=True)
    av['n_snapshots'].data = xr.where(av.n_snapshots != 0, av.n_snapshots, np.nan).data
    av['n_snapshots'] = av['n_snapshots'].astype(np.float32)
    av.n_snapshots.attrs = {'long_name': 'Number of 5-minute files used to average per cell', 'grid_mapping': 'goes_imager_projection'}
    
    return av

def listdir(path):
    for item in os.listdir(path):
        if item.startswith('.'):
            # ignore hidden files
            continue
        else:
            yield item
            
def get_projection(ds, variable):
    dat = ds.metpy.parse_cf(variable)
    return dat, dat.metpy.cartopy_crs
    
def convert_to_geostationary(ds, variable):
    dat = ds.metpy.parse_cf(variable)
    proj = dat.metpy.cartopy_crs
    
    x = dat.x
    y = dat.y
    
    sat_h = ds.goes_imager_projection.perspective_point_height
    ds = ds.assign_coords(x=ds.x * sat_h, y=ds.y * sat_h)
    
    ds.x.attrs = {'units': 'm',
     'axis': 'X',
     'long_name': 'X sweep in crs',
    }
    ds.y.attrs = {'units': 'm',
     'axis': 'Y',
     'long_name': 'Y sweep in crs',
    }

    return ds, proj

def average(date, 
            goes16_path = '/Volumes/Minuet/research/GOES-16/AODC',
            goes17_path = '/Volumes/Minuet/research/GOES-17/AODC',
            save_path = '/Volumes/Canon/averages/hourly',
            north_am = None
           ):
    year = f'{date.year}'
    doy = f'{date.day_of_year:03}'
    hour = f'{date.hour:02}'
    
    save_path = os.path.join(save_path, year, doy)
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(os.path.join(save_path, f'{hour}.nc')):
        return
    
    ds1, ds2 = None, None
    bad_filesg16, bad_filesg17 = [], []
    files_that_need_coordinate_remapping = []
    
    if north_am is None:
        regions = gpd.read_file('data/Natural_Earth_quick_start/50m_cultural/ne_50m_admin_0_countries.shp')
        north_am = regions[regions.CONTINENT == 'North America']
    
    g16_path = os.path.join(goes16_path, year, doy, hour)
    if os.path.exists(g16_path):
        (ds1, proj_g16), bad_filesg16 = open_hour(g16_path)
        ds1 = add_lat_lon(
            clip_dataset(ds1, proj_g16, north_am), proj_g16)
        
    g17_path = os.path.join(goes17_path, year, doy, hour)
    if os.path.exists(g17_path):
        (ds2, proj_g17), bad_filesg17 = open_hour(g17_path)
        ds2 = add_lat_lon(
            clip_dataset(ds2, proj_g17, north_am), proj_g17)
        if ds2.goes_imager_projection.longitude_of_projection_origin == -137.0:
            ds2 = ds2.isel(x=slice(1500,2500))

    if ds1 is None and ds2 is None:
        return None
    
    if (ds1 is not None) and (ds2 is not None):
        ds2 = regrid(ds1, ds2)
    elif ds2 is not None:
        # GOES-16 didn't have a file this day, so the output format is in the GOES-16 grid
        # I want the all final datasets to be in the goes16 grid, so after all files have been processed
        # these files will need to be reopened and regridded to goes16
        files_that_need_coordinate_remapping.append(g17_path)
        
    ds = combine_compute_aod470_average(ds1, ds2)
    
    ds.drop_vars(['lat', 'lon']).to_netcdf(os.path.join(save_path, f'{hour}.nc'))
    
    import pickle
    
    with open(os.path.join(save_path, f'{hour}-errors.pcl'), 'wb') as f:
        data = {
            'bad_files': bad_filesg16 + bad_filesg17,
            'coord_remap_needed': files_that_need_coordinate_remapping,
        }
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

import sys

date = pd.to_datetime(sys.argv[1])

regions = gpd.read_file('data/Natural_Earth_quick_start/50m_cultural/ne_50m_admin_0_countries.shp')
north_am = regions[regions.CONTINENT == 'North America']

average(date, north_am = north_am)

