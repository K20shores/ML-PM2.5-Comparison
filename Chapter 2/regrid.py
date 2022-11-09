import xarray as xr
import numpy as np
import xesmf as xe
import metpy
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plate = ccrs.PlateCarree()

def get_regridder(ds_from, ds_to, var_type, i, j):
    lon = ds.goes_imager_projection.longitude_of_projection_origin
    xsz = ds.x.size
    ysz = ds.y.size
    
    regrid_name = f'data/regridders/regird_{var_type}_{lon}_{i}-{xsz}_{j}-{ysz}.nc'
    
    if os.path.exists(regrid_name):
        regridder = xe.Regridder(ds_from, ds_to, "bilinear", unmapped_to_nan=True, weights=regrid_name)
    else:
        regridder = xe.Regridder(ds_from, ds_to, "bilinear", unmapped_to_nan=True)
        regridder.to_netcdf(regrid_name)
    
    return regridder
        
def add_met_blh(ds, met, blh):
    dx = 500
    dy = 500

    time = pd.to_datetime(ds.time.item()).normalize()
    times_da = xr.DataArray([time], [('time', [time])])
    ds['time'] = times_da

    for d in [met, blh]:
        for var in d.data_vars:
            ds[var] = xr.full_like(ds.AOD, fill_value=np.nan)
            ds[var].attrs = d[var].attrs

    metds = met.sel(time=time.strftime('%m/%d/%Y'))
    blhds = blh.sel(time=time.strftime('%m/%d/%Y'))

    pbar = tqdm(range(int(np.ceil(ds.x.size / dx))))
    for i in pbar:
        for j in range(int(np.ceil(ds.y.size / dy))):
            pbar.set_description(f'{i}-{j}')
            sub = ds.isel(x=slice(i * dx, (i+1)*dx), y=slice(j * dy, (j+1)*dy))

            for vartype, ads in zip(['met', 'blh'], [metds, blhds]):
                regridder = get_regridder(ads, ds, vartype, i, j)
                out = regridder(ads)
                out = out.expand_dims(time=times_da)
                for var in ads.data_vars:
                    ds[var].loc[dict(x=out.x, y=out.y)] = out[var]
    return ds

def add_pm(ds, df, proj):
    time = pd.to_datetime(ds.time.item())
    sub = df.loc[time.strftime('%Y-%m-%d')]

    transformed = proj.transform_points(src_crs=ccrs.PlateCarree(), x=sub['longitude'], y=sub['latitude'])
    sub['proj_longitude'] = transformed[:,0]
    sub['proj_latitude'] = transformed[:,1]

    ds['PM25'] = xr.full_like(ds.AOD, fill_value=np.nan).astype(np.float32)
    ds['PM25'].attrs = {}

    for idx, row in sub.iterrows():
        if (row.proj_longitude > ds.x.min().item()) \
            and (row.proj_longitude < ds.x.max().item()) \
            and (row.proj_latitude > ds.y.min().item()) \
            and (row.proj_latitude < ds.y.max().item()):

            data = ds.sel(
                x=row.proj_longitude, 
                y=row.proj_latitude, 
                method='nearest', 
                tolerance=2000
            )
            ds['PM25'].loc[dict(x=data.x, y=data.y)] = row.value
    return ds

def day_number_to_date(year, day_number):
    date = pd.to_datetime(year, format='%Y')
    
    return date + pd.to_timedelta(day_number - 1, unit='D') 

df = pd.concat(
    [pd.read_csv(f'/Volumes/Minuet/research/EPA/daily_88101_{year}.csv', parse_dates=['Date Local']) 
     for year in range(2017, 2023)],
    axis=0
)

df.rename(columns = {
    'Date Local':'time', 
    'Arithmetic Mean': 'value',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
}, inplace = True)

keep = ['latitude', 'longitude', 'time', 'value','Local Site Name', 'Address', 'State Name', 'County Name','City Name', 'POC']
df.drop(columns=df.columns.difference(keep),inplace=True)
df = df.loc[df.value >= 0]
df = df.groupby(['time', 'longitude', 'latitude']).mean().reset_index().set_index('time')
df.sort_index(inplace=True)

met = None
blh = None

import warnings
warnings.filterwarnings('ignore')

path='/Volumes/Canon/matched'

date = pd.to_datetime(sys.argv[1])
dates = []
for i in range(date.days_in_month):
    dates.append(date + pd.Timedelta(days=i))

for date in dates:
    pbar.set_description(f'{date.year} {date.day_of_year}')
    
    day = f'{date.day_of_year:03}'
    
    save_path = os.path.join(path, f'{date.year}')
    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, f'{day}.csv')
    save_path = os.path.join(save_path, f'{day}.nc')
    if os.path.exists(save_path):
        continue

    fname = f'/Volumes/Canon/averages/daily/{date.year}/{date.day_of_year:03}.nc'
    try:
        with xr.open_dataset(fname) as ds:
            ds.load()
        xs, ys = np.meshgrid(ds.x, ds.y)
        proj = ds.metpy.parse_cf('AOD').metpy.cartopy_crs
        transformed = plate.transform_points(src_crs=proj, x=xs, y=ys)

        ds = ds.assign_coords({
                "lat":(["y","x"],transformed[:,:,1]),
                "lon":(["y","x"],transformed[:,:,0])
            })
        ds.lat.attrs["units"] = "degrees_north"
        ds.lon.attrs["units"] = "degrees_east"
    except FileNotFoundError:
        print('file not found', date)
    except OSError:
        print('os error', date)
        
    time = pd.to_datetime(ds.time.item()).normalize()
    
    if (met is None) or not (time in met.time):
        with xr.open_dataset(f'/Volumes/Minuet/research/ERA5/era5_land/era5_land_{time.year}{time.month:02}-averaged.nc') as met:
            met.load()
        met = met.rename({"longitude": "lon", "latitude": "lat"})

    if (blh is None) or not (time in blh.time):
        with xr.open_dataset(f'/Volumes/Minuet/research/ERA5/era5_pblh_{time.year}-averaged.nc') as blh:
            blh.load()
        blh = blh.rename({"longitude": "lon", "latitude": "lat"})

    ds = add_met_blh(ds, met, blh)
    ds = add_pm(ds, df, proj)
    csv = ds.drop_vars('goes_imager_projection').to_dataframe().reset_index().drop(columns=['x', 'y'])
    csv.dropna(subset=['PM25', 'AOD']).to_csv(csv_path, index=False)
    ds.to_netcdf(save_path)
