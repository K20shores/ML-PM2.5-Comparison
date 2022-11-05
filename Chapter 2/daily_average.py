import os

import xarray as xr
import numpy as np
import pandas as pd

xr.set_options(keep_attrs=True)

def daily_average(
        date,
        save_path = '/Volumes/Canon/averages/daily'
    ):
    ds = xr.open_mfdataset(
        f'/Volumes/Canon/averages/hourly/{date.year}/{date.day_of_year:03}/*.nc',
        concat_dim="t",
        parallel=True,
        combine='nested',
        data_vars='different',
    )

    shape = ds.y.size, ds.x.size

    # find the total number of snapshots used in a cell over the 24 hour period
    total = ds.n_snapshots.sum('t').compute()
    # the sum places zeros where nans were, replace those zeros with nans
    # so that we know which grid cells never had data
    total = xr.where(total == 0, np.nan, total)

    # AOD 550nm average
    weights = (ds.n_snapshots / total).compute().data
    weighted = weights * ds.AOD.data.compute()
    avg = np.nansum(weighted, axis=0)
    avg = xr.where(total.isnull(), np.nan, avg).data

    # AOD 470nm average if it exists
    avg470 = np.full(shape, np.nan)
    ae = np.full(shape, np.nan)
    aod470_attrs = {}

    if 'AOD470' in ds:
        aod470_attrs = ds.AOD470.attrs
        # this one shows us were aod470 was not null at least once in all cells. not sure
        # how to do this by hand
        a = ds.AOD470.resample(t='1d').mean()
        snaps = ds.n_snapshots.copy()
        snaps = xr.where(ds['AOD470'].isnull(), np.nan, snaps)

        aod470_total = snaps.sum('t').compute()
        aod470_total = xr.where(aod470_total == 0, np.nan, aod470_total)
        aod470_weights = (snaps / aod470_total).compute().data

        weighted = aod470_weights * ds.AOD470.data.compute()
        avg470 = np.nansum(weighted, axis=0)
        avg470 = xr.where(a.isnull(), np.nan, avg470).data.compute()
        ae = (- np.log(avg470 / avg) / np.log(470/550))

    # create the dataset
    time = [pd.to_datetime(ds.t.resample(t='1d').mean().item())]
    coord_order = ['time', 'y', 'x']
    shape = (1, *shape)

    ae1_attrs = {'long_name': 'Angstrom Exponent corresponding to 0.47 / 0.86 micron wavelength pair',
     'valid_range': [0, 65530],
     'grid_mapping': 'goes_imager_projection'}

    data_vars = {
        'AOD':        ( coord_order, avg.reshape(shape), ds.AOD.attrs ),
        'AOD470':     ( coord_order, avg470.reshape(shape).astype(np.float32)  , aod470_attrs ),
        'AE':         ( coord_order, ae.reshape(shape).astype(np.float32)  , ae1_attrs ),
        'n_snapshots':( coord_order, total.data.reshape(shape), ds.n_snapshots.attrs ),
        'goes_imager_projection': ([], 0, ds.goes_imager_projection.attrs)
    }

    # define coordinates
    coords = {
        'time': (['time'], time),
        'y': (['y'], ds.y.data, ds.y.attrs),
        'x': (['x'], ds.x.data, ds.x.attrs)
    }

    # define global attributes
    attrs = {}

    # create dataset
    ds1 = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    year = os.path.join(save_path, f'{date.year}')
    os.makedirs(year, exist_ok=True)
    path = os.path.join(year, f'{date.day_of_year:03}.nc')
    ds1.to_netcdf(path)

import sys

date = pd.to_datetime(sys.argv[1])
daily_average(date)

