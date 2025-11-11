from pathlib import Path
from typing import Any, cast
import gc

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import rioxarray as rxr
import xarray as xr


def read_dem(
        raster_path: Path,
        crs: str | None = None,
        **params: Any,
) -> xr.DataArray:
    """
    Read a GeoTIFF file into an xarray DataArray.
    Check if CRS is in EPSG:4326, if not convert it.
    If it does not have a CRS, assign the provided one.
    Then, convert to EPSG:4326.

    Parameters
    ----------
    raster_path : Path
        Path to the input raster file.
    crs : str, optional
        Coordinate Reference System to assign if the raster lacks one. Defaults to None.
    **params : Any
        Additional parameters passed to rioxarray.open_rasterio().
    """

    raster = cast(xr.DataArray, rxr.open_rasterio(raster_path, masked=True, **params))

    if raster.rio.crs is None:
        if crs is None:
            raise ValueError("CRS must be provided for rasters without CRS.")
        else:
            raster = raster.rio.write_crs(crs)

    if raster.rio.crs != 'EPSG:4326':
        raster = raster.rio.reproject('EPSG:4326')

    return raster


def dem2xyz(
        dem_path: Path,
        name: str | None = None,
        **params: Any,
) -> None:
    """
    Convert a DEM xarray DataArray (small or big) to three columns XYZ format
    and save it to a file for easy retrieval using pyGMT process.
    
    Parameters
    ----------
    dem_path : Path
        Path to the input DEM raster file.
    name : str, optional
        Name to assign to the DataArray if it is unnamed. Defaults to 'elevation'.
    **params : Any
        Additional parameters passed to read_dem() (e.g., crs, chunks).
    """

    dem = read_dem(dem_path, **params)
    new_path = dem_path.with_suffix('.xyz')

    # Ensure the DataArray has a name for conversion to DataFrame
    if dem.name is None:
        dem = dem.rename(name or 'elevation')

    try:
        if isinstance(dem.data, da.Array):
            # Handle large DEMs with Dask
            dask_df: dd.DataFrame = dem.to_dask_dataframe().reset_index()
            dask_df = dask_df[['x', 'y', dem.name]]
            dask_df = dask_df.dropna(subset=[dem.name])
            dask_df.to_csv(str(new_path), sep=' ', index=False, header=False, single_file=True)
            
            # Explicit cleanup for Dask objects
            del dask_df
            gc.collect()
        else:
            # Handle small DEMs with Pandas
            pandas_df: pd.DataFrame = dem.to_dataframe(name=dem.name).reset_index()
            pandas_df = pandas_df.dropna(subset=[dem.name])
            pandas_df.to_csv(new_path, sep=' ', index=False, header=False)
            
            # Explicit cleanup for Pandas objects
            del pandas_df
            gc.collect()
    finally:
        # Always cleanup the dem object after use
        del dem
        gc.collect()