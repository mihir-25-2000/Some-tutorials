import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.tutorial.load_dataset("air_temperature")
test_plot1=ds['air'].isel(time=0).plot()
plt.savefig("/home/mihir.more/myenv/test_plot1.png")
print("Random image saved as test_plot1.png")

zarr_path = "/Datastorage/saptarishi.dhanuka_asp25/20230101_20240926_imerg_era5.zarr"
ds=xr.open_zarr(zarr_path, consolidated=True)
print(ds)
print("test")