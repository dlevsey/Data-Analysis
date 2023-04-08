#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr


# In[2]:


#import debugger
#import ipdb


# #### To Use Debugger:
# 
# ```
# def my_function(a, b):
#     result = a + b
#     ipdb.set_trace()  # Set the breakpoint here
#     return result
# 
# my_function(3, 4)
# ```

# In[3]:


carbon_data = nc.Dataset("dataset_satellite_data_carbon_dioxide.nc", 'r')


# In[4]:


global_attributes = carbon_data.ncattrs()
print(global_attributes)


# In[5]:


print(carbon_data.title)


# In[6]:


carbon_data = nc.Dataset("dataset_satellite_data_carbon_dioxide.nc", 'r')

# Print variable names and their attributes
for var_name in carbon_data.variables:
    print(f"Variable: {var_name}")
    print("Attributes:")
    for attr_name in carbon_data.variables[var_name].ncattrs():
        print(f"  {attr_name}: {getattr(carbon_data.variables[var_name], attr_name)}")
    print()


# In[7]:


carbon_data['lat']


# In[8]:


carbon_test = xr.open_dataset("dataset_satellite_data_carbon_dioxide.nc")


# In[9]:


import cartopy.crs as ccrs
import ipywidgets as widgets
from IPython.display import display
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# In[10]:


#help(carbon_test.xco2)
#help(carbon_test.xco2.sel)
#help(sliced_data)
#help(slice)


# In[11]:


import nc_time_axis
import cartopy.feature as cfeature


# In[12]:


states = cfeature.NaturalEarthFeature(category = 'cultural', 
                                     name= 'admin_1_states_provinces_lines',
                                     scale= '50m',
                                     facecolor = 'none')


# In[13]:


#help(ax)


# In[14]:


min_longitude = -125
max_longitude = -116
min_latitude = 41.5
max_latitude = 47
sliced_data = carbon_test.xco2[10,:,:].sel(lat=slice(min_latitude, max_latitude), 
                                                          lon=slice(min_longitude, max_longitude)) 

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(states, edgecolor='black')
ax.coastlines()

ax.set_extent((min_longitude, max_longitude, 
               min_latitude, max_latitude), crs=ccrs.PlateCarree())


# In[15]:


def plot_co2(time_index):
    
    """
    Plots CO2 levels at the specified time index using a PlateCarree projection map.
    
    Args:
        time_index (int): The time index of the CO2 data to be plotted.
        
    Returns:
        None
    """
    
    
    # Create a new figure with appropriate size
    fig = plt.figure(figsize=(12,6))
    
    # Create an axis with the PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set the extent of the plot to cover the entire globe
    ax.set_extent((min_longitude, max_longitude, 
               min_latitude, max_latitude), crs=ccrs.PlateCarree())

    ax.add_feature(states, edgecolor='black')
    # Add coastlines to the plot
    ax.coastlines()

    # Plot the CO2 data with appropriate latitude and longitude coordinates
    img = ax.pcolormesh(carbon_test.lon, carbon_test.lat,
                        carbon_test.xco2[time_index,:,:], 
                        cmap='viridis', transform=ccrs.PlateCarree())

    # Add gridlines and labels for latitude and longitude
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha= 1.0 )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Add colorbar and labels
    plt.colorbar(img, label='CO2 levels', 
                 orientation='vertical',location = 'left', pad=0.05)
    plt.title(f'Time index: {time_index}')
    
    plt.show()



# In[16]:


slider = widgets.IntSlider(min=0, max=len(carbon_test.xco2)-1, 
                           step=1, value=0, description='Time index')

widgets.interact(plot_co2, time_index=slider)


# One common approach is to use emission factors, which represent the mass of a pollutant (in this case, PM) emitted per unit mass of fuel burned. These factors can be specific to different types of vegetation, fuel load, and combustion phase (i.e., flaming or smoldering). The general equation for estimating PM emissions from a forest fire using emission factors is:
# 
# ``` PM emissions = Fuel consumed × Emission factor ```
# 
# #### where:
# 
# **PM emissions** represent the mass of particulate matter released (usually in grams, kilograms, or tons)
# **Fuel consumed** is the mass of fuel burned during the fire (usually in grams, kilograms, or tons)
# **Emission factor** is the mass of PM emitted per unit mass of fuel burned (usually in grams per kilogram, g/kg)
# Emission factors for PM can vary widely depending on the type of vegetation and combustion conditions. 
# 
# Researchers often use values from field studies, laboratory experiments, or databases, such as the U.S. Environmental Protection Agency's National Emission Inventory (NEI) or the Emissions Database for Global Atmospheric Research (EDGAR).
# 
# It is important to note that these estimates have inherent uncertainties due to the variability in fuel characteristics, combustion conditions, and other factors. More advanced mathematical models, such as the BlueSky Smoke Modeling Framework, the Fire Emission Production Simulator (FEPS), or the Canadian Forest Fire Emission Prediction System (C

# #### AQI Daily Oregon

# In[17]:


# read in data
dailyAQI_1990 = pd.read_csv("daily summary aqi_1990.csv")


# In[18]:


#print(dailyAQI_1990)


# In[ ]:





# In[19]:


#Men tips data
#men = dailyAQI_1990[dailyAQI_1990["State Name"] == "Oregon"]["tip"]

AQI = dailyAQI_1990[dailyAQI_1990["State Name"] == "Oregon"][["AQI", "Latitude", "Longitude"]]
AQI


# In[20]:


# vmin = AQI["AQI"].min()
# vmax = AQI["AQI"].max()


# In[21]:


# Reshape the DataFrame so that each time index has its own column
AQI_reshaped = AQI.pivot_table(index=["Latitude", "Longitude"], columns=AQI.groupby(["Latitude", "Longitude"]).cumcount(), values="AQI")
AQI_reshaped.reset_index(inplace= True)


# In[22]:


def plot_data3(time_index):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(states, edgecolor = 'black')
     
    # Set the extent of the plot to cover the entire globe
    ax.set_extent((min_longitude, max_longitude, 
               min_latitude, max_latitude), crs=ccrs.PlateCarree())

    
    # Plot the AQI values from the reshaped DataFrame with appropriate latitude and longitude coordinates
    img = ax.scatter(AQI_reshaped["Longitude"], AQI_reshaped["Latitude"], 
                     c=AQI_reshaped[time_index], 
                     cmap="viridis", vmin=AQI["AQI"].min(), vmax=AQI["AQI"].max(),
                     transform=ccrs.PlateCarree())
  # Add gridlines and labels for latitude and longitude
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha= 1.0 )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.colorbar(img, label = "Air Quality Index")


# In[23]:


slider = widgets.IntSlider(min=0, max=AQI_reshaped.columns.size - 3, 
                           step=1, value=0)
widgets.interact(plot_data3, time_index=slider)


# In[24]:


import geopandas as gpd
from shapely import wkt


# In[25]:


AQI_counties_csv = pd.read_csv("daily_aqi_by_county_2022.csv")
print(AQI_counties_csv.head())


# In[26]:


shapefile_path = "tl_2022_41_cousub.shp"
counties_geometries = gpd.read_file(shapefile_path)
import us

from uszipcode import SearchEngine


# In[27]:


print(counties_geometries.head())


# In[28]:


AQI_counties_csv['FIPS'] = AQI_counties_csv['State Code'].astype(str).str.zfill(2) + AQI_counties_csv['County Code'].astype(str).str.zfill(3)


# In[29]:


AQI_counties_csv


# In[30]:


counties_geometries['FIPS'] = counties_geometries['STATEFP'] + counties_geometries['COUNTYFP']


# In[31]:


merged_data = pd.merge(AQI_counties_csv, counties_geometries, on='FIPS', how='left')


# In[50]:


oregon_data = merged_data[merged_data['State Name'] == 'Oregon']

AQI_columns = ['AQI', 'County Code', 'Defining Site']
oregon_data


# In[33]:


oregon_geo = gpd.GeoDataFrame(oregon_data, geometry='geometry', crs='epsg:4326')
AQI = oregon_data[AQI_columns]
AQI


# In[34]:


def plot_data4(time_index):
    fig = plt.figure(figsize=(12, 6))
    
    ax = plt.subplots(1, 1, projection = ccrs.PlateCarree())
    col_name = oregon_geo.columns[time_index + 2]  # Add 2 to skip the 'geometry' and 'index' columns
    
    ax.add_feature(states, edgecolor = 'black')
     
    # Set the extent of the plot to cover the entire globe
    ax.set_extent((min_longitude, max_longitude, 
               min_latitude, max_latitude), crs=ccrs.PlateCarree())
    
    # Plot the AQI values from the reshaped DataFrame with appropriate latitude and longitude coordinates
    img = ax.scatter(AQI["Longitude"], AQI_reshaped["Latitude"], 
                     c=AQI_reshaped[time_index], 
                     cmap="viridis", vmin=AQI["AQI"].min(), vmax=AQI["AQI"].max(),
                     transform=ccrs.PlateCarree())
    
    oregon_geo.plot(column=col_name, cmap='viridis', 
                    linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    
    plt.title('Air Quality Index in Oregon')
    
    plt.draw()
    plt.pause(0.01)


# In[35]:


state_boundaries = gpd.read_file('ne_50m_admin_1_states_provinces.shp')
state_boundaries.head()


# In[36]:


oregon_boundary = state_boundaries[state_boundaries['iso_3166_2'].str.startswith("US-OR")]


# In[37]:


def plot_data5(time_index):
    col_name = oregon_geo.columns[time_index + 2]  # Add 2 to skip the 'geometry' and 'index' columns
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    oregon_geo.plot(column=col_name, cmap='viridis', 
                    linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    
    # Add the Oregon state boundary to the plot
    oregon_boundary.plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)
    
    plt.title('Air Quality Index in Oregon')
    
    plt.draw()
    plt.pause(0.01)


# In[44]:


oregon_geo


# In[47]:


def pivot_oregon_geo(df):
    # Filter the columns we want to keep
    columns_to_keep = ['geometry', 'FIPS'] + [str(i) for i in range(df.columns.get_loc('0'), df.columns.get_loc('23') + 1)]
    df_filtered = df[columns_to_keep]

    # Melt the DataFrame
    df_filtered = df_filtered.melt(id_vars=['geometry', 'FIPS'], var_name='time_index', value_name='AQI')

    df_filtered['time_index'] = df_filtered['time_index'].astype(int)
    return gpd.GeoDataFrame(df_filtered, geometry='geometry', crs='epsg:4326')


# In[39]:


def plot_data5(time_index):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Filter the data based on the time index
    data_to_plot = oregon_geo_pivoted[oregon_geo_pivoted['time_index'] == time_index]

    # Plot the filtered data
    data_to_plot.plot(column='AQI', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Add the Oregon state boundary to the plot


# In[ ]:



slider = widgets.IntSlider(min=0, max=oregon_data.columns.size - 3, 
                             step=1, value=0)
widgets.interact(plot_data5, time_index=slider)


# In[ ]:




