---
layout: post
title: How to save a dataframe to a netcdf file
tags: xarray
---

### How to save a txt file to Netcdf file format

* Read one daily station file from [USCRN](https://www.ncdc.noaa.gov/crn/)
* Save it as a Netcdf file with dimension (time, station)


```python
import xarray as xr
import pandas as pd
```


```python
# An example of USCRN station data
uscrn_file = 'https://www1.ncdc.noaa.gov/pub/data/uscrn/products/daily01/2019/CRND0103-2019-AK_Bethel_87_WNW.txt'
uscrn_header_file = 'ftp://ftp.ncdc.noaa.gov/pub/data/uscrn/products/daily01/HEADERS.txt'
# Read header file
uscrn_header = pd.read_csv(uscrn_header_file, sep='\s+')
# Read data file
uscrn_df = pd.read_csv(uscrn_file, sep='\s+', header=None)
```


```python
uscrn_header
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WBANNO</td>
      <td>LST_DATE</td>
      <td>CRX_VN</td>
      <td>LONGITUDE</td>
      <td>LATITUDE</td>
      <td>T_DAILY_MAX</td>
      <td>T_DAILY_MIN</td>
      <td>T_DAILY_MEAN</td>
      <td>T_DAILY_AVG</td>
      <td>P_DAILY_CALC</td>
      <td>...</td>
      <td>SOIL_MOISTURE_5_DAILY</td>
      <td>SOIL_MOISTURE_10_DAILY</td>
      <td>SOIL_MOISTURE_20_DAILY</td>
      <td>SOIL_MOISTURE_50_DAILY</td>
      <td>SOIL_MOISTURE_100_DAILY</td>
      <td>SOIL_TEMP_5_DAILY</td>
      <td>SOIL_TEMP_10_DAILY</td>
      <td>SOIL_TEMP_20_DAILY</td>
      <td>SOIL_TEMP_50_DAILY</td>
      <td>SOIL_TEMP_100_DAILY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>XXXXX</td>
      <td>YYYYMMDD</td>
      <td>XXXXXX</td>
      <td>Decimal_degrees</td>
      <td>Decimal_degrees</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>mm</td>
      <td>...</td>
      <td>m^3/m^3</td>
      <td>m^3/m^3</td>
      <td>m^3/m^3</td>
      <td>m^3/m^3</td>
      <td>m^3/m^3</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>Celsius</td>
      <td>Celsius</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>




```python
uscrn_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26656</td>
      <td>20190101</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>0.7</td>
      <td>-12.4</td>
      <td>-5.8</td>
      <td>-5.1</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26656</td>
      <td>20190102</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-12.3</td>
      <td>-17.0</td>
      <td>-14.7</td>
      <td>-14.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26656</td>
      <td>20190103</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-9.9</td>
      <td>-14.0</td>
      <td>-12.0</td>
      <td>-11.8</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26656</td>
      <td>20190104</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-7.6</td>
      <td>-12.3</td>
      <td>-9.9</td>
      <td>-10.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26656</td>
      <td>20190105</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-8.2</td>
      <td>-14.1</td>
      <td>-11.2</td>
      <td>-11.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# Set uscrn_df column name using uscrn_header information
uscrn_df.columns = uscrn_header.iloc[0, :]
```


```python
uscrn_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WBANNO</th>
      <th>LST_DATE</th>
      <th>CRX_VN</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>T_DAILY_MAX</th>
      <th>T_DAILY_MIN</th>
      <th>T_DAILY_MEAN</th>
      <th>T_DAILY_AVG</th>
      <th>P_DAILY_CALC</th>
      <th>...</th>
      <th>SOIL_MOISTURE_5_DAILY</th>
      <th>SOIL_MOISTURE_10_DAILY</th>
      <th>SOIL_MOISTURE_20_DAILY</th>
      <th>SOIL_MOISTURE_50_DAILY</th>
      <th>SOIL_MOISTURE_100_DAILY</th>
      <th>SOIL_TEMP_5_DAILY</th>
      <th>SOIL_TEMP_10_DAILY</th>
      <th>SOIL_TEMP_20_DAILY</th>
      <th>SOIL_TEMP_50_DAILY</th>
      <th>SOIL_TEMP_100_DAILY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26656</td>
      <td>20190101</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>0.7</td>
      <td>-12.4</td>
      <td>-5.8</td>
      <td>-5.1</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26656</td>
      <td>20190102</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-12.3</td>
      <td>-17.0</td>
      <td>-14.7</td>
      <td>-14.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26656</td>
      <td>20190103</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-9.9</td>
      <td>-14.0</td>
      <td>-12.0</td>
      <td>-11.8</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26656</td>
      <td>20190104</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-7.6</td>
      <td>-12.3</td>
      <td>-9.9</td>
      <td>-10.4</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26656</td>
      <td>20190105</td>
      <td>2.515</td>
      <td>-164.08</td>
      <td>61.35</td>
      <td>-8.2</td>
      <td>-14.1</td>
      <td>-11.2</td>
      <td>-11.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# Drop constant variables
uscrn_df_clean = uscrn_df.drop(['WBANNO', 'LONGITUDE', 'LATITUDE', 'CRX_VN'], axis=1)
```


```python
uscrn_df_clean.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LST_DATE</th>
      <th>T_DAILY_MAX</th>
      <th>T_DAILY_MIN</th>
      <th>T_DAILY_MEAN</th>
      <th>T_DAILY_AVG</th>
      <th>P_DAILY_CALC</th>
      <th>SOLARAD_DAILY</th>
      <th>SUR_TEMP_DAILY_TYPE</th>
      <th>SUR_TEMP_DAILY_MAX</th>
      <th>SUR_TEMP_DAILY_MIN</th>
      <th>...</th>
      <th>SOIL_MOISTURE_5_DAILY</th>
      <th>SOIL_MOISTURE_10_DAILY</th>
      <th>SOIL_MOISTURE_20_DAILY</th>
      <th>SOIL_MOISTURE_50_DAILY</th>
      <th>SOIL_MOISTURE_100_DAILY</th>
      <th>SOIL_TEMP_5_DAILY</th>
      <th>SOIL_TEMP_10_DAILY</th>
      <th>SOIL_TEMP_20_DAILY</th>
      <th>SOIL_TEMP_50_DAILY</th>
      <th>SOIL_TEMP_100_DAILY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20190101</td>
      <td>0.7</td>
      <td>-12.4</td>
      <td>-5.8</td>
      <td>-5.1</td>
      <td>0.0</td>
      <td>0.26</td>
      <td>C</td>
      <td>-1.1</td>
      <td>-12.4</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20190102</td>
      <td>-12.3</td>
      <td>-17.0</td>
      <td>-14.7</td>
      <td>-14.4</td>
      <td>0.0</td>
      <td>0.68</td>
      <td>C</td>
      <td>-12.4</td>
      <td>-21.3</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20190103</td>
      <td>-9.9</td>
      <td>-14.0</td>
      <td>-12.0</td>
      <td>-11.8</td>
      <td>0.0</td>
      <td>0.17</td>
      <td>C</td>
      <td>-9.9</td>
      <td>-15.0</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20190104</td>
      <td>-7.6</td>
      <td>-12.3</td>
      <td>-9.9</td>
      <td>-10.4</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>C</td>
      <td>-7.7</td>
      <td>-13.3</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20190105</td>
      <td>-8.2</td>
      <td>-14.1</td>
      <td>-11.2</td>
      <td>-11.5</td>
      <td>0.0</td>
      <td>0.48</td>
      <td>C</td>
      <td>-8.8</td>
      <td>-14.3</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# Set LST_DATE as the dataframe index 
uscrn_df_clean = uscrn_df_clean.set_index('LST_DATE')
uscrn_df_clean.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T_DAILY_MAX</th>
      <th>T_DAILY_MIN</th>
      <th>T_DAILY_MEAN</th>
      <th>T_DAILY_AVG</th>
      <th>P_DAILY_CALC</th>
      <th>SOLARAD_DAILY</th>
      <th>SUR_TEMP_DAILY_TYPE</th>
      <th>SUR_TEMP_DAILY_MAX</th>
      <th>SUR_TEMP_DAILY_MIN</th>
      <th>SUR_TEMP_DAILY_AVG</th>
      <th>...</th>
      <th>SOIL_MOISTURE_5_DAILY</th>
      <th>SOIL_MOISTURE_10_DAILY</th>
      <th>SOIL_MOISTURE_20_DAILY</th>
      <th>SOIL_MOISTURE_50_DAILY</th>
      <th>SOIL_MOISTURE_100_DAILY</th>
      <th>SOIL_TEMP_5_DAILY</th>
      <th>SOIL_TEMP_10_DAILY</th>
      <th>SOIL_TEMP_20_DAILY</th>
      <th>SOIL_TEMP_50_DAILY</th>
      <th>SOIL_TEMP_100_DAILY</th>
    </tr>
    <tr>
      <th>LST_DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20190101</th>
      <td>0.7</td>
      <td>-12.4</td>
      <td>-5.8</td>
      <td>-5.1</td>
      <td>0.0</td>
      <td>0.26</td>
      <td>C</td>
      <td>-1.1</td>
      <td>-12.4</td>
      <td>-5.3</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>20190102</th>
      <td>-12.3</td>
      <td>-17.0</td>
      <td>-14.7</td>
      <td>-14.4</td>
      <td>0.0</td>
      <td>0.68</td>
      <td>C</td>
      <td>-12.4</td>
      <td>-21.3</td>
      <td>-16.4</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>20190103</th>
      <td>-9.9</td>
      <td>-14.0</td>
      <td>-12.0</td>
      <td>-11.8</td>
      <td>0.0</td>
      <td>0.17</td>
      <td>C</td>
      <td>-9.9</td>
      <td>-15.0</td>
      <td>-11.8</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>20190104</th>
      <td>-7.6</td>
      <td>-12.3</td>
      <td>-9.9</td>
      <td>-10.4</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>C</td>
      <td>-7.7</td>
      <td>-13.3</td>
      <td>-10.5</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
    <tr>
      <th>20190105</th>
      <td>-8.2</td>
      <td>-14.1</td>
      <td>-11.2</td>
      <td>-11.5</td>
      <td>0.0</td>
      <td>0.48</td>
      <td>C</td>
      <td>-8.8</td>
      <td>-14.3</td>
      <td>-11.9</td>
      <td>...</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-99.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
      <td>-9999.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# Use xarray function to convert dataframe into dataset
uscrn_ds = xr.Dataset.from_dataframe(uscrn_df_clean)
```


```python
uscrn_ds
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
    Data variables:
        T_DAILY_MAX              (LST_DATE) float64 0.7 -12.3 -9.9 ... 0.0 0.0
        T_DAILY_MIN              (LST_DATE) float64 -12.4 -17.0 -14.0 ... -4.5 -2.5
        T_DAILY_MEAN             (LST_DATE) float64 -5.8 -14.7 -12.0 ... -2.3 -1.3
        T_DAILY_AVG              (LST_DATE) float64 -5.1 -14.4 -11.8 ... -1.4 -0.3
        P_DAILY_CALC             (LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (LST_DATE) float64 0.26 0.68 0.17 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (LST_DATE) object 'C' 'C' 'C' 'C' ... 'U' 'C' 'C'
        SUR_TEMP_DAILY_MAX       (LST_DATE) float64 -1.1 -12.4 -9.9 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (LST_DATE) float64 -12.4 -21.3 -15.0 ... -6.5 -3.6
        SUR_TEMP_DAILY_AVG       (LST_DATE) float64 -5.3 -16.4 -11.8 ... -0.2 0.1
        RH_DAILY_MAX             (LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_10_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_20_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_50_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_100_DAILY  (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_TEMP_5_DAILY        (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (LST_DATE) float64 -9.999e+03 ... -9.999e+03




```python
# Add dropped constant ['WBANNO', 'LONGITUDE', 'LATITUDE', 'CRX_VN'] to xarray dataset
for x in ['WBANNO', 'LONGITUDE', 'LATITUDE', 'CRX_VN']:
    uscrn_ds[x] = uscrn_df[x][0]
```


```python
uscrn_ds
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
    Data variables:
        T_DAILY_MAX              (LST_DATE) float64 0.7 -12.3 -9.9 ... 0.0 0.0
        T_DAILY_MIN              (LST_DATE) float64 -12.4 -17.0 -14.0 ... -4.5 -2.5
        T_DAILY_MEAN             (LST_DATE) float64 -5.8 -14.7 -12.0 ... -2.3 -1.3
        T_DAILY_AVG              (LST_DATE) float64 -5.1 -14.4 -11.8 ... -1.4 -0.3
        P_DAILY_CALC             (LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (LST_DATE) float64 0.26 0.68 0.17 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (LST_DATE) object 'C' 'C' 'C' 'C' ... 'U' 'C' 'C'
        SUR_TEMP_DAILY_MAX       (LST_DATE) float64 -1.1 -12.4 -9.9 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (LST_DATE) float64 -12.4 -21.3 -15.0 ... -6.5 -3.6
        SUR_TEMP_DAILY_AVG       (LST_DATE) float64 -5.3 -16.4 -11.8 ... -0.2 0.1
        RH_DAILY_MAX             (LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_10_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_20_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_50_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_100_DAILY  (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_TEMP_5_DAILY        (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        WBANNO                   int64 26656
        LONGITUDE                float64 -164.1
        LATITUDE                 float64 61.35
        CRX_VN                   float64 2.515




```python
# Set time and station as xarray dataset coordinates
uscrn_ds.set_coords(['LST_DATE', 'WBANNO'])
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
        WBANNO                   int64 26656
    Data variables:
        T_DAILY_MAX              (LST_DATE) float64 0.7 -12.3 -9.9 ... 0.0 0.0
        T_DAILY_MIN              (LST_DATE) float64 -12.4 -17.0 -14.0 ... -4.5 -2.5
        T_DAILY_MEAN             (LST_DATE) float64 -5.8 -14.7 -12.0 ... -2.3 -1.3
        T_DAILY_AVG              (LST_DATE) float64 -5.1 -14.4 -11.8 ... -1.4 -0.3
        P_DAILY_CALC             (LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (LST_DATE) float64 0.26 0.68 0.17 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (LST_DATE) object 'C' 'C' 'C' 'C' ... 'U' 'C' 'C'
        SUR_TEMP_DAILY_MAX       (LST_DATE) float64 -1.1 -12.4 -9.9 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (LST_DATE) float64 -12.4 -21.3 -15.0 ... -6.5 -3.6
        SUR_TEMP_DAILY_AVG       (LST_DATE) float64 -5.3 -16.4 -11.8 ... -0.2 0.1
        RH_DAILY_MAX             (LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_10_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_20_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_50_DAILY   (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_MOISTURE_100_DAILY  (LST_DATE) float64 -99.0 -99.0 ... -99.0 -99.0
        SOIL_TEMP_5_DAILY        (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (LST_DATE) float64 -9.999e+03 ... -9.999e+03
        LONGITUDE                float64 -164.1
        LATITUDE                 float64 61.35
        CRX_VN                   float64 2.515




```python
# Expand dataset dimension from one to two
uscrn_ds.set_coords(['LST_DATE', 'WBANNO']).expand_dims('WBANNO')
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89, WBANNO: 1)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
      * WBANNO                   (WBANNO) int64 26656
    Data variables:
        T_DAILY_MAX              (WBANNO, LST_DATE) float64 0.7 -12.3 ... 0.0 0.0
        T_DAILY_MIN              (WBANNO, LST_DATE) float64 -12.4 -17.0 ... -2.5
        T_DAILY_MEAN             (WBANNO, LST_DATE) float64 -5.8 -14.7 ... -2.3 -1.3
        T_DAILY_AVG              (WBANNO, LST_DATE) float64 -5.1 -14.4 ... -1.4 -0.3
        P_DAILY_CALC             (WBANNO, LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (WBANNO, LST_DATE) float64 0.26 0.68 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (WBANNO, LST_DATE) object 'C' 'C' 'C' ... 'C' 'C'
        SUR_TEMP_DAILY_MAX       (WBANNO, LST_DATE) float64 -1.1 -12.4 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (WBANNO, LST_DATE) float64 -12.4 -21.3 ... -3.6
        SUR_TEMP_DAILY_AVG       (WBANNO, LST_DATE) float64 -5.3 -16.4 ... -0.2 0.1
        RH_DAILY_MAX             (WBANNO, LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (WBANNO, LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (WBANNO, LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_10_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_20_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_50_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_100_DAILY  (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_TEMP_5_DAILY        (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        LONGITUDE                (WBANNO) float64 -164.1
        LATITUDE                 (WBANNO) float64 61.35
        CRX_VN                   (WBANNO) float64 2.515




```python
# Make sure assign the results to a variable
uscrn_ds = uscrn_ds.set_coords(['LST_DATE', 'WBANNO']).expand_dims('WBANNO')
uscrn_ds
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89, WBANNO: 1)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
      * WBANNO                   (WBANNO) int64 26656
    Data variables:
        T_DAILY_MAX              (WBANNO, LST_DATE) float64 0.7 -12.3 ... 0.0 0.0
        T_DAILY_MIN              (WBANNO, LST_DATE) float64 -12.4 -17.0 ... -2.5
        T_DAILY_MEAN             (WBANNO, LST_DATE) float64 -5.8 -14.7 ... -2.3 -1.3
        T_DAILY_AVG              (WBANNO, LST_DATE) float64 -5.1 -14.4 ... -1.4 -0.3
        P_DAILY_CALC             (WBANNO, LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (WBANNO, LST_DATE) float64 0.26 0.68 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (WBANNO, LST_DATE) object 'C' 'C' 'C' ... 'C' 'C'
        SUR_TEMP_DAILY_MAX       (WBANNO, LST_DATE) float64 -1.1 -12.4 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (WBANNO, LST_DATE) float64 -12.4 -21.3 ... -3.6
        SUR_TEMP_DAILY_AVG       (WBANNO, LST_DATE) float64 -5.3 -16.4 ... -0.2 0.1
        RH_DAILY_MAX             (WBANNO, LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (WBANNO, LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (WBANNO, LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_10_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_20_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_50_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_100_DAILY  (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_TEMP_5_DAILY        (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        LONGITUDE                (WBANNO) float64 -164.1
        LATITUDE                 (WBANNO) float64 61.35
        CRX_VN                   (WBANNO) float64 2.515




```python
# List the variables of the dataset
uscrn_ds.data_vars
```




    Data variables:
        T_DAILY_MAX              (WBANNO, LST_DATE) float64 0.7 -12.3 ... 0.0 0.0
        T_DAILY_MIN              (WBANNO, LST_DATE) float64 -12.4 -17.0 ... -2.5
        T_DAILY_MEAN             (WBANNO, LST_DATE) float64 -5.8 -14.7 ... -2.3 -1.3
        T_DAILY_AVG              (WBANNO, LST_DATE) float64 -5.1 -14.4 ... -1.4 -0.3
        P_DAILY_CALC             (WBANNO, LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (WBANNO, LST_DATE) float64 0.26 0.68 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (WBANNO, LST_DATE) object 'C' 'C' 'C' ... 'C' 'C'
        SUR_TEMP_DAILY_MAX       (WBANNO, LST_DATE) float64 -1.1 -12.4 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (WBANNO, LST_DATE) float64 -12.4 -21.3 ... -3.6
        SUR_TEMP_DAILY_AVG       (WBANNO, LST_DATE) float64 -5.3 -16.4 ... -0.2 0.1
        RH_DAILY_MAX             (WBANNO, LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (WBANNO, LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (WBANNO, LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_10_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_20_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_50_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_100_DAILY  (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_TEMP_5_DAILY        (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        LONGITUDE                (WBANNO) float64 -164.1
        LATITUDE                 (WBANNO) float64 61.35
        CRX_VN                   (WBANNO) float64 2.515




```python
# Add units for each variable
for x in uscrn_ds.data_vars:
    uscrn_ds[x].attrs['unit'] = uscrn_header.loc[1, uscrn_header.iloc[0, :]==x].values[0]
```


```python
uscrn_ds
```




    <xarray.Dataset>
    Dimensions:                  (LST_DATE: 89, WBANNO: 1)
    Coordinates:
      * LST_DATE                 (LST_DATE) int64 20190101 20190102 ... 20190330
      * WBANNO                   (WBANNO) int64 26656
    Data variables:
        T_DAILY_MAX              (WBANNO, LST_DATE) float64 0.7 -12.3 ... 0.0 0.0
        T_DAILY_MIN              (WBANNO, LST_DATE) float64 -12.4 -17.0 ... -2.5
        T_DAILY_MEAN             (WBANNO, LST_DATE) float64 -5.8 -14.7 ... -2.3 -1.3
        T_DAILY_AVG              (WBANNO, LST_DATE) float64 -5.1 -14.4 ... -1.4 -0.3
        P_DAILY_CALC             (WBANNO, LST_DATE) float64 0.0 0.0 0.0 ... 0.0 0.0
        SOLARAD_DAILY            (WBANNO, LST_DATE) float64 0.26 0.68 ... 2.02 0.57
        SUR_TEMP_DAILY_TYPE      (WBANNO, LST_DATE) object 'C' 'C' 'C' ... 'C' 'C'
        SUR_TEMP_DAILY_MAX       (WBANNO, LST_DATE) float64 -1.1 -12.4 ... 6.5 3.6
        SUR_TEMP_DAILY_MIN       (WBANNO, LST_DATE) float64 -12.4 -21.3 ... -3.6
        SUR_TEMP_DAILY_AVG       (WBANNO, LST_DATE) float64 -5.3 -16.4 ... -0.2 0.1
        RH_DAILY_MAX             (WBANNO, LST_DATE) float64 97.6 92.8 ... -9.999e+03
        RH_DAILY_MIN             (WBANNO, LST_DATE) float64 79.3 82.0 ... -9.999e+03
        RH_DAILY_AVG             (WBANNO, LST_DATE) float64 92.0 87.4 ... -9.999e+03
        SOIL_MOISTURE_5_DAILY    (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_10_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_20_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_50_DAILY   (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_MOISTURE_100_DAILY  (WBANNO, LST_DATE) float64 -99.0 -99.0 ... -99.0
        SOIL_TEMP_5_DAILY        (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_10_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_20_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_50_DAILY       (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        SOIL_TEMP_100_DAILY      (WBANNO, LST_DATE) float64 -9.999e+03 ... -9.999e+03
        LONGITUDE                (WBANNO) float64 -164.1
        LATITUDE                 (WBANNO) float64 61.35
        CRX_VN                   (WBANNO) float64 2.515




```python
# Save the dataset to a netcdf file
uscrn_ds.to_netcdf(str(uscrn_ds.WBANNO.values[0]) + '_' + str(uscrn_ds.LST_DATE.values[0]) + '_' + str(uscrn_ds.LST_DATE.values[-1]) + '.nc')
```


```python
!ls -lrth
```

    total 232
    -rw-r--r--  1 zeng  staff    61K Mar 31 21:11 xarray_notes.ipynb
    -rw-r--r--  1 zeng  staff    49K Mar 31 21:11 26656_20190101_20190330.nc

