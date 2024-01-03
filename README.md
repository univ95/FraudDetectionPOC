# Insurance Fraud Detection Machine Learning Model (PoC)

### This project will aim to preprocess insurance data, feed it to a machine learning model and then check which statistical model is providing the best accuracy in finding the true positive frauds using a bar chart. We will then use this as a basis for a tool which will learn data from the client's data lake/warehouse and then produce a smaller list of likely fraud claims for the manual users to check instead of having to go through every claim. We have used 11 different algorithms to find out which one is providing the most training and testing accuracy.


```python
import pandas as pd
import numpy as np
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
```


```python
import matplotlib.pyplot as plt
```


```python
df = pd.read_csv("insurance_claims.csv")
```


```python
df.head(5)
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
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_number</th>
      <th>policy_bind_date</th>
      <th>policy_state</th>
      <th>policy_csl</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_zip</th>
      <th>...</th>
      <th>police_report_available</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>auto_make</th>
      <th>auto_model</th>
      <th>auto_year</th>
      <th>fraud_reported</th>
      <th>_c39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>48</td>
      <td>521585</td>
      <td>2014-10-17</td>
      <td>OH</td>
      <td>250/500</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>466132</td>
      <td>...</td>
      <td>YES</td>
      <td>71610</td>
      <td>6510</td>
      <td>13020</td>
      <td>52080</td>
      <td>Saab</td>
      <td>92x</td>
      <td>2004</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>42</td>
      <td>342868</td>
      <td>2006-06-27</td>
      <td>IN</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>468176</td>
      <td>...</td>
      <td>?</td>
      <td>5070</td>
      <td>780</td>
      <td>780</td>
      <td>3510</td>
      <td>Mercedes</td>
      <td>E400</td>
      <td>2007</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>29</td>
      <td>687698</td>
      <td>2000-09-06</td>
      <td>OH</td>
      <td>100/300</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>430632</td>
      <td>...</td>
      <td>NO</td>
      <td>34650</td>
      <td>7700</td>
      <td>3850</td>
      <td>23100</td>
      <td>Dodge</td>
      <td>RAM</td>
      <td>2007</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>41</td>
      <td>227811</td>
      <td>1990-05-25</td>
      <td>IL</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>608117</td>
      <td>...</td>
      <td>NO</td>
      <td>63400</td>
      <td>6340</td>
      <td>6340</td>
      <td>50720</td>
      <td>Chevrolet</td>
      <td>Tahoe</td>
      <td>2014</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>44</td>
      <td>367455</td>
      <td>2014-06-06</td>
      <td>IL</td>
      <td>500/1000</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>610706</td>
      <td>...</td>
      <td>NO</td>
      <td>6500</td>
      <td>1300</td>
      <td>650</td>
      <td>4550</td>
      <td>Accura</td>
      <td>RSX</td>
      <td>2009</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
# we can see some missing values denoted by '?' so lets replace missing values with np.nan

df.replace('?', np.nan, inplace = True)
```


```python
df.head(5)
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
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_number</th>
      <th>policy_bind_date</th>
      <th>policy_state</th>
      <th>policy_csl</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_zip</th>
      <th>...</th>
      <th>police_report_available</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>auto_make</th>
      <th>auto_model</th>
      <th>auto_year</th>
      <th>fraud_reported</th>
      <th>_c39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>48</td>
      <td>521585</td>
      <td>2014-10-17</td>
      <td>OH</td>
      <td>250/500</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>466132</td>
      <td>...</td>
      <td>YES</td>
      <td>71610</td>
      <td>6510</td>
      <td>13020</td>
      <td>52080</td>
      <td>Saab</td>
      <td>92x</td>
      <td>2004</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>42</td>
      <td>342868</td>
      <td>2006-06-27</td>
      <td>IN</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>468176</td>
      <td>...</td>
      <td>NaN</td>
      <td>5070</td>
      <td>780</td>
      <td>780</td>
      <td>3510</td>
      <td>Mercedes</td>
      <td>E400</td>
      <td>2007</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>29</td>
      <td>687698</td>
      <td>2000-09-06</td>
      <td>OH</td>
      <td>100/300</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>430632</td>
      <td>...</td>
      <td>NO</td>
      <td>34650</td>
      <td>7700</td>
      <td>3850</td>
      <td>23100</td>
      <td>Dodge</td>
      <td>RAM</td>
      <td>2007</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>41</td>
      <td>227811</td>
      <td>1990-05-25</td>
      <td>IL</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>608117</td>
      <td>...</td>
      <td>NO</td>
      <td>63400</td>
      <td>6340</td>
      <td>6340</td>
      <td>50720</td>
      <td>Chevrolet</td>
      <td>Tahoe</td>
      <td>2014</td>
      <td>Y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>44</td>
      <td>367455</td>
      <td>2014-06-06</td>
      <td>IL</td>
      <td>500/1000</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>610706</td>
      <td>...</td>
      <td>NO</td>
      <td>6500</td>
      <td>1300</td>
      <td>650</td>
      <td>4550</td>
      <td>Accura</td>
      <td>RSX</td>
      <td>2009</td>
      <td>N</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
df.describe()
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
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_number</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_zip</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>auto_year</th>
      <th>_c39</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>203.954000</td>
      <td>38.948000</td>
      <td>546238.648000</td>
      <td>1136.000000</td>
      <td>1256.406150</td>
      <td>1.101000e+06</td>
      <td>501214.488000</td>
      <td>25126.100000</td>
      <td>-26793.700000</td>
      <td>11.644000</td>
      <td>1.83900</td>
      <td>0.992000</td>
      <td>1.487000</td>
      <td>52761.94000</td>
      <td>7433.420000</td>
      <td>7399.570000</td>
      <td>37928.950000</td>
      <td>2005.103000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>115.113174</td>
      <td>9.140287</td>
      <td>257063.005276</td>
      <td>611.864673</td>
      <td>244.167395</td>
      <td>2.297407e+06</td>
      <td>71701.610941</td>
      <td>27872.187708</td>
      <td>28104.096686</td>
      <td>6.951373</td>
      <td>1.01888</td>
      <td>0.820127</td>
      <td>1.111335</td>
      <td>26401.53319</td>
      <td>4880.951853</td>
      <td>4824.726179</td>
      <td>18886.252893</td>
      <td>6.015861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>100804.000000</td>
      <td>500.000000</td>
      <td>433.330000</td>
      <td>-1.000000e+06</td>
      <td>430104.000000</td>
      <td>0.000000</td>
      <td>-111100.000000</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>100.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>70.000000</td>
      <td>1995.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>115.750000</td>
      <td>32.000000</td>
      <td>335980.250000</td>
      <td>500.000000</td>
      <td>1089.607500</td>
      <td>0.000000e+00</td>
      <td>448404.500000</td>
      <td>0.000000</td>
      <td>-51500.000000</td>
      <td>6.000000</td>
      <td>1.00000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>41812.50000</td>
      <td>4295.000000</td>
      <td>4445.000000</td>
      <td>30292.500000</td>
      <td>2000.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>199.500000</td>
      <td>38.000000</td>
      <td>533135.000000</td>
      <td>1000.000000</td>
      <td>1257.200000</td>
      <td>0.000000e+00</td>
      <td>466445.500000</td>
      <td>0.000000</td>
      <td>-23250.000000</td>
      <td>12.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>58055.00000</td>
      <td>6775.000000</td>
      <td>6750.000000</td>
      <td>42100.000000</td>
      <td>2005.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>276.250000</td>
      <td>44.000000</td>
      <td>759099.750000</td>
      <td>2000.000000</td>
      <td>1415.695000</td>
      <td>0.000000e+00</td>
      <td>603251.000000</td>
      <td>51025.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>70592.50000</td>
      <td>11305.000000</td>
      <td>10885.000000</td>
      <td>50822.500000</td>
      <td>2010.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>479.000000</td>
      <td>64.000000</td>
      <td>999435.000000</td>
      <td>2000.000000</td>
      <td>2047.590000</td>
      <td>1.000000e+07</td>
      <td>620962.000000</td>
      <td>100500.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>4.00000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>114920.00000</td>
      <td>21450.000000</td>
      <td>23670.000000</td>
      <td>79560.000000</td>
      <td>2015.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis and Data Pre-processing


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 40 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   months_as_customer           1000 non-null   int64  
     1   age                          1000 non-null   int64  
     2   policy_number                1000 non-null   int64  
     3   policy_bind_date             1000 non-null   object 
     4   policy_state                 1000 non-null   object 
     5   policy_csl                   1000 non-null   object 
     6   policy_deductable            1000 non-null   int64  
     7   policy_annual_premium        1000 non-null   float64
     8   umbrella_limit               1000 non-null   int64  
     9   insured_zip                  1000 non-null   int64  
     10  insured_sex                  1000 non-null   object 
     11  insured_education_level      1000 non-null   object 
     12  insured_occupation           1000 non-null   object 
     13  insured_hobbies              1000 non-null   object 
     14  insured_relationship         1000 non-null   object 
     15  capital-gains                1000 non-null   int64  
     16  capital-loss                 1000 non-null   int64  
     17  incident_date                1000 non-null   object 
     18  incident_type                1000 non-null   object 
     19  collision_type               822 non-null    object 
     20  incident_severity            1000 non-null   object 
     21  authorities_contacted        909 non-null    object 
     22  incident_state               1000 non-null   object 
     23  incident_city                1000 non-null   object 
     24  incident_location            1000 non-null   object 
     25  incident_hour_of_the_day     1000 non-null   int64  
     26  number_of_vehicles_involved  1000 non-null   int64  
     27  property_damage              640 non-null    object 
     28  bodily_injuries              1000 non-null   int64  
     29  witnesses                    1000 non-null   int64  
     30  police_report_available      657 non-null    object 
     31  total_claim_amount           1000 non-null   int64  
     32  injury_claim                 1000 non-null   int64  
     33  property_claim               1000 non-null   int64  
     34  vehicle_claim                1000 non-null   int64  
     35  auto_make                    1000 non-null   object 
     36  auto_model                   1000 non-null   object 
     37  auto_year                    1000 non-null   int64  
     38  fraud_reported               1000 non-null   object 
     39  _c39                         0 non-null      float64
    dtypes: float64(2), int64(17), object(21)
    memory usage: 312.6+ KB
    


```python
# missing values
df.isna().sum()
```




    months_as_customer                0
    age                               0
    policy_number                     0
    policy_bind_date                  0
    policy_state                      0
    policy_csl                        0
    policy_deductable                 0
    policy_annual_premium             0
    umbrella_limit                    0
    insured_zip                       0
    insured_sex                       0
    insured_education_level           0
    insured_occupation                0
    insured_hobbies                   0
    insured_relationship              0
    capital-gains                     0
    capital-loss                      0
    incident_date                     0
    incident_type                     0
    collision_type                  178
    incident_severity                 0
    authorities_contacted            91
    incident_state                    0
    incident_city                     0
    incident_location                 0
    incident_hour_of_the_day          0
    number_of_vehicles_involved       0
    property_damage                 360
    bodily_injuries                   0
    witnesses                         0
    police_report_available         343
    total_claim_amount                0
    injury_claim                      0
    property_claim                    0
    vehicle_claim                     0
    auto_make                         0
    auto_model                        0
    auto_year                         0
    fraud_reported                    0
    _c39                           1000
    dtype: int64




```python
pip install missingno
```

    Requirement already satisfied: missingno in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (0.5.2)
    Requirement already satisfied: numpy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from missingno) (1.26.2)
    Requirement already satisfied: matplotlib in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from missingno) (3.8.1)
    Requirement already satisfied: scipy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from missingno) (1.11.3)
    Requirement already satisfied: seaborn in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from missingno) (0.13.0)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (4.44.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (23.2)
    Requirement already satisfied: pillow>=8 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (10.1.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from matplotlib->missingno) (2.8.2)
    Requirement already satisfied: pandas>=1.2 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from seaborn->missingno) (2.1.3)
    Requirement already satisfied: pytz>=2020.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from pandas>=1.2->seaborn->missingno) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from pandas>=1.2->seaborn->missingno) (2023.3)
    Requirement already satisfied: six>=1.5 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from python-dateutil>=2.7->matplotlib->missingno) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import missingno as msno
```


```python
msno.bar(df)
plt.show()
```


    
![png](output_15_0.png)
    



```python
df.collision_type.unique()
```




    array(['Side Collision', nan, 'Rear Collision', 'Front Collision'],
          dtype=object)




```python
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
```


```python
df.collision_type.unique()
```




    array(['Side Collision', 'Rear Collision', 'Front Collision'],
          dtype=object)




```python
df.property_damage.unique()
```




    array(['YES', nan, 'NO'], dtype=object)




```python
df.property_damage.mode()
```




    0    NO
    Name: property_damage, dtype: object




```python
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
```


```python
df.property_damage.unique()
```




    array(['YES', 'NO'], dtype=object)




```python
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
```


```python
df.police_report_available.unique()
```




    array(['YES', 'NO'], dtype=object)




```python
df.isna().sum()
```




    months_as_customer                0
    age                               0
    policy_number                     0
    policy_bind_date                  0
    policy_state                      0
    policy_csl                        0
    policy_deductable                 0
    policy_annual_premium             0
    umbrella_limit                    0
    insured_zip                       0
    insured_sex                       0
    insured_education_level           0
    insured_occupation                0
    insured_hobbies                   0
    insured_relationship              0
    capital-gains                     0
    capital-loss                      0
    incident_date                     0
    incident_type                     0
    collision_type                    0
    incident_severity                 0
    authorities_contacted            91
    incident_state                    0
    incident_city                     0
    incident_location                 0
    incident_hour_of_the_day          0
    number_of_vehicles_involved       0
    property_damage                   0
    bodily_injuries                   0
    witnesses                         0
    police_report_available           0
    total_claim_amount                0
    injury_claim                      0
    property_claim                    0
    vehicle_claim                     0
    auto_make                         0
    auto_model                        0
    auto_year                         0
    fraud_reported                    0
    _c39                           1000
    dtype: int64




```python
df.authorities_contacted.unique()
```




    array(['Police', nan, 'Fire', 'Other', 'Ambulance'], dtype=object)




```python
df['authorities_contacted'] = df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0])
```


```python
df.isna().sum()
```




    months_as_customer                0
    age                               0
    policy_number                     0
    policy_bind_date                  0
    policy_state                      0
    policy_csl                        0
    policy_deductable                 0
    policy_annual_premium             0
    umbrella_limit                    0
    insured_zip                       0
    insured_sex                       0
    insured_education_level           0
    insured_occupation                0
    insured_hobbies                   0
    insured_relationship              0
    capital-gains                     0
    capital-loss                      0
    incident_date                     0
    incident_type                     0
    collision_type                    0
    incident_severity                 0
    authorities_contacted             0
    incident_state                    0
    incident_city                     0
    incident_location                 0
    incident_hour_of_the_day          0
    number_of_vehicles_involved       0
    property_damage                   0
    bodily_injuries                   0
    witnesses                         0
    police_report_available           0
    total_claim_amount                0
    injury_claim                      0
    property_claim                    0
    vehicle_claim                     0
    auto_make                         0
    auto_model                        0
    auto_year                         0
    fraud_reported                    0
    _c39                           1000
    dtype: int64




```python
# heatmap

plt.figure(figsize = (18, 12))

corr = df.corr(method='pearson', min_periods=1, numeric_only=True)

sns.heatmap(data = corr, annot = True, fmt = '.2g', linewidth = 1)
plt.show()
```


    
![png](output_29_0.png)
    



```python
df.nunique()
```




    months_as_customer              391
    age                              46
    policy_number                  1000
    policy_bind_date                951
    policy_state                      3
    policy_csl                        3
    policy_deductable                 3
    policy_annual_premium           991
    umbrella_limit                   11
    insured_zip                     995
    insured_sex                       2
    insured_education_level           7
    insured_occupation               14
    insured_hobbies                  20
    insured_relationship              6
    capital-gains                   338
    capital-loss                    354
    incident_date                    60
    incident_type                     4
    collision_type                    3
    incident_severity                 4
    authorities_contacted             4
    incident_state                    7
    incident_city                     7
    incident_location              1000
    incident_hour_of_the_day         24
    number_of_vehicles_involved       4
    property_damage                   2
    bodily_injuries                   3
    witnesses                         4
    police_report_available           2
    total_claim_amount              763
    injury_claim                    638
    property_claim                  626
    vehicle_claim                   726
    auto_make                        14
    auto_model                       39
    auto_year                        21
    fraud_reported                    2
    _c39                              0
    dtype: int64




```python
# dropping columns which are not necessary for prediction

to_drop = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
           'incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year', '_c39']

df.drop(to_drop, inplace = True, axis = 1)
```


```python
df.head()
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
      <th>months_as_customer</th>
      <th>age</th>
      <th>policy_csl</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_sex</th>
      <th>insured_education_level</th>
      <th>insured_occupation</th>
      <th>insured_relationship</th>
      <th>...</th>
      <th>number_of_vehicles_involved</th>
      <th>property_damage</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>police_report_available</th>
      <th>total_claim_amount</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>fraud_reported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>48</td>
      <td>250/500</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>MALE</td>
      <td>MD</td>
      <td>craft-repair</td>
      <td>husband</td>
      <td>...</td>
      <td>1</td>
      <td>YES</td>
      <td>1</td>
      <td>2</td>
      <td>YES</td>
      <td>71610</td>
      <td>6510</td>
      <td>13020</td>
      <td>52080</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>42</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>MALE</td>
      <td>MD</td>
      <td>machine-op-inspct</td>
      <td>other-relative</td>
      <td>...</td>
      <td>1</td>
      <td>NO</td>
      <td>0</td>
      <td>0</td>
      <td>NO</td>
      <td>5070</td>
      <td>780</td>
      <td>780</td>
      <td>3510</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>29</td>
      <td>100/300</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>sales</td>
      <td>own-child</td>
      <td>...</td>
      <td>3</td>
      <td>NO</td>
      <td>2</td>
      <td>3</td>
      <td>NO</td>
      <td>34650</td>
      <td>7700</td>
      <td>3850</td>
      <td>23100</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>41</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>armed-forces</td>
      <td>unmarried</td>
      <td>...</td>
      <td>1</td>
      <td>NO</td>
      <td>1</td>
      <td>2</td>
      <td>NO</td>
      <td>63400</td>
      <td>6340</td>
      <td>6340</td>
      <td>50720</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>44</td>
      <td>500/1000</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>MALE</td>
      <td>Associate</td>
      <td>sales</td>
      <td>unmarried</td>
      <td>...</td>
      <td>1</td>
      <td>NO</td>
      <td>0</td>
      <td>1</td>
      <td>NO</td>
      <td>6500</td>
      <td>1300</td>
      <td>650</td>
      <td>4550</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
# checking for multicollinearity

plt.figure(figsize = (18, 12))

corr = df.corr(method='pearson', min_periods=1, numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()
```


    
![png](output_33_0.png)
    


### From the above plot, we can see that there is high correlation between age and months_as_customer.We will drop the "Age" column. Also there is high correlation between total_claim_amount, injury_claim, property_claim, vehicle_claim. Total claim is the sum of all others. So we will drop the total claim column.


```python
df.drop(columns = ['age', 'total_claim_amount'], inplace = True, axis = 1)
```


```python
df.head()
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
      <th>months_as_customer</th>
      <th>policy_csl</th>
      <th>policy_deductable</th>
      <th>policy_annual_premium</th>
      <th>umbrella_limit</th>
      <th>insured_sex</th>
      <th>insured_education_level</th>
      <th>insured_occupation</th>
      <th>insured_relationship</th>
      <th>capital-gains</th>
      <th>...</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>property_damage</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>police_report_available</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
      <th>fraud_reported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>250/500</td>
      <td>1000</td>
      <td>1406.91</td>
      <td>0</td>
      <td>MALE</td>
      <td>MD</td>
      <td>craft-repair</td>
      <td>husband</td>
      <td>53300</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>YES</td>
      <td>1</td>
      <td>2</td>
      <td>YES</td>
      <td>6510</td>
      <td>13020</td>
      <td>52080</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1197.22</td>
      <td>5000000</td>
      <td>MALE</td>
      <td>MD</td>
      <td>machine-op-inspct</td>
      <td>other-relative</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1</td>
      <td>NO</td>
      <td>0</td>
      <td>0</td>
      <td>NO</td>
      <td>780</td>
      <td>780</td>
      <td>3510</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>100/300</td>
      <td>2000</td>
      <td>1413.14</td>
      <td>5000000</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>sales</td>
      <td>own-child</td>
      <td>35100</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>NO</td>
      <td>2</td>
      <td>3</td>
      <td>NO</td>
      <td>7700</td>
      <td>3850</td>
      <td>23100</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>250/500</td>
      <td>2000</td>
      <td>1415.74</td>
      <td>6000000</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>armed-forces</td>
      <td>unmarried</td>
      <td>48900</td>
      <td>...</td>
      <td>5</td>
      <td>1</td>
      <td>NO</td>
      <td>1</td>
      <td>2</td>
      <td>NO</td>
      <td>6340</td>
      <td>6340</td>
      <td>50720</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>500/1000</td>
      <td>1000</td>
      <td>1583.91</td>
      <td>6000000</td>
      <td>MALE</td>
      <td>Associate</td>
      <td>sales</td>
      <td>unmarried</td>
      <td>66000</td>
      <td>...</td>
      <td>20</td>
      <td>1</td>
      <td>NO</td>
      <td>0</td>
      <td>1</td>
      <td>NO</td>
      <td>1300</td>
      <td>650</td>
      <td>4550</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 25 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   months_as_customer           1000 non-null   int64  
     1   policy_csl                   1000 non-null   object 
     2   policy_deductable            1000 non-null   int64  
     3   policy_annual_premium        1000 non-null   float64
     4   umbrella_limit               1000 non-null   int64  
     5   insured_sex                  1000 non-null   object 
     6   insured_education_level      1000 non-null   object 
     7   insured_occupation           1000 non-null   object 
     8   insured_relationship         1000 non-null   object 
     9   capital-gains                1000 non-null   int64  
     10  capital-loss                 1000 non-null   int64  
     11  incident_type                1000 non-null   object 
     12  collision_type               1000 non-null   object 
     13  incident_severity            1000 non-null   object 
     14  authorities_contacted        1000 non-null   object 
     15  incident_hour_of_the_day     1000 non-null   int64  
     16  number_of_vehicles_involved  1000 non-null   int64  
     17  property_damage              1000 non-null   object 
     18  bodily_injuries              1000 non-null   int64  
     19  witnesses                    1000 non-null   int64  
     20  police_report_available      1000 non-null   object 
     21  injury_claim                 1000 non-null   int64  
     22  property_claim               1000 non-null   int64  
     23  vehicle_claim                1000 non-null   int64  
     24  fraud_reported               1000 non-null   object 
    dtypes: float64(1), int64(12), object(12)
    memory usage: 195.4+ KB
    


```python
# separating the feature and target columns

X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']
```

## Encoding columns with limited data (categorical columns)


```python
# extracting categorical columns
cat_df = X.select_dtypes(include = ['object'])
```


```python
cat_df.head()

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
      <th>policy_csl</th>
      <th>insured_sex</th>
      <th>insured_education_level</th>
      <th>insured_occupation</th>
      <th>insured_relationship</th>
      <th>incident_type</th>
      <th>collision_type</th>
      <th>incident_severity</th>
      <th>authorities_contacted</th>
      <th>property_damage</th>
      <th>police_report_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250/500</td>
      <td>MALE</td>
      <td>MD</td>
      <td>craft-repair</td>
      <td>husband</td>
      <td>Single Vehicle Collision</td>
      <td>Side Collision</td>
      <td>Major Damage</td>
      <td>Police</td>
      <td>YES</td>
      <td>YES</td>
    </tr>
    <tr>
      <th>1</th>
      <td>250/500</td>
      <td>MALE</td>
      <td>MD</td>
      <td>machine-op-inspct</td>
      <td>other-relative</td>
      <td>Vehicle Theft</td>
      <td>Rear Collision</td>
      <td>Minor Damage</td>
      <td>Police</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100/300</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>sales</td>
      <td>own-child</td>
      <td>Multi-vehicle Collision</td>
      <td>Rear Collision</td>
      <td>Minor Damage</td>
      <td>Police</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250/500</td>
      <td>FEMALE</td>
      <td>PhD</td>
      <td>armed-forces</td>
      <td>unmarried</td>
      <td>Single Vehicle Collision</td>
      <td>Front Collision</td>
      <td>Major Damage</td>
      <td>Police</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>500/1000</td>
      <td>MALE</td>
      <td>Associate</td>
      <td>sales</td>
      <td>unmarried</td>
      <td>Vehicle Theft</td>
      <td>Rear Collision</td>
      <td>Minor Damage</td>
      <td>Police</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
</div>




```python
# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")
```

    policy_csl: 
    ['250/500' '100/300' '500/1000']
    
    insured_sex: 
    ['MALE' 'FEMALE']
    
    insured_education_level: 
    ['MD' 'PhD' 'Associate' 'Masters' 'High School' 'College' 'JD']
    
    insured_occupation: 
    ['craft-repair' 'machine-op-inspct' 'sales' 'armed-forces' 'tech-support'
     'prof-specialty' 'other-service' 'priv-house-serv' 'exec-managerial'
     'protective-serv' 'transport-moving' 'handlers-cleaners' 'adm-clerical'
     'farming-fishing']
    
    insured_relationship: 
    ['husband' 'other-relative' 'own-child' 'unmarried' 'wife' 'not-in-family']
    
    incident_type: 
    ['Single Vehicle Collision' 'Vehicle Theft' 'Multi-vehicle Collision'
     'Parked Car']
    
    collision_type: 
    ['Side Collision' 'Rear Collision' 'Front Collision']
    
    incident_severity: 
    ['Major Damage' 'Minor Damage' 'Total Loss' 'Trivial Damage']
    
    authorities_contacted: 
    ['Police' 'Fire' 'Other' 'Ambulance']
    
    property_damage: 
    ['YES' 'NO']
    
    police_report_available: 
    ['YES' 'NO']
    
    


```python
cat_df = pd.get_dummies(cat_df, drop_first = True)
```


```python
# extracting the numerical columns

num_df = X.select_dtypes(include = ['int64'])
```


```python
num_df.head()
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
      <th>months_as_customer</th>
      <th>policy_deductable</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>259</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>-58300</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>6340</td>
      <td>6340</td>
      <td>44380</td>
    </tr>
    <tr>
      <th>894</th>
      <td>45</td>
      <td>500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1000</td>
      <td>1000</td>
      <td>4000</td>
    </tr>
    <tr>
      <th>615</th>
      <td>153</td>
      <td>500</td>
      <td>0</td>
      <td>45600</td>
      <td>-61400</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>14140</td>
      <td>14140</td>
      <td>49490</td>
    </tr>
    <tr>
      <th>449</th>
      <td>239</td>
      <td>500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>6990</td>
      <td>13980</td>
      <td>55920</td>
    </tr>
    <tr>
      <th>661</th>
      <td>330</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>11900</td>
      <td>5950</td>
      <td>41650</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = pd.concat([num_df, cat_df], axis = 1)
```


```python
X.head()
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
      <th>months_as_customer</th>
      <th>policy_deductable</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>injury_claim</th>
      <th>...</th>
      <th>collision_type_Rear Collision</th>
      <th>collision_type_Side Collision</th>
      <th>incident_severity_Minor Damage</th>
      <th>incident_severity_Total Loss</th>
      <th>incident_severity_Trivial Damage</th>
      <th>authorities_contacted_Fire</th>
      <th>authorities_contacted_Other</th>
      <th>authorities_contacted_Police</th>
      <th>property_damage_YES</th>
      <th>police_report_available_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>328</td>
      <td>1000</td>
      <td>0</td>
      <td>53300</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6510</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>228</td>
      <td>2000</td>
      <td>5000000</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>780</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134</td>
      <td>2000</td>
      <td>5000000</td>
      <td>35100</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>7700</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>256</td>
      <td>2000</td>
      <td>6000000</td>
      <td>48900</td>
      <td>-62400</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6340</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>228</td>
      <td>1000</td>
      <td>6000000</td>
      <td>66000</td>
      <td>-46000</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1300</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>




```python
plt.figure(figsize = (25, 20))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(X[col])
        plt.xlabel(col, fontsize = 15)
        
    plotnumber += 1
    
plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


## Outliers Detection


```python
plt.figure(figsize = (20, 15))
plotnumber = 1

for col in X.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X[col])
        plt.xlabel(col, fontsize = 15)
    
    plotnumber += 1
plt.tight_layout()
plt.show()
```


    
![png](output_50_0.png)
    



```python
pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (1.3.2)
    Requirement already satisfied: numpy<2.0,>=1.17.3 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from scikit-learn) (1.26.2)
    Requirement already satisfied: scipy>=1.5.0 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from scikit-learn) (1.11.3)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from scikit-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from scikit-learn) (3.2.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
```


```python
X_train.head()
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
      <th>months_as_customer</th>
      <th>policy_deductable</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>injury_claim</th>
      <th>...</th>
      <th>collision_type_Rear Collision</th>
      <th>collision_type_Side Collision</th>
      <th>incident_severity_Minor Damage</th>
      <th>incident_severity_Total Loss</th>
      <th>incident_severity_Trivial Damage</th>
      <th>authorities_contacted_Fire</th>
      <th>authorities_contacted_Other</th>
      <th>authorities_contacted_Police</th>
      <th>property_damage_YES</th>
      <th>police_report_available_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>259</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>-58300</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>6340</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>894</th>
      <td>45</td>
      <td>500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1000</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>615</th>
      <td>153</td>
      <td>500</td>
      <td>0</td>
      <td>45600</td>
      <td>-61400</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>14140</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>449</th>
      <td>239</td>
      <td>500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>6990</td>
      <td>...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>661</th>
      <td>330</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>11900</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>




```python
num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
```


```python
# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)
```


```python
scaled_num_df = pd.DataFrame(data = scaled_data, columns = num_df.columns, index = X_train.index)
scaled_num_df.head()
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
      <th>months_as_customer</th>
      <th>policy_deductable</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>injury_claim</th>
      <th>property_claim</th>
      <th>vehicle_claim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>0.503619</td>
      <td>1.380800</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>-1.079157</td>
      <td>-1.225683</td>
      <td>1.129489</td>
      <td>-1.220715</td>
      <td>0.453334</td>
      <td>-0.245694</td>
      <td>-0.240629</td>
      <td>0.330195</td>
    </tr>
    <tr>
      <th>894</th>
      <td>-1.338134</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-0.802385</td>
      <td>-0.822393</td>
      <td>-0.003247</td>
      <td>1.362425</td>
      <td>-1.347949</td>
      <td>-1.333627</td>
      <td>-1.815776</td>
    </tr>
    <tr>
      <th>615</th>
      <td>-0.408651</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>0.734253</td>
      <td>-1.188163</td>
      <td>-1.648981</td>
      <td>1.129489</td>
      <td>-0.003247</td>
      <td>-1.364849</td>
      <td>1.364341</td>
      <td>1.355886</td>
      <td>0.601763</td>
    </tr>
    <tr>
      <th>449</th>
      <td>0.331493</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-1.648981</td>
      <td>2.105429</td>
      <td>-1.220715</td>
      <td>1.362425</td>
      <td>-0.111524</td>
      <td>1.323137</td>
      <td>0.943482</td>
    </tr>
    <tr>
      <th>661</th>
      <td>1.114668</td>
      <td>-0.253928</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-0.237988</td>
      <td>-0.822393</td>
      <td>-0.003247</td>
      <td>1.362425</td>
      <td>0.901972</td>
      <td>-0.320455</td>
      <td>0.185111</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train.drop(columns = scaled_num_df.columns, inplace = True, axis=1)
```


```python
X_train = pd.concat([scaled_num_df, X_train*1], axis = 1)
```


```python
X_train.head()
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
      <th>months_as_customer</th>
      <th>policy_deductable</th>
      <th>umbrella_limit</th>
      <th>capital-gains</th>
      <th>capital-loss</th>
      <th>incident_hour_of_the_day</th>
      <th>number_of_vehicles_involved</th>
      <th>bodily_injuries</th>
      <th>witnesses</th>
      <th>injury_claim</th>
      <th>...</th>
      <th>collision_type_Rear Collision</th>
      <th>collision_type_Side Collision</th>
      <th>incident_severity_Minor Damage</th>
      <th>incident_severity_Total Loss</th>
      <th>incident_severity_Trivial Damage</th>
      <th>authorities_contacted_Fire</th>
      <th>authorities_contacted_Other</th>
      <th>authorities_contacted_Police</th>
      <th>property_damage_YES</th>
      <th>police_report_available_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>810</th>
      <td>0.503619</td>
      <td>1.380800</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>-1.079157</td>
      <td>-1.225683</td>
      <td>1.129489</td>
      <td>-1.220715</td>
      <td>0.453334</td>
      <td>-0.245694</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>894</th>
      <td>-1.338134</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-0.802385</td>
      <td>-0.822393</td>
      <td>-0.003247</td>
      <td>1.362425</td>
      <td>-1.347949</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>615</th>
      <td>-0.408651</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>0.734253</td>
      <td>-1.188163</td>
      <td>-1.648981</td>
      <td>1.129489</td>
      <td>-0.003247</td>
      <td>-1.364849</td>
      <td>1.364341</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>0.331493</td>
      <td>-1.071291</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-1.648981</td>
      <td>2.105429</td>
      <td>-1.220715</td>
      <td>1.362425</td>
      <td>-0.111524</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>661</th>
      <td>1.114668</td>
      <td>-0.253928</td>
      <td>-0.476232</td>
      <td>-0.907007</td>
      <td>0.970846</td>
      <td>-0.237988</td>
      <td>-0.822393</td>
      <td>-0.003247</td>
      <td>1.362425</td>
      <td>0.901972</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>



# Support Vector Classifier


```python
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Support Vector Classifier is : 0.8466666666666667
    Test accuracy of Support Vector Classifier is : 0.712
    [[178   0]
     [ 72   0]]
                  precision    recall  f1-score   support
    
               N       0.71      1.00      0.83       178
               Y       0.00      0.00      0.00        72
    
        accuracy                           0.71       250
       macro avg       0.36      0.50      0.42       250
    weighted avg       0.51      0.71      0.59       250
    
    

# K-Nearest Neighbors classifier


```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn_train_acc = accuracy_score(y_train, knn.predict(X_train))
knn_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of KNN is : {knn_train_acc}")
print(f"Test accuracy of KNN is : {knn_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of KNN is : 0.7693333333333333
    Test accuracy of KNN is : 0.712
    [[178   0]
     [ 72   0]]
                  precision    recall  f1-score   support
    
               N       0.71      1.00      0.83       178
               Y       0.00      0.00      0.00        72
    
        accuracy                           0.71       250
       macro avg       0.36      0.50      0.42       250
    weighted avg       0.51      0.71      0.59       250
    
    

# Decision Tree Classifier


```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Decision Tree is : 1.0
    Test accuracy of Decision Tree is : 0.396
    [[83 95]
     [56 16]]
                  precision    recall  f1-score   support
    
               N       0.60      0.47      0.52       178
               Y       0.14      0.22      0.17        72
    
        accuracy                           0.40       250
       macro avg       0.37      0.34      0.35       250
    weighted avg       0.47      0.40      0.42       250
    
    


```python
# hyper parameter tuning

from sklearn.model_selection import GridSearchCV

grid_params = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : [3, 5, 7, 10],
    'min_samples_split' : range(2, 10, 1),
    'min_samples_leaf' : range(2, 10, 1)
}

grid_search = GridSearchCV(dtc, grid_params, cv = 5, n_jobs = -1, verbose = 1)
grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 512 candidates, totalling 2560 fits
    




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(), n_jobs=-1,
             param_grid={&#x27;criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;max_depth&#x27;: [3, 5, 7, 10],
                         &#x27;min_samples_leaf&#x27;: range(2, 10),
                         &#x27;min_samples_split&#x27;: range(2, 10)},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(), n_jobs=-1,
             param_grid={&#x27;criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;max_depth&#x27;: [3, 5, 7, 10],
                         &#x27;min_samples_leaf&#x27;: range(2, 10),
                         &#x27;min_samples_split&#x27;: range(2, 10)},
             verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div></div></div></div></div></div>




```python
# best parameters and best score

print(grid_search.best_params_)
print(grid_search.best_score_)
```

    {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 7, 'min_samples_split': 2}
    0.8160000000000001
    


```python
# best estimator 

dtc = grid_search.best_estimator_

y_pred = dtc.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))
dtc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Decision Tree is : 0.8173333333333334
    Test accuracy of Decision Tree is : 0.76
    [[137  41]
     [ 19  53]]
                  precision    recall  f1-score   support
    
               N       0.88      0.77      0.82       178
               Y       0.56      0.74      0.64        72
    
        accuracy                           0.76       250
       macro avg       0.72      0.75      0.73       250
    weighted avg       0.79      0.76      0.77       250
    
    

# Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
rand_clf.fit(X_train, y_train)

y_pred = rand_clf.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Random Forest is : 0.9626666666666667
    Test accuracy of Random Forest is : 0.752
    [[168  10]
     [ 52  20]]
                  precision    recall  f1-score   support
    
               N       0.76      0.94      0.84       178
               Y       0.67      0.28      0.39        72
    
        accuracy                           0.75       250
       macro avg       0.72      0.61      0.62       250
    weighted avg       0.74      0.75      0.71       250
    
    

# Ada Boost Classifier


```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)

parameters = {
    'n_estimators' : [50, 70, 90, 120, 180, 200],
    'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
    'algorithm' : ['SAMME', 'SAMME.R']
}

grid_search = GridSearchCV(ada, parameters, n_jobs = -1, cv = 5, verbose = 1)
grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 60 candidates, totalling 300 fits
    




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                                                min_samples_leaf=7)),
             n_jobs=-1,
             param_grid={&#x27;algorithm&#x27;: [&#x27;SAMME&#x27;, &#x27;SAMME.R&#x27;],
                         &#x27;learning_rate&#x27;: [0.001, 0.01, 0.1, 1, 10],
                         &#x27;n_estimators&#x27;: [50, 70, 90, 120, 180, 200]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                                                min_samples_leaf=7)),
             n_jobs=-1,
             param_grid={&#x27;algorithm&#x27;: [&#x27;SAMME&#x27;, &#x27;SAMME.R&#x27;],
                         &#x27;learning_rate&#x27;: [0.001, 0.01, 0.1, 1, 10],
                         &#x27;n_estimators&#x27;: [50, 70, 90, 120, 180, 200]},
             verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                         min_samples_leaf=7))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">base_estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=7)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=7)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
# best parameter and best score

print(grid_search.best_params_)
print(grid_search.best_score_)
```

    {'algorithm': 'SAMME', 'learning_rate': 10, 'n_estimators': 50}
    0.8160000000000001
    


```python
# best estimator 

ada = grid_search.best_estimator_

y_pred = ada.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

ada_train_acc = accuracy_score(y_train, ada.predict(X_train))
ada_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Ada Boost is : {ada_train_acc}")
print(f"Test accuracy of Ada Boost is : {ada_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Ada Boost is : 0.8173333333333334
    Test accuracy of Ada Boost is : 0.76
    [[137  41]
     [ 19  53]]
                  precision    recall  f1-score   support
    
               N       0.88      0.77      0.82       178
               Y       0.56      0.74      0.64        72
    
        accuracy                           0.76       250
       macro avg       0.72      0.75      0.73       250
    weighted avg       0.79      0.76      0.77       250
    
    

# Gradient Boost Classifier


```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of gradient boosting classifier

gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"Training Accuracy of Gradient Boosting Classifier is {accuracy_score(y_train, gb.predict(X_train))}")
print(f"Test Accuracy of Gradient Boosting Classifier is {gb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, gb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, gb.predict(X_test))}")
```

    Training Accuracy of Gradient Boosting Classifier is 0.9386666666666666
    Test Accuracy of Gradient Boosting Classifier is 0.388 
    
    Confusion Matrix :- 
    [[ 33 145]
     [  8  64]]
    
    Classification Report :- 
                   precision    recall  f1-score   support
    
               N       0.80      0.19      0.30       178
               Y       0.31      0.89      0.46        72
    
        accuracy                           0.39       250
       macro avg       0.56      0.54      0.38       250
    weighted avg       0.66      0.39      0.35       250
    
    

# Stochastic Gradient Boosting


```python
sgb = GradientBoostingClassifier(subsample = 0.90, max_features = 0.70)
sgb.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of stochastic gradient boosting classifier

sgb_acc = accuracy_score(y_test, sgb.predict(X_test))

print(f"Training Accuracy of Stochastic Gradient Boosting is {accuracy_score(y_train, sgb.predict(X_train))}")
print(f"Test Accuracy of Stochastic Gradient Boosting is {sgb_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, sgb.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, sgb.predict(X_test))}")
```

    Training Accuracy of Stochastic Gradient Boosting is 0.9453333333333334
    Test Accuracy of Stochastic Gradient Boosting is 0.692 
    
    Confusion Matrix :- 
    [[130  48]
     [ 29  43]]
    
    Classification Report :- 
                   precision    recall  f1-score   support
    
               N       0.82      0.73      0.77       178
               Y       0.47      0.60      0.53        72
    
        accuracy                           0.69       250
       macro avg       0.65      0.66      0.65       250
    weighted avg       0.72      0.69      0.70       250
    
    

# XgBoost Classifier


```python
pip install xgboost
```

    Requirement already satisfied: xgboost in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (2.0.2)
    Requirement already satisfied: numpy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from xgboost) (1.26.2)
    Requirement already satisfied: scipy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from xgboost) (1.11.3)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_pred
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 1, 0, 0, 0, 0])




```python
# accuracy_score, confusion_matrix and classification_report
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)

xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
xgb_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of XgBoost is : {xgb_train_acc}")
print(f"Test accuracy of XgBoost is : {xgb_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of XgBoost is : 1.0
    Test accuracy of XgBoost is : 0.736
    [[163  15]
     [ 51  21]]
                  precision    recall  f1-score   support
    
               0       0.76      0.92      0.83       178
               1       0.58      0.29      0.39        72
    
        accuracy                           0.74       250
       macro avg       0.67      0.60      0.61       250
    weighted avg       0.71      0.74      0.70       250
    
    


```python
param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 10, 1)}

grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)
grid_search.fit(X_train, y_train)
```

    Fitting 5 folds for each of 60 candidates, totalling 300 fits
    




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                                                min_samples_leaf=7)),
             n_jobs=-1,
             param_grid={&#x27;algorithm&#x27;: [&#x27;SAMME&#x27;, &#x27;SAMME.R&#x27;],
                         &#x27;learning_rate&#x27;: [0.001, 0.01, 0.1, 1, 10],
                         &#x27;n_estimators&#x27;: [50, 70, 90, 120, 180, 200]},
             verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                                                min_samples_leaf=7)),
             n_jobs=-1,
             param_grid={&#x27;algorithm&#x27;: [&#x27;SAMME&#x27;, &#x27;SAMME.R&#x27;],
                         &#x27;learning_rate&#x27;: [0.001, 0.01, 0.1, 1, 10],
                         &#x27;n_estimators&#x27;: [50, 70, 90, 120, 180, 200]},
             verbose=1)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3,
                                                         min_samples_leaf=7))</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">base_estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=7)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=3, min_samples_leaf=7)</pre></div></div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
# best estimator 

xgb = grid_search.best_estimator_

y_pred = xgb.predict(X_test)
```


```python
# accuracy_score, confusion_matrix and classification_report

xgb_train_acc = accuracy_score(y_train, xgb.predict(X_train))
xgb_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of XgBoost is : {xgb_train_acc}")
print(f"Test accuracy of XgBoost is : {xgb_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of XgBoost is : 0.8173333333333334
    Test accuracy of XgBoost is : 0.76
    [[137  41]
     [ 19  53]]
                  precision    recall  f1-score   support
    
               0       0.88      0.77      0.82       178
               1       0.56      0.74      0.64        72
    
        accuracy                           0.76       250
       macro avg       0.72      0.75      0.73       250
    weighted avg       0.79      0.76      0.77       250
    
    

# Cat Boost Classifier

work in progress

# Extra Trees Classifier


```python
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, etc.predict(X_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")
```

    Training Accuracy of Extra Trees Classifier is 1.0
    Test Accuracy of Extra Trees Classifier is 0.756 
    
    Confusion Matrix :- 
    [[157  21]
     [ 40  32]]
    
    Classification Report :- 
                   precision    recall  f1-score   support
    
               0       0.80      0.88      0.84       178
               1       0.60      0.44      0.51        72
    
        accuracy                           0.76       250
       macro avg       0.70      0.66      0.67       250
    weighted avg       0.74      0.76      0.74       250
    
    

# LGBM Classifier


```python
pip install lightgbm
```

    Collecting lightgbm
      Downloading lightgbm-4.1.0-py3-none-win_amd64.whl.metadata (19 kB)
    Requirement already satisfied: numpy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from lightgbm) (1.26.2)
    Requirement already satisfied: scipy in c:\users\indrajit.ghosh\appdata\local\programs\python\python312\lib\site-packages (from lightgbm) (1.11.3)
    Downloading lightgbm-4.1.0-py3-none-win_amd64.whl (1.3 MB)
       ---------------------------------------- 0.0/1.3 MB ? eta -:--:--
       ---------------------------------------- 0.0/1.3 MB ? eta -:--:--
       - -------------------------------------- 0.0/1.3 MB 495.5 kB/s eta 0:00:03
       ---- ----------------------------------- 0.1/1.3 MB 1.2 MB/s eta 0:00:01
       --------- ------------------------------ 0.3/1.3 MB 1.9 MB/s eta 0:00:01
       -------------- ------------------------- 0.5/1.3 MB 2.2 MB/s eta 0:00:01
       ------------------ --------------------- 0.6/1.3 MB 2.4 MB/s eta 0:00:01
       ---------------------- ----------------- 0.7/1.3 MB 2.4 MB/s eta 0:00:01
       --------------------------- ------------ 0.9/1.3 MB 2.6 MB/s eta 0:00:01
       ------------------------------- -------- 1.0/1.3 MB 2.6 MB/s eta 0:00:01
       ----------------------------------- ---- 1.2/1.3 MB 2.6 MB/s eta 0:00:01
       -------------------------------------- - 1.2/1.3 MB 2.6 MB/s eta 0:00:01
       ---------------------------------------- 1.3/1.3 MB 2.5 MB/s eta 0:00:00
    Installing collected packages: lightgbm
    Successfully installed lightgbm-4.1.0
    Note: you may need to restart the kernel to use updated packages.
    


```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(learning_rate = 1)
lgbm.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of lgbm classifier

lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

print(f"Training Accuracy of LGBM Classifier is {accuracy_score(y_train, lgbm.predict(X_train))}")
print(f"Test Accuracy of LGBM Classifier is {lgbm_acc} \n")

print(f"{confusion_matrix(y_test, lgbm.predict(X_test))}\n")
print(classification_report(y_test, lgbm.predict(X_test)))
```

    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 175, number of negative: 575
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000207 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1366
    [LightGBM] [Info] Number of data points in the train set: 750, number of used features: 52
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.233333 -> initscore=-1.189584
    [LightGBM] [Info] Start training from score -1.189584
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    Training Accuracy of LGBM Classifier is 1.0
    Test Accuracy of LGBM Classifier is 0.732 
    
    [[137  41]
     [ 26  46]]
    
                  precision    recall  f1-score   support
    
               0       0.84      0.77      0.80       178
               1       0.53      0.64      0.58        72
    
        accuracy                           0.73       250
       macro avg       0.68      0.70      0.69       250
    weighted avg       0.75      0.73      0.74       250
    
    

# Voting Classifier


```python
from sklearn.ensemble import VotingClassifier

classifiers = [('Support Vector Classifier', svc), ('KNN', knn),  ('Decision Tree', dtc), ('Random Forest', rand_clf),
               ('Ada Boost', ada), ('XGboost', xgb), ('Gradient Boosting Classifier', gb), ('SGB', sgb),
            ('Extra Trees Classifier', etc), ('LGBM', lgbm)]

vc = VotingClassifier(estimators = classifiers)
vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)
```

    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Number of positive: 175, number of negative: 575
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000213 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1366
    [LightGBM] [Info] Number of data points in the train set: 750, number of used features: 52
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.233333 -> initscore=-1.189584
    [LightGBM] [Info] Start training from score -1.189584
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    


```python
# accuracy_score, confusion_matrix and classification_report

vc_train_acc = accuracy_score(y_train, vc.predict(X_train))
vc_test_acc = accuracy_score(y_test, y_pred)

print(f"Training accuracy of Voting Classifier is : {vc_train_acc}")
print(f"Test accuracy of Voting Classifier is : {vc_test_acc}")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    Training accuracy of Voting Classifier is : 0.9333333333333333
    Test accuracy of Voting Classifier is : 0.8
    [[154  24]
     [ 26  46]]
                  precision    recall  f1-score   support
    
               0       0.86      0.87      0.86       178
               1       0.66      0.64      0.65        72
    
        accuracy                           0.80       250
       macro avg       0.76      0.75      0.75       250
    weighted avg       0.80      0.80      0.80       250
    
    


```python
models = pd.DataFrame({
    'Model' : ['SVC', 'KNN', 'Decision Tree', 'Random Forest','Ada Boost', 'Gradient Boost', 'SGB', 'Extra Trees', 'LGBM', 'XgBoost', 'Voting Classifier'],
    'Score' : [svc_test_acc, knn_test_acc, dtc_test_acc, rand_clf_test_acc, ada_test_acc, gb_acc, sgb_acc, etc_acc, lgbm_acc, xgb_test_acc, vc_test_acc]
})


models.sort_values(by = 'Score', ascending = False)
models.sort_values(by = 'Score', ascending = False).head(5)
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
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>Voting Classifier</td>
      <td>0.800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree</td>
      <td>0.760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ada Boost</td>
      <td>0.760</td>
    </tr>
    <tr>
      <th>9</th>
      <td>XgBoost</td>
      <td>0.760</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Extra Trees</td>
      <td>0.756</td>
    </tr>
  </tbody>
</table>
</div>




```python
px.bar(data_frame = models.sort_values(by = 'Score', ascending = False).head(5), x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')
```


<div>                            <div id="21d7f42e-4698-4e9e-93cf-c49cb05cdc42" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("21d7f42e-4698-4e9e-93cf-c49cb05cdc42")) {                    Plotly.newPlot(                        "21d7f42e-4698-4e9e-93cf-c49cb05cdc42",                        [{"alignmentgroup":"True","hovertemplate":"Score=%{marker.color}\u003cbr\u003eModel=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":[0.8,0.76,0.76,0.76,0.756],"coloraxis":"coloraxis","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"h","showlegend":false,"textposition":"auto","x":[0.8,0.76,0.76,0.76,0.756],"xaxis":"x","y":["Voting Classifier","Decision Tree","Ada Boost","XgBoost","Extra Trees"],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#f2f5fa"},"error_y":{"color":"#f2f5fa"},"marker":{"line":{"color":"rgb(17,17,17)","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"baxis":{"endlinecolor":"#A2B1C6","gridcolor":"#506784","linecolor":"#506784","minorgridcolor":"#506784","startlinecolor":"#A2B1C6"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"line":{"color":"#283442"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"marker":{"line":{"color":"#283442"}},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#506784"},"line":{"color":"rgb(17,17,17)"}},"header":{"fill":{"color":"#2a3f5f"},"line":{"color":"rgb(17,17,17)"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#f2f5fa","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#f2f5fa"},"geo":{"bgcolor":"rgb(17,17,17)","lakecolor":"rgb(17,17,17)","landcolor":"rgb(17,17,17)","showlakes":true,"showland":true,"subunitcolor":"#506784"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"dark"},"paper_bgcolor":"rgb(17,17,17)","plot_bgcolor":"rgb(17,17,17)","polar":{"angularaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","radialaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"yaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"},"zaxis":{"backgroundcolor":"rgb(17,17,17)","gridcolor":"#506784","gridwidth":2,"linecolor":"#506784","showbackground":true,"ticks":"","zerolinecolor":"#C8D4E3"}},"shapedefaults":{"line":{"color":"#f2f5fa"}},"sliderdefaults":{"bgcolor":"#C8D4E3","bordercolor":"rgb(17,17,17)","borderwidth":1,"tickwidth":0},"ternary":{"aaxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"baxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""},"bgcolor":"rgb(17,17,17)","caxis":{"gridcolor":"#506784","linecolor":"#506784","ticks":""}},"title":{"x":0.05},"updatemenudefaults":{"bgcolor":"#506784","borderwidth":0},"xaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#283442","linecolor":"#506784","ticks":"","title":{"standoff":15},"zerolinecolor":"#283442","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Score"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Model"}},"coloraxis":{"colorbar":{"title":{"text":"Score"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"title":{"text":"Models Comparison"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('21d7f42e-4698-4e9e-93cf-c49cb05cdc42');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


![image.png](141d83e7-6860-4380-a453-6a6d9594200b.png)


```python

```
