import streamlit as st 
import pandas as pd 
import numpy as np 
from prophet import Prophet
from prophet.diagnostics import performance_metrics 
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Automated Time Series Forecasting')

"""
This data app uses Facebook open source prophet library to autmaticllly
    """

df =st.file_uploader('Import the time series csv file here. Coumns name : ds and y')

if df is not None:
    data = pd.read_csv(df)
    data['ds']=pd.to_datetime(data['ds'], errors='coerce')
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)
    
### step 2: select Forecast Horizon
""" 
Keep in mind that forecasts become less accurate with larger forecasting
"""
periods_input = st.number_input('how many periods would you like to forecast into the future?',
min_value=1, max_value=365)

if df is not None:
    m=Prophet()
    m.fit(data)
'''
# Step3 L Visualize Forecast Data
The below visual shows future preicted values. 'yhat' is the prediction

'''

if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast=m.predict(future)
    fcst=forecast[['ds','yhat','yhat_lower','yhat_upper']]

    fcst_filtered=fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted 
    """
    fig1=m.plot(forecast)
    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted value
    """
    fig2=m.plot_components(forecast)
    st.write(fig2)

"""
Step 4 Download the Forecast Date 
The below link allows you to download the newly created forecast
"""

# if df is not None:
#     csv_exp=fcst_filtered.to_csv(index=False)
#     b64=base64.b64decode(csv_exp.encode()).decode()
#     href=f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>(right-click and save as ** &lt;forecast_name&gt;.csv**)'
#     st.markdown(href,unsafe_allow_html=True)
