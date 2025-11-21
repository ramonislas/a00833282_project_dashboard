import streamlit as st 
import pandas as pd
import datetime
import plotly.express as px

'''
Im thinking on a filter that selects the date range and the country or continent
on the top maybe a map and if you select a country you can visualize the number of cummulative deaths
in the bottom maybe bar graphs comparing the cumulative deaths up to that point
'''

df_death = pd.read_csv('WHO-COVID-19-global-daily-data.csv')
df_death['Date_reported'] = pd.to_datetime(df_death['Date_reported'])

st.title('COVID-19')

options = st.multiselect(
    "Select Country",
    df_death['Country'].unique().tolist(),
    default=[],
)

first_date = df_death["Date_reported"].min().date()
last_date = df_death["Date_reported"].max().date()

selected_dates = st.date_input(
    "Select your date range",
    (first_date, last_date),        # default range
    min_value=first_date,           # limit backward selection
    max_value=last_date,            # limit forward selection
    format="YYYY-MM-DD"
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    st.error("Please select a start and end date.")
    st.stop()

df_filtered = df_death[
    (df_death["Country"].isin(options)) &
    (df_death["Date_reported"].dt.date >= start_date) &
    (df_death["Date_reported"].dt.date <= end_date)
]

df_filtered

column1, column2 = st.columns(2)

with column1:
    fig = px.bar(df_filtered, x='Country', y='Cumulative_deaths')
    st.plotly_chart(fig)

with column2:
    fig = px.bar(df_filtered, x='Country', y='Cumulative_cases')
    st.plotly_chart(fig)