import streamlit as st 
import pandas as pd
import datetime
import plotly.express as px

st.set_page_config(layout="wide") # make it full page

df_death = pd.read_csv('final_covid_pop.csv')
df_death['Date_reported'] = pd.to_datetime(df_death['Date_reported']).dt.date

latin_america = ['Mexico','Guatemala','Belize','El Salvador','Honduras','Nicaragua','Costa Rica','Panama','Cuba',
                 'Dominican Republic','Haiti','Puerto Rico', 'Aruba','Bonaire, Sint Eustatius and Saba','Sint Maarten (Dutch part)',
                 'Saint Barthelemy', 'Saint Martin (French part)', 'Martinique', 'Guadeloupe', 'French Guiana','Argentina', 'Bolivia (Plurinational State of)',
                 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay','Peru','Uruguay','Venezuela (Bolivarian Republic of)', 'Latin America', 'Europe']

df_death['New_cases'] = df_death['New_cases'].fillna(0)
df_death['New_deaths'] = df_death['New_deaths'].fillna(0)

df_death = df_death[df_death['Country'].isin(latin_america)]
df_death['New_cases_per'] = (df_death['New_cases'] / df_death['population']) 
df_death['Cum_cases_per'] = (df_death['Cumulative_cases'] / df_death['population'])
df_death['New_deaths_per'] = (df_death['New_deaths'] / df_death['population']) 
df_death['Cum_deaths_per'] = (df_death['Cumulative_deaths'] / df_death['population']) 

st.title('COVID-19 Impact and Vulnerability in Latin America')

st.sidebar.header('Filters')
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

options = st.sidebar.multiselect(
    "Select Country",
    df_death['Country'].unique().tolist(),
    default=['Europe', 'Latin America'],
)

first_date = df_death["Date_reported"].min()
last_date = df_death["Date_reported"].max()

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

selected_dates = st.sidebar.date_input(
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
    (df_death["Date_reported"] >= start_date) &
    (df_death["Date_reported"] <= end_date)
]

df_death2 = df_death[~df_death["Country"].isin(["Europe", "Latin America"])]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
df_full = df_death2.groupby(['Date_reported'], as_index=False).agg(New_deaths=('New_deaths', 'sum'), New_cases=('New_cases', 'sum'))
df_full_country = df_death2.groupby(['Date_reported', 'Country'], as_index=False).agg(New_deaths=('New_deaths', 'sum'), New_cases=('New_cases', 'sum'))
last7_dates = (df_full_country["Date_reported"].sort_values().unique()[-7:])
df_last7_global = df_full_country[df_full_country["Date_reported"].isin(last7_dates)]
df_highest_death = (df_last7_global.groupby("Country", as_index=False).agg(total_7day_deaths=("New_deaths", "sum")).sort_values("total_7day_deaths", ascending=False))
top_country_death = df_highest_death.iloc[0]["Country"]
df_highest_case = (df_last7_global.groupby("Country", as_index=False).agg(total_7day_deaths=("New_cases", "sum")).sort_values("total_7day_deaths", ascending=False))
top_country_case = df_highest_case.iloc[0]["Country"]

with kpi1:
    st.metric(label='New Deaths (WTD)', value=df_full['New_deaths'].iloc[-7:].sum(), border=True)
with kpi2:
    st.metric(label='New Cases (WTD)', value=df_full['New_cases'].iloc[-7:].sum(), border=True)
with kpi3:
    st.metric(label='Deathliest Country (WTD)', value=top_country_death, border=True)
with kpi4:
    st.metric(label='Most Infected Country (WTD)', value=top_country_case, border=True)

df_filtered2 = df_filtered[~df_filtered["Country"].isin(["Europe", "Latin America"])]
fig_map = px.choropleth(
    df_filtered2,
    locations="Country",                #using names
    locationmode="country names",       
    color="Cumulative_deaths",
    hover_name="Country",
    hover_data={
        "Cumulative_deaths": True,
        "Cumulative_cases": True,},
    color_continuous_scale="Reds",
    projection="natural earth",
    title=f"Cumulative COVID Deaths (as of {last_date})"
)

fig_map.update_geos(
    showcoastlines=False,
    showframe=False,
    bgcolor="rgba(0,0,0,0)",   # transparent background
    showland=True,
    landcolor="#1A1A1A",
    showocean=True,
    oceancolor="#000000")

st.plotly_chart(fig_map, use_container_width=True)

column1, column2 = st.columns(2)

df_filtered3 = df_filtered[df_filtered["Date_reported"] == end_date]
with column1:
    fig = px.bar(df_filtered3.sort_values('Cumulative_deaths', ascending=True), x='Cumulative_deaths', y='Country', orientation='h',
                 color='Cumulative_deaths', color_continuous_scale='Reds', #gradient colors
                 title = f'Cumulative Deaths (as of {last_date})',
                 labels={"Cumulative_deaths": "Cumulative Deaths", "Country": "Country"})
    
    fig.update_layout(coloraxis_showscale=False) #remove color bar
    st.plotly_chart(fig)

with column2:
    fig = px.bar(df_filtered3.sort_values('Cumulative_cases', ascending=True), x='Cumulative_cases', y='Country', orientation='h',
                 color='Cumulative_cases', color_continuous_scale='Reds', 
                 title = f'Cumulative Cases (as of {last_date})',
                 labels={"Cumulative_cases": "Cumulative Cases", "Country": "Country"})
    
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig)

column1, column2 = st.columns(2)

with column1:
    fig = px.bar(df_filtered3.sort_values('Cum_deaths_per', ascending=True), x='Cum_deaths_per', y='Country', orientation='h',
                 color='Cum_deaths_per', color_continuous_scale='Reds',
                 title = f'Cumulative Deaths as a % of Population (as of {last_date})', 
                 labels={"Cum_deaths_per": "Cumulative Deaths (% of Population)", "Country": "Country"})
    
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(xaxis_tickformat=".2%") #2 decimals
    
    st.plotly_chart(fig)

with column2:
    fig = px.bar(df_filtered3.sort_values('Cum_cases_per', ascending=True), x='Cum_cases_per', y='Country', orientation='h',
                 color='Cum_cases_per', color_continuous_scale='Reds',
                 title = f'Cumulative Cases as a % of Population (as of {last_date})',
                 labels={"Cum_cases_per": "Cumulative Cases (% of Population)", "Country": "Country"})

    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(xaxis_tickformat=".2%") #2 decimals
    
    st.plotly_chart(fig)

column1, column2 = st.columns(2)

with column1:
    #new cases per day
    df_filtered_cases = df_filtered.groupby(['Date_reported', 'Country'], as_index=False).agg(New_cases=('New_cases', 'sum'),
                                                                                              New_cases_per=('New_cases_per', 'sum'))
    fig = px.line(df_filtered_cases, x="Date_reported", y="New_cases", color='Country', title = f'New cases (as of {last_date})',
                  labels={"Date_reported": "Date", "New_cases": "New Cases"}) 
    st.plotly_chart(fig)

with column2:
    #new deaths per day
    df_filtered_deaths = df_filtered.groupby(['Date_reported', 'Country'], as_index=False).agg(New_deaths=('New_deaths', 'sum'),
                                                                                               New_deaths_per=('New_deaths_per', 'sum'))
    fig = px.line(df_filtered_deaths, x="Date_reported", y="New_deaths", color='Country', title = f'New deaths (as of {last_date})',
                  labels={"Date_reported": "Date", "New_deaths": "New Deaths"}) 
    st.plotly_chart(fig)

column1, column2 = st.columns(2)

with column1:
    #new cases per day
    fig = px.line(df_filtered_cases, x="Date_reported", y="New_cases_per", color='Country', 
                  title = f'New cases as a % of Population (as of {last_date})',
                  labels={"Date_reported": "Date", "New_cases_per": "New Cases (% of Population)"}) 
    fig.update_layout(yaxis_tickformat=".2%") #2 decimals
    
    st.plotly_chart(fig)

with column2:
    #new deaths per day
    fig = px.line(df_filtered_deaths, x="Date_reported", y="New_deaths_per", color='Country', 
                  title = f'New deaths as a % of Population (as of {last_date})',
                  labels={"Date_reported": "Date", "New_deaths_per": "New Deaths (% of Population)"}) 
    fig.update_layout(yaxis_tickformat=".3%") #3 decimals
    
    st.plotly_chart(fig)