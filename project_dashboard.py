import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px

st.title("My Dashboard")

#databases
df_full = pd.read_csv(r"C:\Users\Ramon\Desktop\escuela\Data Analytics and AI\project\final_database.csv")
df_full = df_full.drop(columns=['Unnamed: 0'])
df_full['date'] = pd.to_datetime(df_full['YEAR'].astype(str) + df_full['WEEK'].astype(str) + '1',format='%G%V%u')

df = pd.read_csv(r"C:\Users\Ramon\Desktop\escuela\Data Analytics and AI\project\weekly_model_dashboard.csv")
df = df.drop(columns=['Unnamed: 0'])
df['date'] = pd.to_datetime(df['YEAR'].astype(str) + df['WEEK'].astype(str) + '1',format='%G%V%u')

df_tp_segment = df_full.groupby(['TP', 'segment'], as_index=False).agg(quantity=('quantity', 'sum'))

# for the kpi of top store by quantity per week
top_store = df_full.groupby(['TP', 'date'], as_index=False).agg(quantity=('quantity', 'sum'))
# LAST WEEK
last_week_date = top_store['date'].max()
df_last_week = top_store[top_store['date'] == last_week_date]
last_row = df_last_week.loc[df_last_week['quantity'].idxmax()]
store = last_row['TP']
top_qty = last_row['quantity']
# PREVIOUS WEEK
all_weeks = sorted(top_store['date'].unique())
if len(all_weeks) > 1:
    prev_week_date = all_weeks[-2]
    df_prev_week = top_store[top_store['date'] == prev_week_date]
    prev_row = df_prev_week.loc[df_prev_week['quantity'].idxmax()]
    prev_qty = prev_row['quantity']

    # DELTA
    delta_abs = top_qty - prev_qty
    delta_pct = (delta_abs / prev_qty) * 100
else:
    delta_abs = 0
    delta_pct = 0

# kpi of avg dcm
avg_dcm = df_full.groupby(['date'], as_index=False).agg(dcm=('DCM_margin', 'mean'))
last_week_dcm = avg_dcm['dcm'].iloc[-1]
#delta dcm
delta_dcm = (last_week_dcm - avg_dcm['dcm'].iloc[-2]) / avg_dcm['dcm'].iloc[-2]

#kpi of total quantity
total_quantity = df_full.groupby(['date'], as_index=False).agg(total_quantity=('quantity', 'sum'))
total_quantity_last_week = total_quantity['total_quantity'].iloc[-1]
#delta quantity
delta_quantity = (total_quantity_last_week - total_quantity['total_quantity'].iloc[-2]) / total_quantity['total_quantity'].iloc[-2]

#kpi for top sku
top_sku = df_full.groupby(['SKU', 'date'], as_index=False).agg(quantity=('quantity', 'sum'))
# LAST WEEK
last_week_date = top_sku['date'].max()
df_last_week = top_sku[top_sku['date'] == last_week_date]
last_row = df_last_week.loc[df_last_week['quantity'].idxmax()]
sku = last_row['SKU']
top_qty_sku = last_row['quantity']
# PREVIOUS WEEK
all_weeks = sorted(top_sku['date'].unique())
if len(all_weeks) > 1:
    prev_week_date = all_weeks[-2]
    df_prev_week = top_sku[top_sku['date'] == prev_week_date]
    prev_row = df_prev_week.loc[df_prev_week['quantity'].idxmax()]
    prev_qty = prev_row['quantity']

    # DELTA
    delta_abs_sku = top_qty_sku - prev_qty
    delta_pct_sku = (delta_abs / prev_qty) * 100
else:
    delta_abs_sku = 0
    delta_pct_sku = 0

segment_percent = {}

for tp, group in df_tp_segment.groupby('TP'):
    # sum quantity per segment for this TP
    df_sum1 = (
        group.groupby('segment', as_index=False)['quantity']
        .sum()
        .sort_values('quantity', ascending=False)
    )

    #total quantity for this TP
    total_quantity = df_sum1['quantity'].sum()
    df_sum1['%'] = (df_sum1['quantity'] / total_quantity)

    #total row
    total_row = pd.DataFrame({
        'segment': ['Total'],
        'quantity': [total_quantity],
        '%': [1]
    })

    df_sum1 = pd.concat([df_sum1, total_row], ignore_index=True)
    segment_percent[tp] = df_sum1
    
    df_sum = df_tp_segment.groupby(['TP'], as_index=False).agg(quantity=('quantity', 'sum'))
    total_quantity = df_sum['quantity'].sum()
    last_row = {
        'TP': 'Total',
        'quantity': total_quantity
    }

    df_sum = pd.concat([df_sum, pd.DataFrame([last_row])], ignore_index=True)
    total = df_sum.iloc[-1]['quantity'] #the last one is the total
    df_sum['%'] = (df_sum['quantity'] / total)
    
# Group by segment
df_segment = df_full.groupby('segment', as_index=False).agg(quantity=('quantity', 'sum'))

# Compute total and append
total_quantity = df_segment['quantity'].sum()
last_row = {'segment': 'Total', 'quantity': total_quantity}
df_segment = pd.concat([df_segment, pd.DataFrame([last_row])], ignore_index=True)

# Add percentage column
total = df_segment.iloc[-1]['quantity']
df_segment['%'] = df_segment['quantity'] / total
    
tab1, tab2, tab3 = st.tabs(["Overview", "Prediction", "Sensitivity Analysis"])

with tab1:
    st.header("Dashboard Overview")

    # KPI boxes
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            label="Avg Profit Margin",
            value=f"{(last_week_dcm*100):.2f}%",
            delta=f'{(delta_dcm*100):.2f}%'
        )

    with kpi2:
        st.metric(
            label="Top Performing Store",
            value=store,
            delta=f'{delta_pct:.2f}%'
        )

    with kpi3:
        st.metric(
            label="Total Qty Sold (WTD)",
            value=total_quantity_last_week,
            delta=f'{(delta_quantity*100):.2f}%'
        )

    with kpi4:
        st.metric(
            label="Top Performing SKU",
            value=sku,
            delta=f'{delta_pct_sku:.2f}%'
        )

    #filters
    
    all_tp_values = df_full['TP'].unique().tolist() #all tp selected
    
    options = st.multiselect(
    "Select TP",
    df_full['TP'].unique().tolist(),
    default=all_tp_values,)
    
    first_date = df_full["date"].min().date()
    last_date = df_full["date"].max().date()
    
    selected_dates = st.date_input(
    "Select your date range",
    (first_date, last_date),        # default range
    min_value=first_date,           # limit backward selection
    max_value=last_date,            # limit forward selection
    format="YYYY-MM-DD")
    
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        st.error("Please select a start and end date.")
        st.stop()

    df_filtered = df_full[
        (df_full["TP"].isin(options)) &
        (df_full["date"].dt.date >= start_date) &
        (df_full["date"].dt.date <= end_date)
]
    
    # color theme (yellow, black, gray, white)
    colors = {
        "yellow": "#FFD700",
        "black": "#111111",
        "gray": "#A9A9A9",
        "white": "#FFFFFF"
    }

    # 2 graph columns

    col1, col2 = st.columns(2)

    # column 1: top skus bar chart
    with col1:

        df_sku = df_filtered.groupby('SKU', as_index=False)['quantity'].sum()
        df_top10 = df_sku.sort_values('quantity', ascending=False).head(10)

        fig1 = px.bar(
            df_top10,
            x='SKU',
            y='quantity',
            title='Top 10 SKUs by Quantity',
            color='quantity',
            color_continuous_scale=[colors["gray"], colors["yellow"]],
        )

        fig1.update_layout(
            template="plotly_dark", 
            xaxis_title="SKU",
            yaxis_title="Total Quantity",
            title_font=dict(size=18, color=colors["white"]),
            font=dict(color=colors["white"]),  
            plot_bgcolor=colors["black"],      
            paper_bgcolor=colors["black"]      
        )

        fig1.update_xaxes(tickangle=45)

        st.plotly_chart(fig1, use_container_width=True)

    # column 2: segment share bar chart
    with col2:
        df_segment1 = df_filtered.groupby('segment', as_index=False).agg(quantity=('quantity', 'sum')) 
        df_bar = df_segment1[df_segment1['segment'] != 'Total'].copy()
        df_bar["pct"] = df_bar["quantity"] / df_bar["quantity"].sum()
        df_bar = df_bar.sort_values('quantity', ascending=False).head(10)

        fig2 = px.bar(
            df_bar,
            x="pct",
            y="segment",
            orientation="h",
            title="Quantity Share by Segment",
            color="pct",
            color_continuous_scale=[colors["gray"], colors["yellow"]],
        )

        fig2.update_layout(
            template="plotly_dark",
            xaxis_title="Percentage",
            yaxis_title="Segment",
            title_font=dict(size=18, color=colors["white"]),
            font=dict(color=colors["white"]),
            plot_bgcolor=colors["black"],
            paper_bgcolor=colors["black"]
        )

        st.plotly_chart(fig2, use_container_width=True)
        
    # ttm trend chart

    st.subheader("Quantity Trend")
    df_week = df_filtered.groupby('date', as_index=False).agg(quantity=('quantity', 'sum'))
    
    fig3 = px.line(
        df_week,
        x="date",
        y="quantity",
        title="Trend Over Last 12 Months",
        markers=True
    )

    fig3.update_traces(
        line=dict(color=colors["yellow"], width=3),
        marker=dict(color=colors["white"])  
    )

    fig3.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Quantity",
        font=dict(color=colors["white"]),   
        title_font=dict(color=colors["white"]),
        plot_bgcolor=colors["black"],
        paper_bgcolor=colors["black"]
    )

    st.plotly_chart(fig3, use_container_width=True)


with tab2:   
    st.header("Model Prediction")
    st.write("Enter the inputs below to generate a prediction:")

    # user inputs
    week_number = st.number_input("Week Number (1–52)", min_value=1, max_value=52, value=10)

    tp = st.selectbox("TP", df_sum['TP'].unique())

    segment = st.selectbox(
        "Segment",
        segment_percent[tp]['segment'].unique()
    )

    price_last_week = st.number_input("Price (Last Week)", min_value=0.0, value=10.0)
    qty_last_week = st.number_input("Quantity (Last Week)", min_value=0.0, value=100.0)
    qty_avg_prev_4w = st.number_input("Average Qty Previous 4 Weeks", min_value=0.0, value=120.0)

    # model
    # drop NaNs from initial lag features
    df_model = df.dropna(subset=['qty_last_week', 'qty_avg_prev_4w']).copy()

    df_model['week_sin'] = np.sin(2 * np.pi * df_model['WEEK'] / 52)
    df_model['week_cos'] = np.cos(2 * np.pi * df_model['WEEK'] / 52)
    
    features = [
        'week_sin', 'week_cos', 'price_avg_last_week', 
        'qty_last_week', 'qty_avg_prev_4w', 'event_effect'
    ]
    target = 'quantity'

    train_size = int(len(df_model) * 0.7)

    X_train = df_model[features].iloc[:train_size]
    y_train = df_model[target].iloc[:train_size]

    X_test = df_model[features].iloc[train_size:]
    y_test = df_model[target].iloc[train_size:]

    #best model parameters
    best_xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.7,
        learning_rate=0.3,
        max_depth=3,
        n_estimators=100,
        subsample=0.5
    )

    # Train the model
    best_xgb.fit(X_train, y_train)

    # pred function
    def predict_quantity(week_number, tp, segment, price_last_week, qty_last_week, qty_avg_prev_4w):
        # cyclical features
        week_sin = np.sin(2 * np.pi * week_number / 52) 
        week_cos = np.cos(2 * np.pi * week_number / 52)

        # find event_effect for that week
        event_effect = df_model.loc[df_model['WEEK'] == week_number, 'event_effect'].iloc[0]

        x = np.array([[week_sin, week_cos, price_last_week, qty_last_week, qty_avg_prev_4w, event_effect]])
        y_hat = float(best_xgb.predict(x))

        quantity_tp = float(y_hat * df_sum.loc[df_sum['TP'] == tp, '%'].iloc[0])
        quantity_tp_segment = float(
            quantity_tp * segment_percent[tp].loc[segment_percent[tp]['segment'] == segment, '%']
        )
        
        result = pd.DataFrame([{
            'week_number': week_number,
            'tp': tp,
            'segment': segment,
            'price_last_week': price_last_week,
            'qty_last_week': qty_last_week,
            'qty_avg_prev_4w': qty_avg_prev_4w,
            'Overall quantity': y_hat,
            f'Quantity for {tp}': quantity_tp,
            f'Quantity for {tp} for {segment}': quantity_tp_segment
        }])

        return result

    #run prediction
    if st.button("Generate Prediction"):
        result = predict_quantity(
            week_number=week_number,
            tp=tp,
            segment=segment,
            price_last_week=price_last_week,
            qty_last_week=qty_last_week,
            qty_avg_prev_4w=qty_avg_prev_4w
        )

        st.subheader("Prediction Output")
        st.dataframe(result)

with tab3:
    st.header("Sensitivity Analysis")
    st.write("Model outputs or charts go here.")


