import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

# Inject CSS
st.markdown("""
    <style>
    /* ======= GLOBAL LAYOUT ======= */

    /* Main background */
    .stApp {
        background-color: #f2f2f2 !important; /* light gray */
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #e6e6e6 !important; /* slightly darker gray */
    }

    /* Force text color (headings, paragraphs, widget labels) */
    .stApp, .stApp * {
        color: #000000 !important; /* black text */
    }

    /* ======= WIDGET STYLING ======= */
    /* Light yellow widget background */
    input, select, textarea {
        background-color: #fff9c4 !important; /* light yellow */
        border: 1px solid #e0d98c !important;
        color: #000 !important;
    }

    /* Streamlit inputs (text_input, number_input, date_input, etc.) */
    div[data-baseweb="input"] input {
        background-color: #fff9c4 !important;
        color: #000 !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #fff9c4 !important;
        color: #000 !important;
    }

    /* Expander headers */
    details > summary {
        background-color: #fff9c4 !important;
        color: #000 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #fff9c4 !important;
        color: #000000 !important;
        border: 1px solid #e0d98c !important;
        border-radius: 6px;
        padding: 0.6em 1.2em;
    }

    div.stButton > button:hover {
        background-color: #fff176 !important; /* slightly stronger yellow on hover */
    }

    /* Radio + checkbox labels */
    div[data-baseweb="radio"] label,
    div[data-baseweb="checkbox"] label {
        background-color: transparent !important;
        color: #000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide") # make it full page

st.title("Pricing Optimization Dashboard")

#databases
df_full = pd.read_csv("final_database.csv")
df_full = df_full.drop(columns=['Unnamed: 0'])
df_full['date'] = pd.to_datetime(df_full['YEAR'].astype(str) + df_full['WEEK'].astype(str) + '1',format='%G%V%u')

df = pd.read_csv("weekly_model_dashboard.csv")
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
    
tab1, tab2, tab3 = st.tabs(["Overview", "Model Prediction", "Sensitivity Analysis"])

def custom_metric(label, value, delta, background="#f8f9fa"):
    st.markdown(f"""
<div style="
    background-color: {background};
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #ddd;
    margin-bottom: 10px;
    text-align: center;            /* centers the text */
    width: 100%;
    margin-left: auto;             /* centers the block */
    margin-right: auto;
">
    <div style="font-size: 20px; font-weight: 600;">
        {label}
    </div>
    <div style="font-size: 16px; font-weight: 700; margin-top: 6px;">
        {value}
    </div>
    <div style="font-size: 16px; margin-top: 4px;">
        {delta}
    </div>
</div>
""", unsafe_allow_html=True)

with tab1:
    st.header("Dashboard Overview")

    # KPI boxes
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        custom_metric(label="Average DCM (WTD)", value=f"{(last_week_dcm*100):.2f}%", delta=f'{(delta_dcm*100):.2f}%')

    with kpi2:
        custom_metric(label="Top Store (WTD)", value=store, delta=f'{delta_pct:.2f}%')

    with kpi3:
        custom_metric(label="Total Qty Sold (WTD)", value=total_quantity_last_week, delta=f'{(delta_quantity*100):.2f}%')

    with kpi4:
        custom_metric(label="Top Performing SKU (WTD)", value=sku, delta=f'{delta_pct_sku:.2f}%')

    #filters
    
        # Model info box
    st.markdown("""
    <style>
    .info-box {
        background: #eef4ff;
        border-left: 6px solid #3b73ff;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
    }

    .info-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f3b70;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .info-text {
        font-size: 15px;
        color: #333;
        line-height: 1.45;
    }
    </style>

    <div class="info-box">
        <div class="info-title">ℹ️ Important</div>
        <div class="info-text">
            The KPIs displayed at the top are based on the complete dataset.
            Filters only apply to the charts below, not the KPIs.
        </div>
    </div>
    """, unsafe_allow_html=True)


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
        "white": "#FFFFFF",
        "gray": "#A9A9A9",
        "black": "#111111"
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
            title_font=dict(size=18, color=colors["black"]),
            font=dict(color=colors["black"]),  
            plot_bgcolor=colors["white"],      
            paper_bgcolor=colors["white"]      
        )

        fig1.update_xaxes(
        showline=True,
        linecolor=colors["black"],
        tickfont=dict(color=colors["black"]),
        title=dict(text="SKU", font=dict(color=colors["black"])))

        fig1.update_yaxes(
            showline=True,
            linecolor=colors["black"],
            tickfont=dict(color=colors["black"]),
            title=dict(text="Quantity", font=dict(color=colors["black"])))

        fig1.update_xaxes(tickangle=45)

        st.plotly_chart(fig1, use_container_width=True)

    # column 2: segment share bar chart
    with col2:
        df_segment1 = df_filtered.groupby('segment', as_index=False).agg(quantity=('quantity', 'sum')) 
        df_bar = df_segment1[df_segment1['segment'] != 'Total'].copy()
        df_bar["pct"] = df_bar["quantity"] / df_bar["quantity"].sum()
        df_bar = df_bar.sort_values('quantity', ascending=True)

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
            template="plotly_white",
            xaxis_title="Percentage",
            yaxis_title="Segment",
            title_font=dict(size=18, color=colors["black"]),
            font=dict(color=colors["black"]),
            plot_bgcolor=colors["white"],
            paper_bgcolor=colors["white"]
        )

        fig2.update_xaxes(
        showline=True,
        linecolor=colors["black"],
        tickfont=dict(color=colors["black"]),
        title=dict(text="Percentage", font=dict(color=colors["black"])))

        fig2.update_yaxes(
            showline=True,
            linecolor=colors["black"],
            tickfont=dict(color=colors["black"]),
            title=dict(text="Segment", font=dict(color=colors["black"])))

        st.plotly_chart(fig2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ttm trend chart
        df_week = df_filtered.groupby('date', as_index=False).agg(quantity=('quantity', 'sum'))
        
        fig3 = px.line(
            df_week,
            x="date",
            y="quantity",
            title=f"Trend From {start_date} to {end_date}",
            markers=True
        )

        fig3.update_traces(
            line=dict(color=colors["yellow"], width=3),
            marker=dict(color=colors["gray"]))

        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Quantity",
            font=dict(color=colors["black"]),   
            title_font=dict(color=colors["black"]),
            plot_bgcolor=colors["white"],
            paper_bgcolor=colors["white"]
        )

        fig3.update_xaxes(
            showline=True,
            linecolor=colors["black"],
            tickfont=dict(color=colors["black"]),
            title=dict(text="Date", font=dict(color=colors["black"])))

        fig3.update_yaxes(
            showline=True,
            linecolor=colors["black"],
            tickfont=dict(color=colors["black"]),
            title=dict(text="Quantity", font=dict(color=colors["black"])))

        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        df_stores = df_filtered.groupby(['TP'], as_index=False).agg(quantity=('quantity', 'sum'))
        df_stores = df_stores.sort_values('quantity', ascending=True)

        fig2 = px.bar(
            df_stores,
            x="quantity",
            y="TP",
            orientation="h",
            title="Quantity by TP",
            color="quantity",
            color_continuous_scale=[colors["gray"], colors["yellow"]],
        )

        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Quantity",
            yaxis_title="TP",
            title_font=dict(size=18, color=colors["black"]),
            font=dict(color=colors["black"]),
            plot_bgcolor=colors["white"],
            paper_bgcolor=colors["white"]
        )

        fig2.update_xaxes(
        showline=True,
        linecolor=colors["black"],
        tickfont=dict(color=colors["black"]),
        title=dict(text="Quantity", font=dict(color=colors["black"])))

        fig2.update_yaxes(
            showline=True,
            linecolor=colors["black"],
            tickfont=dict(color=colors["black"]),
            title=dict(text="TP", font=dict(color=colors["black"])))

        st.plotly_chart(fig2, use_container_width=True)

with tab2:   
    st.header("Model Prediction")
    import streamlit as st

    # Model info box
    st.markdown("""
    <style>
    .summary-card {
        background-color: #f0f6ff; /* light blue background */
        border-radius: 12px;
        padding: 20px 26px;
        border: 1px solid #d4e3ff;
        margin-bottom: 20px;
    }

    .summary-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .summary-text {
        font-size: 15px;
        color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render the card
    st.markdown("""
    <div class="summary-card">
        <div class="summary-title">
            <span>📈 Model Training Summary</span>
        </div>
        <div class="summary-text">
            We use an XGBoost Regressor trained on historical data with a 70% training / 30% testing split.
            The model is automatically retrained whenever new data is added, ensuring predictions always reflect
            the most current information.
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    date_test = df['date'].iloc[train_size:]

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
    y_pred = best_xgb.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred) #average absolute difference between predicted and actual
    r2 = r2_score(y_test, y_pred) #how much of the variance is explained
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

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
        quantity_tp_segment = float(quantity_tp * segment_percent[tp].loc[segment_percent[tp]['segment'] == segment, '%'])

        return [tp, segment, quantity_tp_segment]

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

        def result_box(tp, segment, quantity):
            st.markdown(f"""
                <div style="
                    background-color: #fff9c4;
                    padding: 22px;
                    border-radius: 10px;
                    border: 1px solid #e6e6e6;
                    width: 100%;
                    text-align: center;
                    height: 130px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 20px; color: #555; font-weight: 500;">
                        Quantity for <b>{tp}</b> for <b>{segment}</b> is:
                    </div>
                    <div style="font-size: 30px; font-weight: 700; margin-top: 8px;">
                        {round(quantity, 2)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            tp, segment, quantity = result
            result_box(tp, segment, quantity)

            st.markdown("<br>", unsafe_allow_html=True)

            # Replace with your actual colors from previous charts
            actual_color = "#1f77b4"      # example
            predicted_color = "#ff7f0e"   # example

            fig_pred = go.Figure()

            # Actual line
            fig_pred.add_trace(go.Scatter(
                x=date_test,
                y=y_test,
                mode='lines',
                name='Actual',
                line=dict(color=actual_color, width=2)
            ))

            # Predicted line
            fig_pred.add_trace(go.Scatter(
                x=date_test,
                y=y_pred,
                mode='lines',
                name='Predicted',
                line=dict(color=predicted_color, width=2, dash="dash")
            ))

            # Layout
            fig_pred.update_layout(
                title="Weekly Quantity Forecast",
                xaxis_title="Date",
                yaxis_title="Quantity",
                template="plotly_white",
                width=900,
                height=450,
                legend=dict(x=0, y=1, font=dict(color=colors['black'])),
                title_font=dict(size=18, color=colors["black"]),
                font=dict(color=colors["black"]),
                plot_bgcolor=colors["white"],
                paper_bgcolor=colors["white"]
            )

            fig_pred.update_xaxes(
                showline=True,
                linecolor=colors["black"],
                tickfont=dict(color=colors["black"]),
                title=dict(text="Quantity", font=dict(color=colors["black"])))

            fig_pred.update_yaxes(
                showline=True,
                linecolor=colors["black"],
                tickfont=dict(color=colors["black"]),
                title=dict(text="TP", font=dict(color=colors["black"])))
            
            # Streamlit visualization
            st.plotly_chart(fig_pred, use_container_width=True)
        
        def kpi_box(label, value, suffix=""):
            st.markdown(f"""
                <div style="
                    background-color: #f8f9fa;
                    padding: 22px;
                    border-radius: 10px;
                    border: 1px solid #e6e6e6;
                    width: 100%;
                    text-align: center;
                    height: 130px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <div style="font-size: 30px; color: #555; font-weight: 500;">
                        {label}
                    </div>
                    <div style="font-size: 20px; font-weight: 600; margin-top: 8px;">
                        {round(value, 2)}{suffix}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader('Model Performance Metrics')
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                kpi_box('RMSE', rmse)
            with kpi2:
                kpi_box('MAE', mae)
            st.markdown("<br>", unsafe_allow_html=True)
            kpi3, kpi4 = st.columns(2)
            with kpi3:
                kpi_box('MAPE', mape, suffix='%')
            with kpi4:
                kpi_box('R2', r2)


with tab3:
    st.header("Sensitivity Analysis")

    last_date = df_full["date"].max()  # keep as Timestamp
    cutoff_date = last_date - pd.Timedelta(days=7)
    df_last_week = df_full[df_full["date"] >= cutoff_date]
    df_segment_last_week = (
        df_last_week
        .groupby("segment", as_index=False)
        .agg(quantity=("quantity", "sum"))
    )
    df_segment_last_week = df_segment_last_week.sort_values('quantity', ascending=True)

    fig5 = px.bar(df_segment_last_week,            
        x="quantity",
        y="segment",
        orientation="h",
        title="Quantity by Segment (Last 7 days)",
        color="quantity",
        color_continuous_scale=[colors["gray"], colors["yellow"]],)

    fig5.update_layout(
        template="plotly_white",
        xaxis_title="Quantity",
        yaxis_title="Segment",
        title_font=dict(size=18, color=colors["black"]),
        font=dict(color=colors["black"]),
        plot_bgcolor=colors["white"],
        paper_bgcolor=colors["white"])

    fig5.update_xaxes(
    showline=True,
    linecolor=colors["black"],
    tickfont=dict(color=colors["black"]),
    title=dict(text="Quantity", font=dict(color=colors["black"])))

    fig5.update_yaxes(
        showline=True,
        linecolor=colors["black"],
        tickfont=dict(color=colors["black"]),
        title=dict(text="Segment", font=dict(color=colors["black"])))

    st.plotly_chart(fig5, use_container_width=True)

    df_merged = pd.read_csv('elasticity_data.csv')
    import statsmodels.formula.api as smf
    results = []

    for seg, data in df_merged.groupby('segment'):
        #drop missing values
        data = data.dropna(subset=['ln_qty', 'ln_price', 'ln_competitor_price', 'ln_cost'])
        if len(data) < 30:
            continue

        model = smf.ols('ln_qty ~ ln_price + ln_competitor_price', data=data).fit()
        own_elast = model.params['ln_price']
        cross_elast = model.params['ln_competitor_price']

        #average cost and current price
        avg_cost = np.exp(data['ln_cost']).mean()
        avg_price = np.exp(data['ln_price']).mean()

        #optimum price based on elasticity
        p_opt = avg_cost * (own_elast / (own_elast + 1))

        results.append({
        'segment': seg,
        'own_elasticity': own_elast,
        'cross_elasticity': cross_elast,
        'opt_price': p_opt,})

        opt_results = pd.DataFrame(results)
    
    def data_box(segment, elasticity, optimal_price):
        st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #ddd;
                width: 100%;
                margin-bottom: 10px;
            ">
                <div style="font-size: 18px; font-weight: 600;">{segment}</div>
                <div style="font-size: 15px; margin-top: 4px;">
                    <strong>Elasticity:</strong> {elasticity:.2f}
                </div>
                <div style="font-size: 15px;">
                    <strong>Optimal Price:</strong> ${optimal_price:.2f}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        data_box(
            segment='Top Load',
            elasticity = opt_results.loc[opt_results['segment'] == 'TOP LOAD', 'own_elasticity'].iloc[0],
            optimal_price = opt_results.loc[opt_results['segment'] == 'TOP LOAD', 'opt_price'].iloc[0])

    with col2:
        data_box(
            segment='Single Tub',
            elasticity = opt_results.loc[opt_results['segment'] == 'SINGLE TUB', 'own_elasticity'].iloc[0],
            optimal_price = opt_results.loc[opt_results['segment'] == 'SINGLE TUB', 'opt_price'].iloc[0])

    with col3:
        data_box(
            segment='Front Load',
            elasticity = opt_results.loc[opt_results['segment'] == 'FRONT LOAD', 'own_elasticity'].iloc[0],
            optimal_price = opt_results.loc[opt_results['segment'] == 'FRONT LOAD', 'opt_price'].iloc[0])

    with col4:
        data_box(
            segment='Twin Tub',
            elasticity = opt_results.loc[opt_results['segment'] == 'TWIN TUB', 'own_elasticity'].iloc[0],
            optimal_price = opt_results.loc[opt_results['segment'] == 'TWIN TUB', 'opt_price'].iloc[0])

    with col5:
        data_box(
            segment='Laundry Center',
            elasticity = opt_results.loc[opt_results['segment'] == 'LAUNDRY CENTER', 'own_elasticity'].iloc[0],
            optimal_price = opt_results.loc[opt_results['segment'] == 'LAUNDRY CENTER', 'opt_price'].iloc[0])



