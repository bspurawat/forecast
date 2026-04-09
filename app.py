import numpy as np
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#test
st.set_page_config(page_title="Demand Forecast Dashboard (Himani Purawat, MUJ)", layout="wide")
st.title("📈 Forecasting at scale - (Himani Purawat, MUJ)")

def get_accuracy(actual, forecast):
    mask = ~np.isnan(forecast) & ~np.isnan(actual) & (actual != 0)
    a, f = np.array(actual)[mask], np.array(forecast)[mask]
    if len(a) == 0: return 0
    mape = np.mean(np.abs((a - f) / a)) * 100
    return max(0, 100 - mape)

# 1. Upload DAta file using Sidebar - Data Upload
uploaded_file = "data.csv" #st.sidebar.file_uploader("Upload your data [.csv]", type=['csv'])
#st.sidebar.file_uploader("Upload your data [.csv]", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 2. Controls
    st.sidebar.header("Global Filters")
    kpi_type = st.sidebar.selectbox("Category", ["brand", "store_type"])
    options = ["All"] + list(df[kpi_type].unique())
    selected = st.sidebar.selectbox(f"Select {kpi_type}", options)
    granularity = st.sidebar.radio("Granularity", ["Daily", "Weekly", "Monthly"])
    periods = st.sidebar.slider("Forecast Periods", 1, 104, 52)
    
    # 3. Processing
    df['ds'] = pd.to_datetime(df['ds'], dayfirst=True)
    filtered_df = df if selected == "All" else df[df[kpi_type] == selected]
    daily_volume = filtered_df.groupby('ds').agg({'quantity': 'sum'}).reset_index().sort_values('ds').set_index('ds')

    # . Data Resampling
    resample_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    daily_volume = daily_volume.resample(resample_map[granularity]).sum().reset_index()

    df = daily_volume.rename(columns={'quantity': 'y'})

    # 4. Modeling
    if not df.empty:
        #Prophet AI
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq="W") #{"Daily": "D", "Weekly": "W", "Monthly": "ME"}[granularity])
        forecast = model.predict(future)
        # Convenstional mothods 
        #Moving Average including future dates from forecast dataframe to calulate MA for future dates
        df_combined = forecast[['ds']].merge(df[['ds', 'y']], on='ds', how='left')
        df_combined['MA_future'] = df_combined['y'].ffill().rolling(window=3).mean()
        df_combined['ES_future'] = df_combined['y'].ffill().ewm(alpha=0.3, adjust=False).mean()
        historical_rows = len(df)
        X_train = np.arange(historical_rows).reshape(-1, 1)
        y_train = df['y'].values
        # 2. Fit the actual LinearRegression model
        reg_model = LinearRegression().fit(X_train, y_train)
        # 3. Predict for the entire timeline (Historical + Future)
        X_all = np.arange(len(df_combined)).reshape(-1, 1)
        df_combined['Reg_future'] = reg_model.predict(X_all)

        # 5. Interactive Visuals / UI Grid Layout (2x2)
        # Row 1
        col1, col2 = st.columns(2)
        #moving average
        with col1:
          # Plotting Moving Average only
            fig1 = go.Figure()
            # Add Historical Actuals    
            fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actuals', marker=dict(color='green', size=3)))
            # Add MA Forecast (Line covers both history and future dates from 'forecast')
            fig1.add_trace(go.Scatter(x=df_combined['ds'], y=df_combined['MA_future'], mode='lines', name='MA Forecast', line=dict(color="#2ECC71", width=3)))
            fig1.update_layout(#height=250,
                               title=f"Moving Average Forecast for {kpi_type}-{selected} ({granularity})", 
                               showlegend=True, xaxis_rangeslider_visible=False,
                                   legend=dict(orientation="h",yanchor="top", y=0.99,
                                               xanchor="right", x=0.99))
            #Calculate KPIs
            acc = get_accuracy(df_combined['y'], df_combined['MA_future'])
            # 1. Period-over-Period Change (Percentage change of the MA)
            # Compares the last MA value to the one before it
            last_ma = df_combined['MA_future'].iloc[-1]
            prev_ma = df_combined['MA_future'].iloc[-2]
            pop_change = ((last_ma - prev_ma) / prev_ma) * 100
            # 2. Volatility (Standard Deviation of the actuals)
            # Tells the user how "noisy" the black dots are compared to the smooth green line
            volatility = df['y'].std()

            with st.container(border=True):
                st.plotly_chart(fig1, use_container_width=True, key="plot_ma")      
                # Layout metrics in 3 columns
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Accuracy", f"{acc:.1f}%")
                with m2:
                    # delta color shows if the average is trending up or down
                    st.metric("Trend (PoP)", f"{last_ma:,.0f}", delta=f"{pop_change:.1f}%")
                with m3:
                    # High volatility means the Moving Average might be less reliable
                    st.metric("Volatility (Std Dev)", f"{volatility:.1f}")  
        #exponential smoothing
        with col2:
            fig2 = go.Figure()
            # Add Historical Actuals
            fig2.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actuals', marker=dict(color='black', size=3)))
            fig2.add_trace(go.Scatter(x=df_combined['ds'], y=df_combined['ES_future'], mode='lines', name='ES Forecast', line=dict(color="#3498DB", width=3)))
            fig2.update_layout(#height=250,
                               title=f"Exponential Smoothing Forecast for {kpi_type}-{selected} ({granularity})",
                               showlegend=True, xaxis_rangeslider_visible=False
                                   ,legend=dict(orientation="h",yanchor="top", y=0.99,
                                               xanchor="right", x=0.99))
            #Calculate KPIs
            acc = get_accuracy(df_combined['y'], df_combined['ES_future'])
            # Calculate additional KPIs
            # 1. RMSE: Error in the same units as your data
            rmse = np.sqrt(mean_squared_error(df_combined.dropna()['y'], df_combined.dropna()['ES_future']))
            # 2. Next Prediction: The latest forecasted value
            next_val = df_combined['ES_future'].iloc[-1]

            with st.container(border=True):
                st.plotly_chart(fig2, use_container_width=True, key="plot_es")
                # Create 3 sub-columns for the KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.metric("Accuracy", f"{acc:.1f}%")
                with kpi2:
                    # Shows the average deviation from actual values
                    st.metric("RMSE (Error)", f"{rmse:.2f}",delta=0.0)
                with kpi3:
                    # Shows the final forecasted point
                    st.metric("Final Forecast", f"{next_val:,.0f}")
                    
        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actuals', marker=dict(color='black', size=3)))
            fig3.add_trace(go.Scatter(x=df_combined['ds'], y=df_combined['Reg_future'], mode='lines', name='Reg Forecast', line=dict(color="#CF14E4", width=3)))
            fig3.update_layout(height=500,
                               title=f"Regression Forecast for {kpi_type}-{selected} ({granularity})",
                               showlegend=True, xaxis_rangeslider_visible=False
                                   ,legend=dict(orientation="h",yanchor="top", y=0.99,
                                               xanchor="right", x=0.99))
            #Calculate KPIs
            acc = get_accuracy(df_combined['y'], df_combined['Reg_future'])
            # 1. R-Squared (How well the line fits the data)
            # Ensure we only compare points where we have both actuals and predictions
            mask = df_combined['y'].notna()
            r2 = r2_score(df_combined.loc[mask, 'y'], df_combined.loc[mask, 'Reg_future'])
            
            # 2. Daily/Period Growth Rate (The slope of your regression line)
            # Difference between last and first point of the forecast period
            forecast_start = df_combined['Reg_future'].iloc[len(df)]
            forecast_end = df_combined['Reg_future'].iloc[-1]
            total_growth = ((forecast_end - forecast_start) / forecast_start) * 100

            with st.container(border=True):
                st.plotly_chart(fig3, use_container_width=True, key="plot_reg")
                # Layout metrics in 3 columns
                r1, r2_col, r3 = st.columns(3)
                with r1:
                    st.metric("Accuracy", f"{acc:.1f}%")
                with r2_col:
                    # R2 above 0.7 is generally considered a "strong" fit
                    st.metric("Fit Score (R²)", f"{r2:.2f}")
                with r3:
                    # Shows the projected growth of the regression line
                    st.metric("Forecast Growth", f"{total_growth:.1f}%", delta=f"{total_growth:.1f}%")

        with col4:
            fig4 = plot_plotly(model, forecast) # Interactive Plotly chart
            fig4.update_layout(height=500,
                               title= f"Prophet Forecast for {kpi_type} - {selected} ({granularity})", 
                               showlegend=False, xaxis_title=None, yaxis_title=None, xaxis_rangeslider_visible=False, xaxis=dict(rangeselector=None))
            #Calculate KPIs
            acc = get_accuracy(forecast['trend'], forecast['yhat'])
            # 2. Seasonality Strength (How much of the movement is 'pattern-based')     # We compare the seasonal component to the overall trend
            seasonal_effect=0
            if granularity in ['Daily', 'Weekly']:
                seasonal_effect = (forecast['yearly'].abs().mean() if 'yearly' in forecast else 0)
            # 3. Forecast Uncertainty (The width of the blue shaded area)               # Higher spread = Higher risk in the forecast
            last_p = forecast.iloc[-1]
            uncertainty_spread = ((last_p['yhat_upper'] - last_p['yhat_lower']) / last_p['yhat']) * 100

            with st.container(border=True):
                st.plotly_chart(fig4, use_container_width=True, key="plot4")
                # Layout metrics in 3 columns
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("Accuracy", f"{acc:.1f}%")
                with p2:
                    # Tells user if patterns (holidays/days of week) are driving the numbers
                    st.metric("Seasonality Impact", f"{seasonal_effect:.1f}")
                with p3:
                    # Lower is better. High % means the model is 'guessing' more at the end
                    st.metric("Forecast Risk", f"{uncertainty_spread:.1f}%", delta="Uncertainty", delta_color="inverse")

        # 6. Data Downloads
        st.sidebar.download_button("Download: MA/ES/Reg CSV", 
                                   df_combined.to_csv(index=False), 
                                   "ma_es_reg_results.csv")
        st.sidebar.download_button("Download: Prophet CSV", 
                                   forecast.to_csv(index=False), 
                                   "prophet_results.csv")
    else:
        st.error("No data found for this category.")
else:
    st.info("Please upload a CSV file to get started.")
