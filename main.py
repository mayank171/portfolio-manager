import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
from mftool import Mftool
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import re
import math
import requests



def fetch_nav_data(schema_code):
    url = f"https://www.amfiindia.com/net-asset-value/nav-history/{scheme_code}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    nav_data = []
    for row in soup.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        date = cols[0].text.strip()
        nav = cols[1].text.strip()
        nav_data.append((date, nav))
    return nav_data



def fv_fixed_income_realistic(principal, rate, years, monthly_contrib=0, employer_contrib=0):
    """
    principal: initial balance
    rate: annual return in decimal (e.g., 0.08)
    years: projection horizon
    monthly_contrib: your monthly contribution
    employer_contrib: company monthly contribution
    """
    values = []
    total = principal
    monthly_rate = rate / 12
    total_monthly_contrib = monthly_contrib + employer_contrib

    for month in range(1, years*12 + 1):
        total = total * (1 + monthly_rate)  # interest for the month
        total += total_monthly_contrib       # add both contributions
        if month % 12 == 0:
            values.append(total)

    return values


def fv_monthly_nps(monthly_contrib, years, rate, employer_contrib=0, principal=0):
    """
    monthly_contrib: your monthly contribution
    employer_contrib: company monthly contribution
    years: projection horizon
    rate: expected annual return in decimal (0.08 for 8%)
    principal: optional starting balance
    """
    total = principal
    monthly_rate = rate / 12
    total_monthly = monthly_contrib + employer_contrib
    values = []

    for month in range(1, years*12 + 1):
        total = total * (1 + monthly_rate)  # monthly compounding
        total += total_monthly
        if month % 12 == 0:
            values.append(total)

    return values





st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("📈 Personal Finance Manager")

tab1, tab2, tab3, tab4 = st.tabs(['PF','NPS','Mutual Funds','Gold'])

mf =Mftool()


with tab1:
    st.markdown(f"<h1 style='text-align: center; color: white;'>PF Projection</h1>", unsafe_allow_html=True)

    
    principal = st.number_input(
        "Current Balance (₹)", min_value=0.0, value=100000.0, step=1000.0
    )
    rate = st.slider(
        "Expected Annual Return (%)", 1.0, 12.0, 8.0
    ) / 100
    years = st.slider("Projection Horizon (Years)", 1, 30, 10)
    monthly_contrib = st.number_input(
        "Your Monthly Contribution (₹)", min_value=0.0, value=1000.0, step=500.0
    )
    employer_contrib = st.number_input(
        "Company/Employer Contribution (₹)", min_value=0.0, value=1000.0, step=500.0
    )

    # Calculate projections
    results = fv_fixed_income_realistic(
        principal, rate, years, monthly_contrib, employer_contrib
    )

    # Create dataframe for plotting
    df = pd.DataFrame({
        "Years": list(range(1, years + 1)),
        "Projected Value": results
    })

    st.subheader("Projected Chart")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Years"],
        y=df["Projected Value"],
        mode="lines+markers",
        name="Projected Value",
        marker=dict(size=8, color="blue"),
        line=dict(color="blue", width=2),
        hovertemplate="Year: %{x}<br>Value: ₹%{y:.2f}"
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Projected Value (₹)",
        
    )

    st.plotly_chart(fig, use_container_width=True)

    total_invest = principal+(+monthly_contrib+employer_contrib)*12*years

    percentage_incr = ((results[-1] - total_invest) / total_invest) * 100


    st.markdown(f"<h2 style='text-align: center; color: white;'>Total Investment: {total_invest:,.2f} ₹</h2>", unsafe_allow_html=True)
    color = "green" if percentage_incr > 0 else "red"
    st.markdown(
        f"""
        <h2 style='text-align: center;'>
            Returns: {results[-1]:,.2f} ₹ &nbsp;&nbsp;
            <span style='color:{color};'>({percentage_incr:,.2f}%)</span>
        </h2>
        """,
        unsafe_allow_html=True
    )




with tab2:
    st.markdown(f"<h1 style='text-align: center; color: white;'>NPS Projection</h1>", unsafe_allow_html=True)
    

    # For now, same inputs
    principal2 = st.number_input("Current Balance (₹)", min_value=0.0, value=0.0, step=1000.0, key="p2")
    monthly_contrib2 = st.number_input("Your Monthly Contribution (₹)", min_value=0.0, value=5000.0, step=500.0, key="mc2")
    employer_contrib2 = st.number_input("Employer Contribution (₹)", min_value=0.0, value=0.0, step=500.0, key="ec2")
    years2 = st.slider("Projection Horizon (Years)", 1, 40, 10, key="y2")
    equity_pct = st.slider("Equity Allocation (%)", 0, 100, 50)
    corp_pct = st.slider("Corporate Bonds Allocation (%)", 0, 100-equity_pct, 30)
    govt_pct = 100 - equity_pct - corp_pct

    # Expected returns (example averages)
    r_equity = 0.10
    r_corp = 0.07
    r_govt = 0.06

    # Weighted expected return
    rate2 = (equity_pct/100)*r_equity + (corp_pct/100)*r_corp + (govt_pct/100)*r_govt
    st.write(f"Weighted expected return: {rate2*100:.2f}%")

    results2 = fv_monthly_nps(monthly_contrib2, years2, rate2, employer_contrib2, principal2)
    df2 = pd.DataFrame({"Years": list(range(1, years2+1)), "Projected Value": results2})

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df2["Years"],
        y=df2["Projected Value"],
        mode="lines+markers",
        name="Projected Value",
        marker=dict(size=8, color="green"),
        line=dict(color="green", width=2),
        hovertemplate="Year: %{x}<br>Value: ₹%{y:.2f}"
    ))
    fig2.update_layout(
        xaxis_title="Year",
        yaxis_title="Projected Value (₹)",
        
    )
    st.plotly_chart(fig2, use_container_width=True)

    total_invest = (monthly_contrib2+employer_contrib2)*12*years2

    percentage_incr = ((results2[-1] - total_invest) / total_invest) * 100


    st.markdown(f"<h2 style='text-align: center; color: white;'>Total Investment: {total_invest:,.2f} ₹</h2>", unsafe_allow_html=True)
    color = "green" if percentage_incr > 0 else "red"
    st.markdown(
        f"""
        <h2 style='text-align: center;'>
            Returns: {results2[-1]:,.2f} ₹ &nbsp;&nbsp;
            <span style='color:{color};'>({percentage_incr:,.2f}%)</span>
        </h2>
        """,
        unsafe_allow_html=True
    )


with tab3:

    
    def infer_periods_per_year(dates: pd.Series):
        diffs = dates.sort_values().diff().dropna().dt.days
        median_days = diffs.median()
        if median_days<=1.5:
            return 252
        if median_days<=9:
            return 52
        if median_days<=40:
            return 12
        return 1

    def prepare_returns(df, date_col='date', nav_col='nav'):
        df = df[[date_col, nav_col]].dropna().copy()
        df[date_col]=pd.to_datetime(df[date_col], format='%d-%m-%Y')
        df=df.sort_values(date_col)
        df=df.drop_duplicates(subset=date_col)
        df['log_ret']=np.log(df[nav_col]).diff()
        df=df.dropna(subset=['log_ret'])
        ppy=infer_periods_per_year(df[date_col])
        return df,ppy

    def annualized_stats(log_returns, periods_per_year):
        mu_period = log_returns.mean()
        sigma_period = log_returns.std(ddof=1)
        mu_annual = mu_period*periods_per_year
        sigma_annual = sigma_period*np.sqrt(periods_per_year)
        return mu_annual, sigma_annual

    
    def simulate_gbm_paths(S0, mu, sigma, years, periods_per_year, n_sims=10000, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        steps = int(years*periods_per_year)
        dt = 1.0/periods_per_year

        drift = (mu-0.5*sigma**2)*dt
        diffusion = sigma*np.sqrt(dt)

        Z = np.random.normal(size=(steps, n_sims))
        increments = drift + diffusion*Z
        log_paths = np.cumsum(increments, axis=0)

        log_paths = np.vstack([np.zeros(n_sims), log_paths])
        price_paths = S0*np.exp(log_paths)
        return price_paths

    
    def summarize_and_plots(price_paths, dates_future, percentiles=(5,25,50,75,95), title=None):
        # Calculate percentiles
        pct = np.percentile(price_paths, percentiles, axis=1)
        median = np.percentile(price_paths, 50, axis=1)
        final_vals = price_paths[-1, :]
        
        # Summary stats
        summary = {
            'median_final': np.median(final_vals),
            'p5_final': np.percentile(final_vals, 5),
            'p25_final': np.percentile(final_vals, 25),
            'p75_final': np.percentile(final_vals, 75),
            'p95_final': np.percentile(final_vals, 95),
        }

        # Create interactive figure
        fig = go.Figure()

        # Percentile bands
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[-1],
            line=dict(color='skyblue', width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[0],
            fill='tonexty',
            fillcolor='rgba(135,206,250,0.2)',
            line=dict(color='skyblue', width=0),
            name=f'{percentiles[0]}-{percentiles[-1]} pct band'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[-2],
            line=dict(color='dodgerblue', width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[1],
            fill='tonexty',
            fillcolor='rgba(30,144,255,0.25)',
            line=dict(color='dodgerblue', width=0),
            name=f'{percentiles[1]}-{percentiles[-2]} pct band'
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=dates_future, y=median,
            mode='markers',
            name='Median',
            line=dict(color='white'),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}'
        ))

        # Sample paths (up to 50)
        for i in range(0, min(50, price_paths.shape[1]), max(1, price_paths.shape[1]//50)):
            fig.add_trace(go.Scatter(
                x=dates_future, y=price_paths[:, i],
                mode='lines',
                line=dict(color='white', width=1),
                name='Sim path' if i == 0 else None,
                hoverinfo='x+y'
            ))

        # Layout
        fig.update_layout(
            autosize=True,
            #responsive=True,
            #title=title or 'Monte Carlo Price Simulation',
            xaxis_title='Date',
            yaxis_title='NAV / Portfolio Value',
            hovermode='closest',
            
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=10, r=10, t=40, b=40),
        )

        fig.update_xaxes(rangeslider_visible=True)

        config = {
            'scrollZoom': True,
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'displaylogo': False
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
        return summary, final_vals

    

    def sip_future_value(p, years, annual_rate):
        n = years * 12
        r = annual_rate / 12
        fv = p * (((1 + r)**n - 1) / r) * (1 + r)
        return fv
        







    st.markdown(f"<h1 style='text-align: center; color: white;'>Mutual Fund Projection</h1>", unsafe_allow_html=True)

    # Read file content
    with open("famous_mutual_funds.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Use regex to extract all names inside single quotes
    mf_list = re.findall(r"'([^']+)'", text)

    default_mf = "360 ONE Balanced Hybrid Fund- Direct Plan - Growth"

    # --- Check if default exists, else use first item ---
    default_index = mf_list.index(default_mf) if default_mf in mf_list else 0

    selected_fund = st.selectbox(
        "Select a Mutual Fund (type to search):",
        options=mf_list,
        index=default_index,
        placeholder="Start typing to search..."
    )
    
    result = mf.get_available_schemes(selected_fund)

    scheme_code = next((code for code, name in result.items() if name == selected_fund), None)
    df = mf.get_scheme_historical_nav(scheme_code,as_Dataframe=True)
        
    date_col = 'date'
    nav_col = 'nav'
    df.reset_index(inplace=True)
    df['nav'] = df['nav'].astype(float)

    df,ppy = prepare_returns(df, date_col=date_col, nav_col=nav_col)
    mu, sigma = annualized_stats(df['log_ret'], ppy)


    years = years = st.slider("Projection Horizon (Years)", 1, 20, 2)
    sims = 10000
    S0 = df.iloc[-1][nav_col]
    
    price_path = simulate_gbm_paths(S0, mu, sigma, years, ppy, n_sims=sims, random_seed=42)

    last_date = pd.to_datetime(df.iloc[-1][date_col])
    median_delta_days = df[date_col].sort_values().diff().dropna().dt.days.median()
    steps = int(years*ppy)
    dates_future = [last_date+timedelta(days = int(median_delta_days*i)) for i in range(steps+1)]
    summary, vals = summarize_and_plots(price_path, dates_future, title=f'{years}-yr Monte Carlo ({sims} sims)')

    for k,v in summary.items():
        st.markdown(f"<h2 style='text-align: center; color: dodgerblue;'>Median price after {years} years: {v:,.2f} ₹</h2>", unsafe_allow_html=True)
        break

    st.markdown("---")
    st.markdown(f"<h1 style='text-align: center; color: white;'>Calculate Returns</h1>", unsafe_allow_html=True)


    # Monthly SIP input
    monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0.0, max_value=50000.0, value=1000.0, step=100.0)

    # Step-up percentage per year
    step_up_percent = st.number_input("Yearly Step-up (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    # Years input
    years = st.number_input("Investment Duration (Years)", min_value=1, max_value=40, value=10, step=1)
    months = years * 12

    nav_start = df['nav'][0]
    nav_end = price_path[-1][-1]

    # Linear growth of NAV
    navs = [nav_start + (nav_end - nav_start) / months * i for i in range(1, months + 1)]

    # Step-up logic
    total_units = 0
    current_sip = monthly_sip
    for month in range(1, months + 1):
        # Add units for this month
        total_units += current_sip / navs[month - 1]

        # Increase SIP at the start of each year
        if month % 12 == 0:
            current_sip *= (1 + step_up_percent / 100)

    # Final calculations
    future_value = total_units * nav_end
    total_invest = 0
    current_sip = monthly_sip
    for year in range(years):
        total_invest += current_sip * 12
        current_sip *= (1 + step_up_percent / 100)

    percentage_incr = ((future_value - total_invest) / total_invest) * 100

    # Display
    st.markdown(f"<h2 style='text-align: center; color: white;'>Total Investment: {total_invest:,.2f} ₹</h2>", unsafe_allow_html=True)
    color = "green" if percentage_incr > 0 else "red"
    st.markdown(
        f"""
        <h2 style='text-align: center;'>
            Returns: {future_value:,.2f} ₹ &nbsp;&nbsp;
            <span style='color:{color};'>({percentage_incr:,.2f}%)</span>
        </h2>
        """,
        unsafe_allow_html=True
    )



with tab4:


    def infer_periods_per_year(dates: pd.Series):
        diffs = dates.sort_values().diff().dropna().dt.days
        median_days = diffs.median()
        if median_days<=1.5:
            return 252
        if median_days<=9:
            return 52
        if median_days<=40:
            return 12
        return 1

    def prepare_returns(df, date_col='date', price_col='price'):
        df = df[[date_col, price_col]].dropna().copy()
        df[date_col]=pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df=df.sort_values(date_col)
        df=df.drop_duplicates(subset=date_col)
        df['log_ret']=np.log(df[price_col]).diff()
        df=df.dropna(subset=['log_ret'])
        ppy=infer_periods_per_year(df[date_col])
        return df,ppy

    def annualized_stats(log_returns, periods_per_year):
        mu_period = log_returns.mean()
        sigma_period = log_returns.std(ddof=1)
        mu_annual = mu_period*periods_per_year
        sigma_annual = sigma_period*np.sqrt(periods_per_year)
        return mu_annual, sigma_annual

    
    def simulate_gbm_paths(S0, mu, sigma, years, periods_per_year, n_sims=10000, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        steps = int(years*periods_per_year)
        dt = 1.0/periods_per_year

        drift = (mu-0.5*sigma**2)*dt
        diffusion = sigma*np.sqrt(dt)

        Z = np.random.normal(size=(steps, n_sims))
        increments = drift + diffusion*Z
        log_paths = np.cumsum(increments, axis=0)

        log_paths = np.vstack([np.zeros(n_sims), log_paths])
        price_paths = S0*np.exp(log_paths)
        return price_paths

    
    def summarize_and_plots(price_paths, dates_future, percentiles=(5,25,50,75,95), title=None):
        # Calculate percentiles
        pct = np.percentile(price_paths, percentiles, axis=1)
        median = np.percentile(price_paths, 50, axis=1)
        final_vals = price_paths[-1, :]
        
        # Summary stats
        summary = {
            'median_final': np.median(final_vals),
            'p5_final': np.percentile(final_vals, 5),
            'p25_final': np.percentile(final_vals, 25),
            'p75_final': np.percentile(final_vals, 75),
            'p95_final': np.percentile(final_vals, 95),
        }

        # Create interactive figure
        fig = go.Figure()

        # Percentile bands
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[-1],
            line=dict(color='skyblue', width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[0],
            fill='tonexty',
            fillcolor='rgba(135,206,250,0.2)',
            line=dict(color='skyblue', width=0),
            name=f'{percentiles[0]}-{percentiles[-1]} pct band'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[-2],
            line=dict(color='dodgerblue', width=0),
            showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates_future, y=pct[1],
            fill='tonexty',
            fillcolor='rgba(30,144,255,0.25)',
            line=dict(color='dodgerblue', width=0),
            name=f'{percentiles[1]}-{percentiles[-2]} pct band'
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=dates_future, y=median,
            mode='markers',
            name='Median',
            line=dict(color='white'),
            hovertemplate='Date: %{x}<br>Value: %{y:.2f}'
        ))

        # Sample paths (up to 50)
        for i in range(0, min(50, price_paths.shape[1]), max(1, price_paths.shape[1]//50)):
            fig.add_trace(go.Scatter(
                x=dates_future, y=price_paths[:, i],
                mode='lines',
                line=dict(color='white', width=1),
                name='Sim path' if i == 0 else None,
                hoverinfo='x+y'
            ))

        # Layout
        fig.update_layout(
            autosize=True,
            #responsive=True,
            #title=title or 'Monte Carlo Price Simulation',
            xaxis_title='Date',
            yaxis_title='NAV / Portfolio Value',
            hovermode='closest',
            
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=10, r=10, t=40, b=40),
        )

        fig.update_xaxes(rangeslider_visible=True)

        config = {
            'scrollZoom': True,
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'displaylogo': False
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
        return summary, final_vals


    st.markdown("<h1 style='text-align: center; color: white;'>Gold Projection</h1>", unsafe_allow_html=True)

    df = pd.read_csv('Gold Price.csv')
    
    df=df.drop(['Open','High','Low','Volume','Chg%'],axis=1)
    #st.write(df.head())

    date_col = 'Date'
    price_col = 'Price'

    df['Price'] = df['Price'].astype(float)
    #st.write(df[price_col])

    df,ppy = prepare_returns(df, date_col=date_col, price_col=price_col)
    mu, sigma = annualized_stats(df['log_ret'], ppy)


    yrs = st.slider("Projection Horizon(Yrs)",1,20,5)
    sims = 10000
    S0 = df.iloc[-1][price_col]
    
    price_path = simulate_gbm_paths(S0, mu, sigma, yrs, ppy, n_sims=sims, random_seed=42)
    #st.write(price_path)

    last_date = pd.to_datetime(df.iloc[-1][date_col])
    median_delta_days = df[date_col].sort_values().diff().dropna().dt.days.median()
    steps = int(years*ppy)
    dates_future = [last_date+timedelta(days = int(median_delta_days*i)) for i in range(steps+1)]
    summary, vals = summarize_and_plots(price_path, dates_future, title=f'{yrs}-yr Monte Carlo ({sims} sims)')

    for k,v in summary.items():
        st.markdown(f"<h2 style='text-align: center; color: dodgerblue;'>Median price after {yrs} years: {v:,.2f} ₹</h2>", unsafe_allow_html=True)
        break


    # --- Session State to hold investments (in-memory only) ---
    if "investments" not in st.session_state:
        st.session_state.investments = pd.DataFrame(columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"])

    # --- Fetch live gold price ---
    @st.cache_data(ttl=3600)
    def get_live_gold_price():
        url = "https://www.goldapi.io/api/XAU/INR"
        headers = {
            "x-access-token": "goldapi-4g9e8p7smgug4cig-io",
            "Content-Type": "application/json"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return {
                "24K": round(data.get("price_gram_24k", 0), 2),
                "22K": round(data.get("price_gram_22k", 0), 2),
                "timestamp": data.get("date")
            }
        except Exception as e:
            st.error(f"⚠️ Error fetching gold price: {e}")
            return None

    gold_price = get_live_gold_price()
    if gold_price:
        st.success(f"💰 24K Gold: ₹{gold_price['24K']}/g | 22K Gold: ₹{gold_price['22K']}/g")
        st.caption(f"Updated on: {gold_price['timestamp']}")
        current_price = gold_price["24K"]
    else:
        st.warning("Could not fetch live gold price right now.")
        current_price = 0

    st.divider()

    import streamlit as st
    import pandas as pd
    from datetime import datetime

    # Initialize table if not present
    if "investments" not in st.session_state:
        st.session_state.investments = pd.DataFrame(
            columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"]
        )

    # ADD NEW INVESTMENT
    st.subheader("Add New Investment")

    with st.form("add_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date = st.date_input("Purchase Date", datetime.today())
        with col2:
            gold_type = st.selectbox("Type", ["Physical", "ETF", "Digital"])
        with col3:
            qty = st.number_input("Quantity (grams)", min_value=0.1, step=0.1)
        with col4:
            buy_price = st.number_input("Buy Price (₹/g)", min_value=0.0, step=0.1)
        add = st.form_submit_button("Add Investment")

    if add and qty > 0:
        new_entry = pd.DataFrame(
            [[date, gold_type, qty, buy_price]],
            columns=st.session_state.investments.columns
        )
        st.session_state.investments = pd.concat(
            [st.session_state.investments, new_entry],
            ignore_index=True
        )
        st.success("✅ Investment added successfully!")

    # DELETE INVESTMENT(S)
    st.divider()
    st.subheader("Delete Investment")

    df = st.session_state.investments.copy()

    if not df.empty:
        # Create label for selection
        df["Label"] = (
            df["Purchase Date"].astype(str)
            + " | " + df["Type"]
            + " | " + df["Quantity (g)"].astype(str) + "g"
        )

        to_delete = st.multiselect(
            "Select investment(s) to delete:",
            df["Label"].tolist()
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected", type="primary"):
                if to_delete:
                    new_df = df[~df["Label"].isin(to_delete)].drop(columns=["Label"])
                    st.session_state.investments = new_df
                    st.success(f"✅ Deleted {len(to_delete)} investment(s) successfully!")
                    st.rerun()
                else:
                    st.warning("⚠ Please select at least one investment to delete.")
        with col2:
            if st.button("🧹 Clear All Investments"):
                st.session_state.investments = pd.DataFrame(
                    columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"]
                )
                st.success("🗑 All investments cleared.")
                st.rerun()
    else:
        st.info("No investments yet. Add one above to get started!")

    

    # --- Portfolio Summary ---
    df = st.session_state.investments.copy()

    if not df.empty:
        df["Invested Value"] = df["Quantity (g)"] * df["Buy Price (₹/g)"]
        df["Current Value"] = df["Quantity (g)"] * current_price
        df["Profit/Loss"] = df["Current Value"] - df["Invested Value"]
        df["% Gain/Loss"] = (df["Profit/Loss"] / df["Invested Value"]) * 100
        df["Years Held"] = (datetime.today() - pd.to_datetime(df["Purchase Date"], errors="coerce")).dt.days / 365
        df["CAGR (%)"] = ((df["Current Value"] / df["Invested Value"]) ** (1 / df["Years Held"]) - 1) * 100
        df["CAGR (%)"] = df["CAGR (%)"].replace([math.inf, -math.inf], 0).fillna(0)

        total_invested = df["Invested Value"].sum()
        total_current = df["Current Value"].sum()
        total_gain = total_current - total_invested
        gain_percent = (total_gain / total_invested) * 100 if total_invested > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invested", f"₹{total_invested:,.0f}")
        col2.metric("Current Value", f"₹{total_current:,.0f}")
        col3.metric("Profit/Loss", f"₹{total_gain:,.0f}", f"{gain_percent:.2f}%")

        st.subheader("📄 Investment Details")
        st.dataframe(df.style.format({
            "Buy Price (₹/g)": "{:.2f}",
            "Current Value": "{:.2f}",
            "Profit/Loss": "{:.2f}"
        }))

        st.subheader("📈 Portfolio by Type")
        pie_fig = px.pie(df, names="Type", values="Current Value", title="Distribution by Type")
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No investments yet. Add one above to get started!")
