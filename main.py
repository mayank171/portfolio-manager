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
import json



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

tab1, tab2, tab3, tab4, tab5 = st.tabs(['PF','NPS','Mutual Funds','Gold','Silver'])

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

    pf_proj = {"principle":principal, "rate:":rate, "years": years, "montly contribution": monthly_contrib, "employee contribution": employer_contrib, "total investment:":total_invest, "returns: ":results[-1]}

    user_query = st.text_area("Ask AI about PF:", key="pf_query")
    if st.button("Ask PF Assistant"):
        context = f"PF Projection:\n{json.dumps(pf_proj, indent=2)}"
        prompt = f"{context}\n\nUser query: {user_query}"
        res = requests.post("http://localhost:8000/ask", json={"text": prompt})
        st.markdown("### 💬 AI Response:")
        st.write(res.json().get("response", "No response"))




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
            title=dict(
                text=title or 'Monte Carlo Price Simulation',
                x=0.5, xanchor='center',
                font=dict(size=16, color='white')
            ),
            xaxis_title='Date',
            yaxis_title='NAV / Portfolio Value',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.0,
                xanchor="center", x=0.5,
                font=dict(size=10, color='white')
            ),
            margin=dict(l=20, r=20, t=60, b=60),
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            showgrid=False,
            tickformat="%b\n%Y",
            rangeslider=dict(
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                thickness=0.1,        # default is ~0.15 → thinner slider
                #yanchor="bottom",     # ensures slider hugs the bottom
            )
        )

        # Add some extra bottom space visually
        fig.update_layout(
            margin=dict(l=20, r=20, t=150, b=100)  # increase bottom margin
        )

        # --- Config for Streamlit ---
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

    # ---------------- FUND ANALYTICS SECTION ---------------- #

    st.markdown("---")
    st.markdown("<h1 style='text-align:center; color:#F5F5F5;'>📊 Fund Analytics</h1>", unsafe_allow_html=True)

    # --- Compute fund’s annualized return ---
    fund_return = mu * 100  # %

    # --- Category classification ---
    if "Large" in selected_fund:
        benchmark_return, peer_avg = 10.0, 9.0
    elif "Mid" in selected_fund:
        benchmark_return, peer_avg = 11.0, 10.0
    elif "Small" in selected_fund:
        benchmark_return, peer_avg = 12.0, 11.0
    else:
        benchmark_return, peer_avg = 9.0, 8.5

    # --- Classification Logic ---
    def classify_performance(fund_return, benchmark_return, peer_avg, tolerance=1.0):
        if fund_return > benchmark_return + tolerance and fund_return > peer_avg + tolerance:
            return "Outperforming"
        elif fund_return < benchmark_return - tolerance and fund_return < peer_avg - tolerance:
            return "Underperforming"
        else:
            return "Inline"

    performance_status = classify_performance(fund_return, benchmark_return, peer_avg)

    # --- Color Map ---
    color_map = {"Outperforming": "#32CD32", "Inline": "#FFA500", "Underperforming": "#FF4C4C"}
    color = color_map.get(performance_status, "white")

    # --- Summary Block (Centered, Compact) ---
    st.markdown(f"""
    <div style='background-color:#1E1E1E; border-radius:12px; padding:18px; text-align:center;'>
        <h3 style='color:white; margin-bottom:10px;'>Annualized Return</h3>
        <h2 style='color:dodgerblue; margin-top:0;'>{fund_return:.2f}%</h2>
        <p style='color:lightgray;'>Benchmark: <b>{benchmark_return:.2f}%</b> &nbsp; | &nbsp; Peer Avg: <b>{peer_avg:.2f}%</b></p>
        <h3 style='color:{color}; margin-top:10px;'>Performance: {performance_status}</h3>
    </div>
    """, unsafe_allow_html=True)

    # --- Return Comparison Chart ---
    # fig_perf = go.Figure()
    # fig_perf.add_trace(go.Bar(
    #     x=["Fund", "Benchmark", "Peer Avg"],
    #     y=[fund_return, benchmark_return, peer_avg],
    #     marker_color=["dodgerblue", "gray", "orange"],
    #     text=[f"{fund_return:.1f}%", f"{benchmark_return:.1f}%", f"{peer_avg:.1f}%"],
    #     textposition="auto"
    # ))
    # fig_perf.update_layout(
    #     title="📈 Return Comparison (%)",
    #     template="plotly_white",
    #     height=320,
    #     margin=dict(l=10, r=10, t=60, b=10),
    #     yaxis_title="Annualized Return (%)",
    # )
    # st.plotly_chart(fig_perf, use_container_width=True)

    # --- RISK VS REWARD SECTION ---
    st.markdown("<h2 style='text-align:center; color:#F5F5F5;'>Risk vs Reward</h2>", unsafe_allow_html=True)

    risk_free_rate = 0.06
    volatility_annual = sigma * 100
    sharpe_ratio = (mu - risk_free_rate) / sigma if sigma != 0 else np.nan
    downside_returns = df.loc[df["log_ret"] < 0, "log_ret"]
    downside_std = downside_returns.std(ddof=1) * np.sqrt(ppy) if not downside_returns.empty else np.nan
    sortino_ratio = (mu - risk_free_rate) / downside_std if downside_std else np.nan

    # --- Friendly Text Cards ---
    def describe_volatility(vol):
        if vol < 10: return "🟢 Very stable — low ups & downs."
        if vol < 20: return "🟡 Moderate swings."
        return "🔴 Quite volatile."

    def describe_sharpe(s): 
        if s > 2: return "🚀 Excellent risk-adjusted return."
        if s > 1: return "✅ Good balance between risk and reward."
        if s > 0: return "⚖️ Average performance."
        return "⚠️ Weak risk-adjusted return."

    def describe_sortino(s):
        if s > 2: return "👍 Great downside protection."
        if s > 1: return "🙂 Handles market dips well."
        return "😕 Sensitive to losses."

    metrics = [
        ("Volatility (%)", volatility_annual, describe_volatility(volatility_annual)),
        ("Sharpe Ratio", sharpe_ratio, describe_sharpe(sharpe_ratio)),
        ("Sortino Ratio", sortino_ratio, describe_sortino(sortino_ratio))
    ]

    cols = st.columns(3)
    for i, (label, value, desc) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div style='background-color:#1E1E1E; border-radius:12px; padding:15px; text-align:center;'>
                <h4 style='color:white; margin-bottom:5px;'>{label}</h4>
                <h2 style='color:dodgerblue; margin-top:0;'>{value:.2f}</h2>
                <p style='color:lightgray; font-size:14px;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Risk Overview Chart ---
    # fig_risk = go.Figure()
    # fig_risk.add_trace(go.Bar(
    #     x=["Volatility", "Sharpe", "Sortino"],
    #     y=[volatility_annual, sharpe_ratio * 10, sortino_ratio * 10],
    #     text=[f"{volatility_annual:.1f}%", f"{sharpe_ratio:.2f}", f"{sortino_ratio:.2f}"],
    #     textposition="auto",
    #     marker_color=["orange", "dodgerblue", "limegreen"]
    # ))
    # fig_risk.update_layout(
    #     title="⚖️ Risk-Reward Overview",
    #     template="plotly_white",
    #     height=320,
    #     margin=dict(l=10, r=10, t=60, b=10)
    # )
    # st.plotly_chart(fig_risk, use_container_width=True)

    # --- Stability & Trend Analysis ---
    df["rolling_mean"] = df["nav"].rolling(6).mean()
    df["cummax"] = df["nav"].cummax()
    df["drawdown"] = (df["nav"] - df["cummax"]) / df["cummax"] * 100
    max_drawdown = df["drawdown"].min()
    positive_trends = (df["rolling_mean"].diff() > 0).sum()
    stability_score = positive_trends / len(df["rolling_mean"].dropna()) * 100

    if stability_score > 80:
        mood, msg = "😃 Steady Growth", "Strong and consistent growth trend."
    elif stability_score > 60:
        mood, msg = "🙂 Generally Stable", "Healthy performance with minor fluctuations."
    elif stability_score > 40:
        mood, msg = "😐 Fluctuating", "Frequent swings — moderate consistency."
    else:
        mood, msg = "😟 Volatile", "Performance unpredictable — high risk."

    st.markdown(f"<h2 style='text-align:center; color:white;'>{mood}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:lightgray;'>{msg}</p>", unsafe_allow_html=True)
    st.line_chart(df[["nav", "rolling_mean"]], height=220, use_container_width=True)

    # --- Relative Performance ---
    # --- Data section ---
    user_return = (np.exp(mu) - 1) * 100
    benchmark_return = 10.2
    peer_avg_return = 9.8

    vs_benchmark = user_return - benchmark_return
    vs_peers = user_return - peer_avg_return

    def describe_performance(diff):
        if diff > 3:
            return ("🏆 Outperforming", "#4CAF50", "Your fund is ahead of most peers.")
        if diff > 0:
            return ("👍 Slightly Ahead", "#8BC34A", "Better than average.")
        if diff > -2:
            return ("😐 Average", "#FFC107", "Close to market levels.")
        return ("⚠️ Lagging", "#F44336", "Underperforming — review strategy.")

    bench_label, bench_color, bench_msg = describe_performance(vs_benchmark)
    peer_label, peer_color, peer_msg = describe_performance(vs_peers)

    # --- Card UI section ---
    st.markdown(f"""
    <div style="
        text-align:center;
        color:white;
        font-size:17px;
        background:linear-gradient(145deg, #232526, #414345);
        border-radius:16px;
        padding:25px;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
        border:1px solid rgba(255,255,255,0.1);
    ">
        <div style="margin-bottom:20px;">
            <div style="font-size:20px; font-weight:600; color:{bench_color};">{bench_label}</div>
            <div style="font-size:22px; font-weight:700;">Benchmark: {vs_benchmark:+.2f}%</div>
            <div style="color:#cccccc; font-size:15px;">{bench_msg}</div>
            <div style="background:rgba(255,255,255,0.1); border-radius:8px; height:8px; width:80%; margin:10px auto;">
                <div style="background:{bench_color}; width:{min(max((vs_benchmark+5)*5,5),100)}%; height:8px; border-radius:8px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown(f"""
        <hr style="border:0.5px solid rgba(255,255,255,0.2); margin:15px 0;">
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="
        text-align:center;
        color:white;
        font-size:17px;
        background:linear-gradient(145deg, #232526, #414345);
        border-radius:16px;
        padding:25px;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
        border:1px solid rgba(255,255,255,0.1);
    ">
        <div>
            <div style="font-size:20px; font-weight:600; color:{peer_color};">{peer_label}</div>
            <div style="font-size:22px; font-weight:700;">Peers: {vs_peers:+.2f}%</div>
            <div style="color:#cccccc; font-size:15px;">{peer_msg}</div>
            <div style="background:rgba(255,255,255,0.1); border-radius:8px; height:8px; width:80%; margin:10px auto;">
                <div style="background:{peer_color}; width:{min(max((vs_peers+5)*5,5),100)}%; height:8px; border-radius:8px;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)




    # ---------------- RAG Integration ----------------

    import streamlit as st
    import json
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # ---------- 1) Build / load a small knowledge base ----------
    KB = [
        # --- Risk-adjusted metrics ---
        {"id": "kb_sharpe_1", "title": "Sharpe Ratio meaning",
        "text": "The Sharpe ratio measures excess return per unit of total risk (standard deviation). It helps compare funds on a risk-adjusted basis. A higher Sharpe ratio generally indicates better risk-adjusted performance."},

        {"id": "kb_sortino_1", "title": "Sortino Ratio meaning",
        "text": "Sortino ratio refines Sharpe by penalizing only downside volatility. It’s preferred when investors care more about losses than total fluctuations."},

        {"id": "kb_treynor_1", "title": "Treynor Ratio meaning",
        "text": "Treynor ratio measures excess return per unit of systematic risk (beta). It’s useful when comparing diversified portfolios exposed mainly to market risk."},

        {"id": "kb_alpha_1", "title": "Alpha definition",
        "text": "Alpha measures how much a fund outperforms its benchmark after accounting for risk. A positive alpha means the manager added value beyond market exposure."},

        {"id": "kb_beta_1", "title": "Beta meaning",
        "text": "Beta shows how sensitive the fund is to market movements. A beta > 1 means the fund moves more than the market, beta < 1 means it’s less volatile."},

        # --- Portfolio construction ---
        {"id": "kb_diversification_1", "title": "Diversification importance",
        "text": "Diversification reduces unsystematic risk. By spreading investments across asset classes and sectors, you reduce the impact of any single underperformer."},

        {"id": "kb_allocation_1", "title": "Asset allocation strategy",
        "text": "Strategic allocation defines long-term risk-reward balance. Tactical allocation adjusts weights based on market cycles."},

        {"id": "kb_rebalance_1", "title": "Rebalancing Tip",
        "text": "Rebalancing yearly or when deviations exceed 5% keeps portfolio risk aligned with goals. It forces ‘buy low, sell high’ behavior."},

        # --- Fees and costs ---
        {"id": "kb_fees_1", "title": "Expense Ratio impact",
        "text": "A 1% higher expense ratio can erode up to 25% of total wealth over 20 years. Opt for low-cost index funds unless active alpha is proven."},

        {"id": "kb_turnover_1", "title": "Turnover ratio meaning",
        "text": "High portfolio turnover may indicate aggressive trading, leading to higher transaction costs and potential tax inefficiency."},

        # --- Performance comparison ---
        {"id": "kb_benchmark_1", "title": "Benchmark comparison",
        "text": "Comparing fund returns with a benchmark helps gauge relative performance. Consistent outperformance suggests good management skill."},

        {"id": "kb_volatility_1", "title": "Volatility explanation",
        "text": "Volatility indicates variability of returns. High volatility may signal higher potential gains but also greater uncertainty."},

        # --- Behavioral & strategy tips ---
        {"id": "kb_behavior_1", "title": "Investor behavior",
        "text": "Avoid chasing short-term returns. Evaluate risk-adjusted consistency over 3–5 years instead of recent one-year performance."},

        {"id": "kb_timehorizon_1", "title": "Time horizon importance",
        "text": "Short-term volatility matters less for long-term investors. Choose fund types (equity, hybrid, debt) based on investment horizon."},

        {"id": "kb_risk_1", "title": "Types of risk",
        "text": "Total risk = Systematic (market) + Unsystematic (stock-specific). Diversification reduces only unsystematic risk."}
    ]




    def retrieve_context(query, kb):
        query_lower = query.lower()
        results = [entry["text"] for entry in kb if any(word in query_lower for word in entry["title"].lower().split())]
        return "\n".join(results[:3]) 

    from groq import Groq
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))



    query = "Give me detailed mutual fund performance insights and one personalized improvement suggestion."

    context = retrieve_context(query, KB)

    runtime_context = f"""
    Current Fund Analytics:
    - Fund Name: {selected_fund}
    - Portfolio Return: {user_return:.2f}%
    - Benchmark Return: {benchmark_return:.2f}%
    - Sharpe Ratio: {sharpe_ratio:.2f}
    - Sortino Ratio: {sortino_ratio:.2f}
    - Annual Volatility: {volatility_annual:.2f}%
    """

    rag_prompt = f"""
    [Knowledge Base Insights]
    {context}

    [Portfolio Data]
    {runtime_context}

    [Task for the AI Assistant]
    Summarize how the fund has performed relative to its benchmark.
    Provide 800-word analysis:

    PERFORMANCE (2-3 sentences)
    RISK FACTORS (3-4 bullets)
    WHY UNDERPERFORMING (3 reasons)
    ACTION: Hold/Sell/Add + ONE reason
    TRACK MONTHLY (3 metrics)

    Simple language. Direct. No jargon.

    Keep technical explanations simple but comprehensive. Use bullet points where appropriate.
    Use simple language that a non-finance investor can understand.
    """


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {"role": "system", "content": "You are a financial RAG assistant. Always explain portfolio analytics in plain English with simple actionable suggestions. Avoid jargon."},
            {"role": "user", "content": rag_prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )

    rag_insight = response.choices[0].message.content

    st.markdown("---")
    st.markdown("<h1 style='text-align:center; color:#F5F5F5;'>🤖 AI Insights</h1>", unsafe_allow_html=True)
    st.write(rag_insight)







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
            title=dict(
                text=title or 'Monte Carlo Price Simulation',
                x=0.5, xanchor='center',
                font=dict(size=16, color='white')
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.0,
                xanchor="center", x=0.5,
                font=dict(size=10, color='white')
            ),
            margin=dict(l=20, r=20, t=60, b=60),
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            showgrid=False,
            tickformat="%b\n%Y",
            rangeslider=dict(
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                thickness=0.1,        # default is ~0.15 → thinner slider
                #yanchor="bottom",     # ensures slider hugs the bottom
            )
        )

        # Add some extra bottom space visually
        fig.update_layout(
            margin=dict(l=20, r=20, t=150, b=100)  # increase bottom margin
        )

        # --- Config for Streamlit ---
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


    st.divider()

    # --- Session State to hold investments (in-memory only) ---
    if "investments" not in st.session_state:
        st.session_state.investments = pd.DataFrame(columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"])

    # --- Fetch live gold price ---
    @st.cache_data(ttl=3600)
    def get_live_gold_price():
        url = "https://www.goldapi.io/api/XAU/INR"
        headers = {
            "x-access-token": "goldapi-c6kenk19mh4dmhji-io",
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
        st.success(f"💰 24K Gold Today: ₹{gold_price['24K']}/g | 22K Gold Today: ₹{gold_price['22K']}/g")
        st.caption(f"Updated on: {gold_price['timestamp']}")
        current_price = gold_price["24K"]
    else:
        st.warning("Could not fetch live gold price right now.")
        current_price = 0


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


with tab5:

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
            title=dict(
                text=title or 'Monte Carlo Price Simulation',
                x=0.5, xanchor='center',
                font=dict(size=16, color='white')
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.0,
                xanchor="center", x=0.5,
                font=dict(size=10, color='white')
            ),
            margin=dict(l=20, r=20, t=60, b=60),
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            showgrid=False,
            tickformat="%b\n%Y",
            rangeslider=dict(
                bgcolor="rgba(255,255,255,0.05)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                thickness=0.1,        # default is ~0.15 → thinner slider
                #yanchor="bottom",     # ensures slider hugs the bottom
            )
        )

        # Add some extra bottom space visually
        fig.update_layout(
            margin=dict(l=20, r=20, t=150, b=100)  # increase bottom margin
        )

        # --- Config for Streamlit ---
        config = {
            'scrollZoom': True,
            'responsive': True,
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'displaylogo': False
        }

        st.plotly_chart(fig, use_container_width=True, config=config)
        return summary, final_vals



    st.markdown("<h1 style='text-align: center; color: white;'>Silver Projection</h1>", unsafe_allow_html=True)

    df = pd.read_csv('silver_prices_2yrs.csv')

    df['Open'] = df['Open']/31.1035
    
    df=df.drop(['High','Low','Last','Change','Volume','%Change'],axis=1)
    #st.write(df.head())

    date_col = 'Time'
    price_col = 'Open'

    #df['Price'] = df['Price'].astype(float)
    #st.write(df[price_col])

    df,ppy = prepare_returns(df, date_col=date_col, price_col=price_col)
    mu, sigma = annualized_stats(df['log_ret'], ppy)


    yr = st.slider("Projection Horizon (Yrs)",1,20,5)
    sims = 10000
    S0 = df.iloc[-1][price_col]
    
    price_path = simulate_gbm_paths(S0, mu, sigma, yr, ppy, n_sims=sims, random_seed=42)
    #st.write(price_path)

    last_date = pd.to_datetime(df.iloc[-1][date_col])
    median_delta_days = df[date_col].sort_values().diff().dropna().dt.days.median()
    steps = int(years*ppy)
    dates_future = [last_date+timedelta(days = int(median_delta_days*i)) for i in range(steps+1)]
    summary, vals = summarize_and_plots(price_path, dates_future, title=f'{yr}-yr Monte Carlo ({sims} sims)')

    for k,v in summary.items():
        st.markdown(f"<h2 style='text-align: center; color: dodgerblue;'>Median price after {yr} years: {v:,.2f} ₹</h2>", unsafe_allow_html=True)
        break


    st.divider()

    # --- Session State to hold investments (in-memory only) ---
    if "investments" not in st.session_state:
        st.session_state.investments = pd.DataFrame(columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"])


    # --- Live Silver Price ---
    @st.cache_data(ttl=3600)
    def get_live_silver_price():
        url = "https://www.goldapi.io/api/XAG/INR"
        headers = {
            "x-access-token": "goldapi-c6kenk19mh4dmhji-io",
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
            st.error(f"⚠️ Error fetching silver price: {e}")
            return None

    silver_price = get_live_silver_price()
    #st.write(silver_price['24K'])
    if silver_price:
        st.success(f"💍 Silver Price Today: ₹{silver_price['24K']}/g")
        st.caption(f"Updated on: {silver_price['timestamp']}")
        current_price = silver_price["24K"]
    else:
        st.warning("Could not fetch live silver price right now.")
        current_price = 0

    # --- Initialize silver investment table ---
    if "silver_investments" not in st.session_state:
        st.session_state.silver_investments = pd.DataFrame(
            columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"]
        )

    # --- Add New Investment ---
    st.subheader("Add New Silver Investment")
    with st.form("add_form_silver", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date = st.date_input("Purchase Date", datetime.today(), key="silver_date")
        with col2:
            silver_type = st.selectbox("Type", ["Physical", "ETF", "Digital"], key="silver_type")
        with col3:
            qty = st.number_input("Quantity (grams)", min_value=0.1, step=0.1, key="silver_qty")
        with col4:
            buy_price = st.number_input("Buy Price (₹/g)", min_value=0.0, step=0.1, key="silver_buy")
        add = st.form_submit_button("Add Investment", type="primary")

    if add and qty > 0:
        new_entry = pd.DataFrame(
            [[date, silver_type, qty, buy_price]],
            columns=st.session_state.silver_investments.columns
        )
        st.session_state.silver_investments = pd.concat(
            [st.session_state.silver_investments, new_entry],
            ignore_index=True
        )
        st.success("✅ Silver investment added successfully!")

    # --- Delete Investment ---
    st.divider()
    st.subheader("Delete Silver Investment")

    df_silver = st.session_state.silver_investments.copy()

    if not df_silver.empty:
        df_silver["Label"] = (
            df_silver["Purchase Date"].astype(str)
            + " | " + df_silver["Type"]
            + " | " + df_silver["Quantity (g)"].astype(str) + "g"
        )

        to_delete = st.multiselect(
            "Select investment(s) to delete:",
            df_silver["Label"].tolist(),
            key="multiselect_silver"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Delete Selected", key="delete_silver"):
                if to_delete:
                    new_df = df_silver[~df_silver["Label"].isin(to_delete)].drop(columns=["Label"])
                    st.session_state.silver_investments = new_df
                    st.success(f"✅ Deleted {len(to_delete)} silver investment(s) successfully!")
                    st.rerun()
                else:
                    st.warning("⚠ Please select at least one investment to delete.")
        with col2:
            if st.button("🧹 Clear All Silver Investments", key="clear_silver"):
                st.session_state.silver_investments = pd.DataFrame(
                    columns=["Purchase Date", "Type", "Quantity (g)", "Buy Price (₹/g)"]
                )
                st.success("🗑 All silver investments cleared.")
                st.rerun()
    else:
        st.info("No silver investments yet. Add one above to get started!")

    # --- Portfolio Summary ---
    df_silver = st.session_state.silver_investments.copy()
    if not df_silver.empty:
        df_silver["Invested Value"] = df_silver["Quantity (g)"] * df_silver["Buy Price (₹/g)"]
        df_silver["Current Value"] = df_silver["Quantity (g)"] * current_price
        df_silver["Profit/Loss"] = df_silver["Current Value"] - df_silver["Invested Value"]
        df_silver["% Gain/Loss"] = (df_silver["Profit/Loss"] / df_silver["Invested Value"]) * 100
        df_silver["Years Held"] = (datetime.today() - pd.to_datetime(df_silver["Purchase Date"], errors="coerce")).dt.days / 365
        df_silver["CAGR (%)"] = ((df_silver["Current Value"] / df_silver["Invested Value"]) ** (1 / df_silver["Years Held"]) - 1) * 100
        df_silver["CAGR (%)"] = df_silver["CAGR (%)"].replace([math.inf, -math.inf], 0).fillna(0)

        total_invested = df_silver["Invested Value"].sum()
        total_current = df_silver["Current Value"].sum()
        total_gain = total_current - total_invested
        gain_percent = (total_gain / total_invested) * 100 if total_invested > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Invested", f"₹{total_invested:,.0f}")
        col2.metric("Current Value", f"₹{total_current:,.0f}")
        col3.metric("Profit/Loss", f"₹{total_gain:,.0f}", f"{gain_percent:.2f}%")

        st.subheader("📄 Silver Investment Details")
        st.dataframe(df_silver.style.format({
            "Buy Price (₹/g)": "{:.2f}",
            "Current Value": "{:.2f}",
            "Profit/Loss": "{:.2f}"
        }))

        st.subheader("📈 Portfolio by Type")
        pie_fig = px.pie(df_silver, names="Type", values="Current Value", title="Distribution by Type")
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No silver investments yet. Add one above to get started!")

