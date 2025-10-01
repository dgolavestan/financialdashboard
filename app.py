import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Financial Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white background and black text
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: white;
        padding: 0rem 1rem;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* All text to black */
    .main, .main p, .main span, .main div {
        color: black !important;
    }
    
    /* Headers to black */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stMetric:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.3s;
    }
    
    /* Ensure markdown text is black */
    .stMarkdown {
        color: black !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #f8f9fa;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Tiingo API configuration - using Streamlit Secrets for security
try:
    TIINGO_API_KEY = st.secrets["TIINGO_API_KEY"]
except Exception as e:
    st.error("⚠️ API key not found in Streamlit Secrets. Please add your Tiingo API key in the app settings.")
    st.stop()

# Sector ETF definitions
SECTOR_ETFS = {
    'XLK': {'name': 'Technology', 'color': '#667eea'},
    'XLV': {'name': 'Healthcare', 'color': '#48bb78'},
    'XLF': {'name': 'Financials', 'color': '#ed8936'},
    'XLY': {'name': 'Consumer Discretionary', 'color': '#9f7aea'},
    'XLE': {'name': 'Energy', 'color': '#38b2ac'},
    'XLB': {'name': 'Materials', 'color': '#f56565'},
    'XLI': {'name': 'Industrials', 'color': '#ecc94b'},
    'XLP': {'name': 'Consumer Staples', 'color': '#4299e1'},
    'XLRE': {'name': 'Real Estate', 'color': '#fc8181'},
    'XLU': {'name': 'Utilities', 'color': '#68d391'},
    'XLC': {'name': 'Communication Services', 'color': '#b794f4'}
}

# Authentication function
def check_password():
    """Returns `True` if the user has entered correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (st.session_state["username"] == "admin" and 
            st.session_state["password"] == "demo123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show login form
        st.markdown("# 📊 Financial Market Dashboard")
        st.markdown("### Please login to access the dashboard")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, use_container_width=True)
            
            with st.expander("ℹ️ Demo Credentials"):
                st.info("**Username:** admin\n\n**Password:** demo123")
        
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show error
        st.markdown("# 📊 Financial Market Dashboard")
        st.markdown("### Please login to access the dashboard")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered, use_container_width=True)
            st.error("😕 Username or password incorrect")
            
            with st.expander("ℹ️ Demo Credentials"):
                st.info("**Username:** admin\n\n**Password:** demo123")
        
        return False
    
    else:
        # Password correct
        return True

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_tiingo_data(symbol, start_date, end_date):
    """Fetch historical price data from Tiingo API"""
    try:
        url = f'https://api.tiingo.com/tiingo/daily/{symbol}/prices'
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'token': TIINGO_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                # Use adjClose for dividend/split adjusted prices
                df['adj_close'] = df['adjClose']
                return df
        return None
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return None

def calculate_rolling_returns(df, windows=[10, 30, 60]):
    """Calculate rolling returns for multiple windows"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    df = df.sort_values('date')
    
    # Calculate rolling returns for each window
    for window in windows:
        # Calculate return as (current_price - price_n_days_ago) / price_n_days_ago * 100
        df[f'return_{window}d'] = ((df['adj_close'] / df['adj_close'].shift(window)) - 1) * 100
    
    return df

def calculate_volatility_targeting(df, target_vol=15, vol_windows=[30, 90]):
    """Calculate volatility targeting exposure for SPY"""
    if df is None or len(df) == 0:
        return None
    
    df = df.copy()
    df = df.sort_values('date')
    
    # Calculate daily returns
    df['daily_return'] = df['adj_close'].pct_change() * 100
    
    # Calculate rolling volatility (annualized) for each window
    for window in vol_windows:
        # Rolling standard deviation of daily returns
        rolling_std = df['daily_return'].rolling(window=window).std()
        # Annualize: multiply by sqrt(252 trading days)
        df[f'volatility_{window}d'] = rolling_std * (252 ** 0.5)
        
        # Calculate target exposure: target_vol / realized_vol
        # Floor at 0% (no negative exposure)
        df[f'exposure_{window}d'] = (target_vol / df[f'volatility_{window}d']).clip(lower=0) * 100
    
    return df

def calculate_performance(df):
    """Calculate performance metrics from price data"""
    if df is None or len(df) == 0:
        return None
    
    current_price = df.iloc[-1]['adj_close']
    previous_close = df.iloc[-2]['adj_close'] if len(df) > 1 else current_price
    
    # Get the latest rolling returns from the pre-calculated columns
    latest_10d = df['return_10d'].dropna().iloc[-1] if not df['return_10d'].dropna().empty else 0
    latest_30d = df['return_30d'].dropna().iloc[-1] if not df['return_30d'].dropna().empty else 0
    latest_60d = df['return_60d'].dropna().iloc[-1] if not df['return_60d'].dropna().empty else 0
    
    return {
        'current_price': current_price,
        'previous_close': previous_close,
        'day_change': current_price - previous_close,
        'day_change_pct': ((current_price - previous_close) / previous_close * 100),
        'return_10d': latest_10d,
        'return_30d': latest_30d,
        'return_60d': latest_60d
    }

def load_all_market_data():
    """Load data for all sector ETFs"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Fetch 2 years of data for rolling return calculations
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    market_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (symbol, info) in enumerate(SECTOR_ETFS.items()):
        status_text.text(f"Loading {symbol} - {info['name']}...")
        
        df = fetch_tiingo_data(symbol, start_date, end_date)
        if df is not None:
            # Calculate rolling returns FIRST
            df_with_returns = calculate_rolling_returns(df)
            # Then calculate performance metrics using those rolling returns
            metrics = calculate_performance(df_with_returns)
            if metrics:
                market_data[symbol] = {
                    **info,
                    **metrics,
                    'data': df_with_returns
                }
        
        progress_bar.progress((idx + 1) / len(SECTOR_ETFS))
        time.sleep(0.1)  # Small delay to avoid rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return market_data

def main():
    """Main dashboard application"""
    
    # Check authentication
    if not check_password():
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Financial Market Dashboard")
        st.markdown("*Real-time sector ETF tracking with Tiingo data*")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["password_correct"] = False
            st.rerun()
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📈 Sector ETF Analysis", "🎯 Vol Targeting Positioning"])
    
    with tab1:
        render_sector_analysis()
    
    with tab2:
        render_vol_targeting()

def render_sector_analysis():
    """Render the sector ETF analysis tab"""
    # Load market data
    with st.spinner("Loading market data from Tiingo..."):
        market_data = load_all_market_data()
    
    if not market_data:
        st.error("Failed to load market data. Please check your API key and try again.")
        return
    
    st.success(f"✅ Successfully loaded data for {len(market_data)} ETFs")
    
    # Calculate global min/max for y-axis scaling
    all_returns = []
    for symbol, data in market_data.items():
        if 'data' in data and data['data'] is not None:
            df = data['data']
            for col in ['return_10d', 'return_30d', 'return_60d']:
                if col in df.columns:
                    all_returns.extend(df[col].dropna().tolist())
    
    if all_returns:
        global_min = min(all_returns)
        global_max = max(all_returns)
        # Add 10% padding
        y_range = [global_min * 1.1 if global_min < 0 else global_min * 0.9,
                   global_max * 1.1 if global_max > 0 else global_max * 0.9]
    else:
        y_range = [-30, 30]
    
    # Display rolling return charts at the top
    st.header("📈 Rolling Rate of Return Analysis (Last 2 Years)")
    st.markdown("*Adjusted for dividends and splits - All charts use the same Y-axis scale for comparison*")
    
    # Create charts in rows of 3
    symbols = list(market_data.keys())
    for i in range(0, len(symbols), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(symbols):
                symbol = symbols[i + j]
                data = market_data[symbol]
                
                with col:
                    if 'data' in data and data['data'] is not None:
                        df = data['data']
                        
                        # Filter to last 2 years only
                        two_years_ago = pd.Timestamp(datetime.now() - timedelta(days=730))
                        # Make sure both are timezone-aware or both are timezone-naive
                        if df['date'].dt.tz is not None:
                            two_years_ago = two_years_ago.tz_localize('UTC')
                        
                        df_filtered = df[df['date'] >= two_years_ago].copy()
                        
                        fig = go.Figure()
                        
                        # Add 10-day rolling return
                        fig.add_trace(go.Scatter(
                            x=df_filtered['date'],
                            y=df_filtered['return_10d'],
                            mode='lines',
                            name='10-Day',
                            line=dict(color='#667eea', width=1.5),
                            hovertemplate='<b>10-Day</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Add 30-day rolling return
                        fig.add_trace(go.Scatter(
                            x=df_filtered['date'],
                            y=df_filtered['return_30d'],
                            mode='lines',
                            name='30-Day',
                            line=dict(color='#48bb78', width=2),
                            hovertemplate='<b>30-Day</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Add 60-day rolling return
                        fig.add_trace(go.Scatter(
                            x=df_filtered['date'],
                            y=df_filtered['return_60d'],
                            mode='lines',
                            name='60-Day',
                            line=dict(color='#ed8936', width=2.5),
                            hovertemplate='<b>60-Day</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Add zero line
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        
                        fig.update_layout(
                            title=f"{symbol} - {data['name']}",
                            xaxis_title="",
                            yaxis_title="Return (%)",
                            height=350,
                            template="plotly_white",
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            yaxis=dict(range=y_range),  # Same scale for all charts
                            margin=dict(l=50, r=20, t=60, b=40)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No data available for {symbol}")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("### 📈 About")
    st.sidebar.info(
        "This dashboard tracks 11 major sector ETFs with real-time data from Tiingo API. "
        "Rolling returns are calculated using dividend and split-adjusted prices."
    )
    
    # Display timestamp
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Rate of Change Summary Table
    st.header("📊 Current Rate of Change Summary")
    st.markdown("*Latest rolling returns - Color coded for quick analysis*")
    
    # Prepare table data
    table_data = []
    for symbol, data in market_data.items():
        if 'data' in data and data['data'] is not None:
            # Use the same values from the performance calculations
            table_data.append({
                'ETF': symbol,
                'Sector': data['name'],
                '10-Day ROC (%)': data['return_10d'],
                '30-Day ROC (%)': data['return_30d'],
                '60-Day ROC (%)': data['return_60d']
            })
    
    summary_df = pd.DataFrame(table_data)
    
    # Function to apply color gradient
    def color_gradient(val):
        """Apply color gradient based on value"""
        if pd.isna(val):
            return 'background-color: white'
        
        # Color scale: red (negative) to white (0) to green (positive)
        if val < 0:
            # Scale from red to white based on how negative
            intensity = min(abs(val) / 20, 1)  # Cap at -20% for full red
            red = 255
            green = int(255 * (1 - intensity))
            blue = int(255 * (1 - intensity))
        else:
            # Scale from white to green based on how positive
            intensity = min(val / 20, 1)  # Cap at +20% for full green
            red = int(255 * (1 - intensity))
            green = 255
            blue = int(255 * (1 - intensity))
        
        return f'background-color: rgb({red}, {green}, {blue})'
    
    # Apply styling
    styled_df = summary_df.style.format({
        '10-Day ROC (%)': '{:.2f}',
        '30-Day ROC (%)': '{:.2f}',
        '60-Day ROC (%)': '{:.2f}'
    }).applymap(
        color_gradient,
        subset=['10-Day ROC (%)', '30-Day ROC (%)', '60-Day ROC (%)']
    ).set_properties(**{
        'text-align': 'center',
        'font-weight': 'bold'
    }, subset=['10-Day ROC (%)', '30-Day ROC (%)', '60-Day ROC (%)']
    ).set_properties(**{
        'text-align': 'left',
        'font-weight': 'bold'
    }, subset=['ETF', 'Sector']
    )
    
    # Display the styled dataframe
    st.dataframe(
        styled_df,
        hide_index=True,
        use_container_width=True,
        height=450
    )
    
    # Add legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟥 **Strong Negative** (< -10%)")
    with col2:
        st.markdown("⬜ **Neutral** (≈ 0%)")
    with col3:
        st.markdown("🟩 **Strong Positive** (> +10%)")

def render_vol_targeting():
    """Render the volatility targeting analysis tab"""
    st.header("🎯 Vol Targeting Positioning - SPY")
    st.markdown("*Estimated exposure for a 15% target volatility strategy*")
    
    # Sidebar settings for this tab
    st.sidebar.markdown("---")
    st.sidebar.subheader("Vol Targeting Settings")
    target_vol = st.sidebar.slider("Target Volatility (%)", min_value=5, max_value=30, value=15, step=1)
    
    # Load SPY data
    with st.spinner("Loading SPY data from Tiingo..."):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        spy_df = fetch_tiingo_data('SPY', start_date, end_date)
        
        if spy_df is None:
            st.error("Failed to load SPY data. Please check your API connection.")
            return
        
        # Calculate volatility targeting
        spy_df = calculate_volatility_targeting(spy_df, target_vol=target_vol, vol_windows=[30, 90])
    
    st.success("✅ Successfully loaded SPY data")
    
    # Filter to last 2 years for display
    two_years_ago = pd.Timestamp(datetime.now() - timedelta(days=730))
    if spy_df['date'].dt.tz is not None:
        two_years_ago = two_years_ago.tz_localize('UTC')
    spy_filtered = spy_df[spy_df['date'] >= two_years_ago].copy()
    
    # Main chart: Exposure levels
    st.subheader("📊 Volatility-Targeted Exposure Over Time")
    
    fig = go.Figure()
    
    # Add 30-day exposure line
    fig.add_trace(go.Scatter(
        x=spy_filtered['date'],
        y=spy_filtered['exposure_30d'],
        mode='lines',
        name='30-Day Vol',
        line=dict(color='#667eea', width=2.5),
        hovertemplate='<b>30-Day Vol Target</b><br>Date: %{x}<br>Exposure: %{y:.1f}%<extra></extra>'
    ))
    
    # Add 90-day exposure line
    fig.add_trace(go.Scatter(
        x=spy_filtered['date'],
        y=spy_filtered['exposure_90d'],
        mode='lines',
        name='90-Day Vol',
        line=dict(color='#48bb78', width=2.5),
        hovertemplate='<b>90-Day Vol Target</b><br>Date: %{x}<br>Exposure: %{y:.1f}%<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=100, line_dash="dash", line_color="orange", opacity=0.5, 
                  annotation_text="100% (Unleveraged)", annotation_position="right")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="0% (Min Exposure)", annotation_position="right")
    fig.add_hline(y=200, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="200% (2x Leverage)", annotation_position="right")
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Target Exposure (%)",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Current Positioning")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_30d_exp = spy_filtered['exposure_30d'].iloc[-1]
    current_90d_exp = spy_filtered['exposure_90d'].iloc[-1]
    current_30d_vol = spy_filtered['volatility_30d'].iloc[-1]
    current_90d_vol = spy_filtered['volatility_90d'].iloc[-1]
    
    with col1:
        st.metric(
            label="30-Day Target Exposure",
            value=f"{current_30d_exp:.1f}%",
            delta=f"{current_30d_exp - spy_filtered['exposure_30d'].iloc[-2]:.1f}% vs yesterday"
        )
    
    with col2:
        st.metric(
            label="90-Day Target Exposure",
            value=f"{current_90d_exp:.1f}%",
            delta=f"{current_90d_exp - spy_filtered['exposure_90d'].iloc[-2]:.1f}% vs yesterday"
        )
    
    with col3:
        st.metric(
            label="30-Day Realized Vol",
            value=f"{current_30d_vol:.1f}%",
            delta=f"{current_30d_vol - target_vol:.1f}% vs target",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="90-Day Realized Vol",
            value=f"{current_90d_vol:.1f}%",
            delta=f"{current_90d_vol - target_vol:.1f}% vs target",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Volatility chart
    st.subheader("📉 Realized Volatility Over Time")
    
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Scatter(
        x=spy_filtered['date'],
        y=spy_filtered['volatility_30d'],
        mode='lines',
        name='30-Day Realized Vol',
        line=dict(color='#ed8936', width=2),
        hovertemplate='<b>30-Day Vol</b><br>Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
    ))
    
    fig_vol.add_trace(go.Scatter(
        x=spy_filtered['date'],
        y=spy_filtered['volatility_90d'],
        mode='lines',
        name='90-Day Realized Vol',
        line=dict(color='#9f7aea', width=2),
        hovertemplate='<b>90-Day Vol</b><br>Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
    ))
    
    # Add target line
    fig_vol.add_hline(y=target_vol, line_dash="dash", line_color="#667eea", opacity=0.7,
                      annotation_text=f"Target: {target_vol}%", annotation_position="right")
    
    fig_vol.update_layout(
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        height=400,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.markdown("---")
    
    # Explanation
    with st.expander("ℹ️ How Volatility Targeting Works"):
        st.markdown("""
        **Volatility targeting** is a strategy where portfolio exposure is adjusted to maintain a constant level of risk (volatility).
        
        **Formula:**
        - Target Exposure = (Target Volatility / Realized Volatility) × 100%
        - Capped at 100% (maximum) and 0% (minimum)
        
        **Example:**
        - If target is 15% and realized vol is 15%: Exposure = 100%
        - If realized vol rises to 30%: Exposure = 50% (reduce exposure)
        - If realized vol falls to 7.5%: Exposure = 100% (at maximum)
        
        **Key Insights:**
        - **Higher volatility** → Lower exposure (defensive positioning)
        - **Lower volatility** → Higher exposure (aggressive positioning)
        - This creates a systematic "buy low, sell high" pattern
        - 30-day window is more responsive to recent changes
        - 90-day window is smoother and less reactive
        """)
    
    # Summary table
    st.subheader("📋 Historical Exposure Statistics")
    
    stats_data = []
    for window in [30, 90]:
        exp_col = f'exposure_{window}d'
        vol_col = f'volatility_{window}d'
        
        stats_data.append({
            'Window': f'{window}-Day',
            'Avg Exposure': f"{spy_filtered[exp_col].mean():.1f}%",
            'Current Exposure': f"{spy_filtered[exp_col].iloc[-1]:.1f}%",
            'Max Exposure': f"{spy_filtered[exp_col].max():.1f}%",
            'Min Exposure': f"{spy_filtered[exp_col].min():.1f}%",
            'Avg Volatility': f"{spy_filtered[vol_col].mean():.1f}%",
            'Current Volatility': f"{spy_filtered[vol_col].iloc[-1]:.1f}%"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
