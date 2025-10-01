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

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stMetric:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.3s;
    }
    h1 {
        color: #667eea;
    }
    h2 {
        color: #2d3748;
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

def calculate_performance(df):
    """Calculate performance metrics from price data"""
    if df is None or len(df) == 0:
        return None
    
    current_price = df.iloc[-1]['close']
    previous_close = df.iloc[-2]['close'] if len(df) > 1 else current_price
    
    # Calculate rolling performance
    price_3m_ago = df.iloc[max(0, len(df) - 63)]['close'] if len(df) > 63 else df.iloc[0]['close']
    price_6m_ago = df.iloc[max(0, len(df) - 126)]['close'] if len(df) > 126 else df.iloc[0]['close']
    price_12m_ago = df.iloc[0]['close']
    
    return {
        'current_price': current_price,
        'previous_close': previous_close,
        'day_change': current_price - previous_close,
        'day_change_pct': ((current_price - previous_close) / previous_close * 100),
        'perf_3m': ((current_price - price_3m_ago) / price_3m_ago * 100),
        'perf_6m': ((current_price - price_6m_ago) / price_6m_ago * 100),
        'perf_12m': ((current_price - price_12m_ago) / price_12m_ago * 100)
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
            # Calculate rolling returns
            df_with_returns = calculate_rolling_returns(df)
            metrics = calculate_performance(df)
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
            df = data['data']
            # Get the most recent non-null values
            latest_10d = df['return_10d'].dropna().iloc[-1] if not df['return_10d'].dropna().empty else 0
            latest_30d = df['return_30d'].dropna().iloc[-1] if not df['return_30d'].dropna().empty else 0
            latest_60d = df['return_60d'].dropna().iloc[-1] if not df['return_60d'].dropna().empty else 0
            
            table_data.append({
                'ETF': symbol,
                'Sector': data['name'],
                '10-Day ROC (%)': latest_10d,
                '30-Day ROC (%)': latest_30d,
                '60-Day ROC (%)': latest_60d
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

if __name__ == "__main__":
    main()
