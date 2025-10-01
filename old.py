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
                return df
        return None
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return None

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
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    market_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (symbol, info) in enumerate(SECTOR_ETFS.items()):
        status_text.text(f"Loading {symbol} - {info['name']}...")
        
        df = fetch_tiingo_data(symbol, start_date, end_date)
        if df is not None:
            metrics = calculate_performance(df)
            if metrics:
                market_data[symbol] = {
                    **info,
                    **metrics,
                    'data': df
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
    
    # Sidebar - Time period selector
    st.sidebar.header("⚙️ Settings")
    time_period = st.sidebar.radio(
        "Performance Period",
        ["3 Month", "6 Month", "12 Month"],
        index=2
    )
    
    period_map = {
        "3 Month": "perf_3m",
        "6 Month": "perf_6m",
        "12 Month": "perf_12m"
    }
    selected_period = period_map[time_period]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 About")
    st.sidebar.info(
        "This dashboard tracks 11 major sector ETFs with real-time data from Tiingo API. "
        "Data is cached for 5 minutes to optimize performance."
    )
    
    # Display timestamp
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main metrics grid
    st.header("🎯 Sector ETF Overview")
    
    # Create metrics in rows of 4
    symbols = list(market_data.keys())
    for i in range(0, len(symbols), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(symbols):
                symbol = symbols[i + j]
                data = market_data[symbol]
                
                with col:
                    delta_color = "normal" if data['day_change_pct'] >= 0 else "inverse"
                    st.metric(
                        label=f"{symbol} - {data['name']}",
                        value=f"${data['current_price']:.2f}",
                        delta=f"{data['day_change_pct']:.2f}%",
                        delta_color=delta_color
                    )
    
    st.markdown("---")
    
    # Performance comparison chart
    st.header(f"📊 {time_period} Performance Comparison")
    
    # Prepare data for chart
    perf_data = []
    for symbol, data in market_data.items():
        perf_data.append({
            'Symbol': symbol,
            'Sector': data['name'],
            'Performance': data[selected_period],
            'Color': data['color']
        })
    
    perf_df = pd.DataFrame(perf_data).sort_values('Performance', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    for _, row in perf_df.iterrows():
        fig.add_trace(go.Bar(
            y=[row['Symbol']],
            x=[row['Performance']],
            orientation='h',
            name=row['Symbol'],
            marker=dict(color=row['Color']),
            text=[f"{row['Performance']:.2f}%"],
            textposition='outside',
            showlegend=False,
            hovertemplate=f"<b>{row['Symbol']} - {row['Sector']}</b><br>" +
                         f"Performance: {row['Performance']:.2f}%<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"{time_period} Returns by Sector",
        xaxis_title="Return (%)",
        yaxis_title="",
        height=500,
        template="plotly_white",
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Performance table
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📈 Top Performers")
        top_performers = perf_df.tail(5)[['Symbol', 'Sector', 'Performance']].sort_values('Performance', ascending=False)
        top_performers['Performance'] = top_performers['Performance'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(top_performers, hide_index=True, use_container_width=True)
    
    with col2:
        st.header("📉 Bottom Performers")
        bottom_performers = perf_df.head(5)[['Symbol', 'Sector', 'Performance']]
        bottom_performers['Performance'] = bottom_performers['Performance'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(bottom_performers, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed performance table
    st.header("📋 Detailed Performance Metrics")
    
    detailed_data = []
    for symbol, data in market_data.items():
        detailed_data.append({
            'Symbol': symbol,
            'Sector': data['name'],
            'Price': f"${data['current_price']:.2f}",
            'Day Change': f"{data['day_change_pct']:.2f}%",
            '3M Return': f"{data['perf_3m']:.2f}%",
            '6M Return': f"{data['perf_6m']:.2f}%",
            '12M Return': f"{data['perf_12m']:.2f}%"
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    st.dataframe(
        detailed_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Sector": st.column_config.TextColumn("Sector", width="medium"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Day Change": st.column_config.TextColumn("Day Change", width="small"),
            "3M Return": st.column_config.TextColumn("3M Return", width="small"),
            "6M Return": st.column_config.TextColumn("6M Return", width="small"),
            "12M Return": st.column_config.TextColumn("12M Return", width="small"),
        }
    )
    
    st.markdown("---")
    
    # Performance trend chart
    st.header("📈 Multi-Period Performance Trends")
    
    trend_data = []
    for symbol, data in market_data.items():
        trend_data.append({
            'Symbol': symbol,
            '3 Month': data['perf_3m'],
            '6 Month': data['perf_6m'],
            '12 Month': data['perf_12m']
        })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig_trend = go.Figure()
    
    for _, row in trend_df.iterrows():
        symbol = row['Symbol']
        color = market_data[symbol]['color']
        fig_trend.add_trace(go.Scatter(
            x=['3 Month', '6 Month', '12 Month'],
            y=[row['3 Month'], row['6 Month'], row['12 Month']],
            mode='lines+markers',
            name=f"{symbol} - {market_data[symbol]['name']}",
            line=dict(color=color, width=3),
            marker=dict(size=8)
        ))
    
    fig_trend.update_layout(
        title="Performance Trends Across Time Periods",
        xaxis_title="Time Period",
        yaxis_title="Return (%)",
        height=600,
        template="plotly_white",
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        )
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

if __name__ == "__main__":
    main()
