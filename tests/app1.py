import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import logging
import os
from SMC import *  
from tvDatafeed import TvDatafeed, Interval
from binance.client import Client  # Add Binance Client
from dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Binance API credentials (Ensure these are securely stored)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Initialize Binance Client if API credentials are provided
if BINANCE_API_KEY and BINANCE_API_SECRET:
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
else:
    binance_client = None
    logging.warning("Binance API credentials not provided. Binance data source will be unavailable.")

# LAYOUT SECTION------------------------------------------------------------------------------------
# Use Bootstrap theme for better visual appeal
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout using Bootstrap components
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(html.H1("SMC Plotting Tool", className="text-center text-primary mb-4",
                    style={
                        "margin-left": "10px",
                        "margin-top": "20px",
                        "margin-bottom": "20px",
                    },)),
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Enter Symbol:", className="text-white"),
                dcc.Input(id='symbol', value='DJI', type='text', className="form-control"),
            ], width=4),
            dbc.Col([
                html.Label("Enter Exchange:", className="text-white"),
                dcc.Input(id='exchange', value='TVC', type='text', className="form-control"),
            ], width=4),
            dbc.Col([
                html.Label("Select Data Source:", className="text-white"),
                dcc.Dropdown(
                    id='data-source',
                    options=[
                        {'label': 'TradingView', 'value': 'tvDatafeed'},
                        {'label': 'Binance', 'value': 'binance'}
                    ],
                    value='tvDatafeed',  # Default value
                    className="form-control"
                )
            ], width=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(html.Button('Submit', id='submit-val', n_clicks=0, className="btn btn-primary mt-3"), width=12, className="text-center")
        ], className="mb-4"),

        # First Row of Plots
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='fig_5t', figure={})]), style={"height": "100%"}), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_15t", figure={})]), style={"height": "100%"}), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_30t", figure={})]), style={"height": "100%"}), width=4),
        ], className="mb-4"),

        # Second Row of Plots
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id='fig_1h', figure={})]), style={"height": "100%"}), width=6),
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_4h", figure={})]), style={"height": "100%"}), width=6),
            # dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_1d", figure={})]), style={"height": "100%"}), width=4),
        ], className="mb-4"),

            # Third Row of Plots
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([dcc.Graph(id="fig_1d_full", figure={})]), style={"height": "100%"}), width=12),
        ], className="mb-4"),
    ],
    fluid=True,
    style={"padding": "20px"}
)


# Function to create a plot with layout improvements
def create_smc_plot(df, timeframe, freq_mapping, swing_length_mapping, dvalue=30, width=400, height=300):
    window_df = df.copy()

    if window_df.empty:
        logging.warning(f"No data available for timeframe {timeframe}.")
        return go.Figure()

    # Compute EMAs before slicing to ensure accuracy
    window_df['EMA20'] = window_df['close'].ewm(span=20, adjust=False).mean()
    window_df['EMA50'] = window_df['close'].ewm(span=50, adjust=False).mean()
    window_df['EMA100'] = window_df['close'].ewm(span=100, adjust=False).mean()

    # Slice to last 100 data points
    # window_df = window_df.tail(100)

    # Recompute SMC indicators on the reduced data
    swing_length = swing_length_mapping[timeframe]
    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=swing_length)
    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
    ob_data = smc.ob(window_df, swing_highs_lows_data)
    sessions = smc.sessions(window_df, session="New York kill zone")

    # Recompute range breaks based on the reduced data
    freq = freq_mapping[timeframe]
    rangebreak_list = find_rangebreaks(window_df, freq)

    fig = go.Figure(data=[go.Candlestick(
        x=window_df.index,
        open=window_df["open"],
        high=window_df["high"],
        low=window_df["low"],
        close=window_df["close"],
        increasing_line_color="#9b9b9b",
        decreasing_line_color="#771b1b",
        name='Candlestick'
    )])

    # Add EMA traces
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA20'],
        mode='lines',
        name='EMA20',
        line=dict(color='darkolivegreen', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA50'],
        mode='lines',
        name='EMA50',
        line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA100'],
        mode='lines',
        name='EMA100',
        line=dict(color='indianred', width=1)
    ))

    # Add SMC-related overlays to the chart
    fig = add_swing_highs_lows(fig, window_df, swing_highs_lows_data)
    fig = add_bos_choch(fig, window_df, bos_choch_data)
    fig = add_OB(fig, window_df, ob_data)
    fig = add_sessions(fig, window_df, sessions)

    # Modify layout directly within the figure definition
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,  # Hide legend to save space
        plot_bgcolor="rgba(34, 42, 58, 0.8)",
        paper_bgcolor="rgba(34, 42, 58, 0.8)",
        font=dict(color="white"),
        # width=width,
        # height=height,
        margin=dict(l=0, r=50, b=0, t=30, pad=0),
        title_text=timeframe,
        title_x=0.5,
        xaxis=dict(
            visible=True,
            showticklabels=False,
            rangebreaks=[dict(values=rangebreak_list, dvalue=(dvalue * 60 * 1000))],
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            visible=False,
            showticklabels=False
        )
    )

    return fig

# Function to create a plot with layout improvements
def create_smc_plot_1d(df, timeframe, freq_mapping, dvalue=30, width=1000, height=450):
    window_df = df.copy()

    if window_df.empty:
        logging.warning(f"No data available for timeframe {timeframe}.")
        return go.Figure()

    # Compute EMAs before slicing to ensure accuracy
    window_df['EMA20'] = window_df['close'].ewm(span=20, adjust=False).mean()
    window_df['EMA50'] = window_df['close'].ewm(span=50, adjust=False).mean()
    window_df['EMA100'] = window_df['close'].ewm(span=100, adjust=False).mean()
    window_df['EMA200'] = window_df['close'].ewm(span=200, adjust=False).mean()

    # Slice to last 100 data points
    window_df = window_df.tail(250)

    # Recompute SMC indicators on the reduced data
    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=10)
    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
    ob_data = smc.ob(window_df, swing_highs_lows_data)
    sessions = smc.sessions(window_df, session="New York kill zone")

    # Recompute range breaks based on the reduced data
    # freq = freq_mapping[timeframe]
    # rangebreak_list = find_rangebreaks(window_df, freq)

    fig = go.Figure(data=[go.Candlestick(
        x=window_df.index,
        open=window_df["open"],
        high=window_df["high"],
        low=window_df["low"],
        close=window_df["close"],
        increasing_line_color="#9b9b9b",
        decreasing_line_color="#771b1b",
        name='Candlestick'
    )])

    # Add EMA traces
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA20'],
        mode='lines',
        name='EMA20',
        line=dict(color='darkolivegreen', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA50'],
        mode='lines',
        name='EMA50',
        line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA100'],
        mode='lines',
        name='EMA100',
        line=dict(color='indianred', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=window_df.index,
        y=window_df['EMA200'],
        mode='lines',
        name='EMA200',
        line=dict(color='gray', width=1)
    ))

    # Add SMC-related overlays to the chart
    fig = add_swing_highs_lows(fig, window_df, swing_highs_lows_data)
    fig = add_bos_choch(fig, window_df, bos_choch_data)
    fig = add_OB(fig, window_df, ob_data)
    fig = add_sessions(fig, window_df, sessions)

    # Modify layout directly within the figure definition
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=False,  # Hide legend to save space
        plot_bgcolor="rgba(34, 42, 58, 0.8)",
        paper_bgcolor="rgba(34, 42, 58, 0.8)",
        font=dict(color="white"),
        # width=width,
        # height=height,
        margin=dict(l=0, r=50, b=0, t=30, pad=0),
        title_text=timeframe,
        title_x=0.5,
        xaxis=dict(
            visible=True,
            showticklabels=False,
            # rangebreaks=[dict(values=rangebreak_list, dvalue=(dvalue * 60 * 1000))],
            showgrid=False,
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            visible=False,
            showticklabels=False
        )
    )

    return fig

# Function to fetch historical data from TradingView
def fetch_data_tv(symbol, exchange, n_bars):
    tv = TvDatafeed()
    df_1d = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=1000)
    df_4h = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_4_hour, n_bars=500)
    df_1h = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_hour, n_bars=500)
    df_30t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_30_minute, n_bars=500)
    df_15t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_15_minute, n_bars=200)
    df_5t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_5_minute, n_bars=200)
    return df_1d, df_4h, df_1h, df_30t, df_15t, df_5t

# Function to fetch historical data using Binance
def fetch_data_binance(client, symbol, interval_limits):
    """
    Retrieve historical data for a symbol from Binance API.

    Parameters:
    - client (binance.client.Client): Initialized Binance client with API credentials.
    - symbol (str): Symbol (e.g., "BTCUSDT").
    - interval_limits (dict): Dictionary of intervals (e.g., {"1d": 1000, "4h": 500, "1h": 500, "30m": 300, "15m": 300, "5m": 300})
                              where keys are intervals and values are the corresponding limits.

    Returns:
    - data (dict): Dictionary containing DataFrames for each interval.
    """
    data = {}
    for interval, limit in interval_limits.items():
        logging.info(f"Fetching {interval} data for {symbol} from Binance with limit {limit}.")
        try:
            # Fetch historical klines data from Binance
            klines = client.get_historical_klines(symbol, interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            
            # Convert numerical columns to appropriate data types
            for column in ['open', 'high', 'low', 'close', 'volume']:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                
            data[interval] = df  # Store the DataFrame in the dictionary for the interval
        except Exception as e:
            logging.error(f"Error fetching {symbol} data at interval {interval} from Binance: {e}")
            data[interval] = pd.DataFrame()  # Return empty DataFrame on error
            
    return data


# Unified function to fetch data based on selected data source
def fetch_data(symbol, exchange, data_source, n_bar=1000):
    if data_source == 'tvDatafeed':
        return fetch_data_tv(symbol, exchange, n_bar)
    elif data_source == 'binance':
        if binance_client is None:
            logging.error("Binance client is not initialized. Check your API credentials.")
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        # Binance requires symbols like "BTCUSDT", ensure the symbol format is correct
        # Adjust exchange if necessary or map it to Binance's exchange format
        # For simplicity, we'll assume the symbol is correctly formatted for Binance
        intervals = ['1d', '4h', '1h', '30m', '15m', '5m']
        interval_limits = {
                            '1d': 1000,
                            '4h': 500,
                            '1h': 500,
                            '30m': 350,
                            '15m': 200,
                            '5m': 200
                        }

        data_binance = fetch_data_binance(binance_client, symbol, interval_limits)
        # Map Binance intervals to the expected order
        df_1d = data_binance.get('1d', pd.DataFrame())
        df_4h = data_binance.get('4h', pd.DataFrame())
        df_1h = data_binance.get('1h', pd.DataFrame())
        df_30t = data_binance.get('30m', pd.DataFrame())
        df_15t = data_binance.get('15m', pd.DataFrame())
        df_5t = data_binance.get('5m', pd.DataFrame())
        return df_1d, df_4h, df_1h, df_30t, df_15t, df_5t
    else:
        logging.error(f"Unsupported data source: {data_source}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

# Function to find range breaks
def find_rangebreaks(df, freq):
    ref_date = pd.date_range(start=pd.Timestamp(df.index[0]), end=pd.Timestamp(df.index[-1]), freq=freq)
    df_index_as_date = pd.to_datetime(df.index)
    rangebreak_list = ref_date.difference(df_index_as_date)
    return rangebreak_list


# CALLBACK SECTION------------------------------------------------------------------------------------
# Callback to update the graphs
@app.callback(
    [
        Output('fig_5t', 'figure'),
        Output('fig_15t', 'figure'),
        Output('fig_30t', 'figure'),
        Output('fig_1h', 'figure'),
        Output('fig_4h', 'figure'),
        # Output('fig_1d', 'figure'),
        Output('fig_1d_full', 'figure'),
    ],
    [Input('submit-val', 'n_clicks')],
    [
        State('symbol', 'value'),       # Current values are accessed when the button is clicked
        State('exchange', 'value'),
        State('data-source', 'value')
    ]
)
def update_graph(n_clicks, symbol, exchange, data_source):

    if n_clicks is None:
        # Optionally prevent the callback from updating before the first click
        raise PreventUpdate
    
    if n_clicks > 0:
        # Fetch the data based on symbol, exchange, and data source
        df_1d, df_4h, df_1h, df_30t, df_15t, df_5t = fetch_data(symbol, exchange, data_source)

        # Adjust frequency strings for pandas
        freq_mapping = {
            '1d': '1D',
            '4h': '4h',
            '1h': '1h',
            '30min': '30min',
            '15min': '15min',
            '5min': '5min'
        }

        # Define dvalue (in minutes) based on timeframe
        dvalue_mapping = {
            '1d': 1440,
            '4h': 240,
            '1h': 60,
            '30min': 30,
            '15min': 15,
            '5min': 5
        }
        
        # Define OB swing_length based on timeframe
        swing_length_mapping = {
            '1d': 10,
            '4h': 20,
            '1h': 20,
            '30min': 20,
            '15min': 10,
            '5min': 10
        }

        # Create the plots with correct dvalue (in minutes)
        # fig_1d = create_smc_plot(df_1d, timeframe='1d', freq_mapping=freq_mapping, dvalue=dvalue_mapping['1d'])
        fig_4h = create_smc_plot(df_4h, timeframe='4h', freq_mapping=freq_mapping, swing_length_mapping=swing_length_mapping, dvalue=dvalue_mapping['4h'])
        fig_1h = create_smc_plot(df_1h, timeframe='1h', freq_mapping=freq_mapping, swing_length_mapping=swing_length_mapping, dvalue=dvalue_mapping['1h'])
        fig_30t = create_smc_plot(df_30t, timeframe='30min', freq_mapping=freq_mapping, swing_length_mapping=swing_length_mapping, dvalue=dvalue_mapping['30min'])
        fig_15t = create_smc_plot(df_15t, timeframe='15min', freq_mapping=freq_mapping, swing_length_mapping=swing_length_mapping, dvalue=dvalue_mapping['15min'])
        fig_5t = create_smc_plot(df_5t, timeframe='5min', freq_mapping=freq_mapping, swing_length_mapping=swing_length_mapping, dvalue=dvalue_mapping['5min'])

        fig_1d_full = create_smc_plot_1d(df_1d, timeframe='1d', freq_mapping=freq_mapping, dvalue=dvalue_mapping['1d'])


        # Return all figures in the order of Outputs
        # return fig_5t, fig_15t, fig_30t, fig_1h, fig_4h, fig_1d, fig_1d_full
        return fig_5t, fig_15t, fig_30t, fig_1h, fig_4h, fig_1d_full

    # If n_clicks is 0, return empty figures
    return {}, {}, {}, {}, {}, {} # {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
