import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from SMC import *  
from tvDatafeed import TvDatafeed, Interval

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
            ], width=6),
            dbc.Col([
                html.Label("Enter Exchange:", className="text-white"),
                dcc.Input(id='exchange', value='TVC', type='text', className="form-control"),
            ], width=6),
            dbc.Col([
                html.Button('Submit', id='submit-val', n_clicks=0, className="btn btn-primary mt-3")
            ], width=12, className="text-center")
        ], className="mb-4"),

        # First Row of Plots
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id='fig_5t', figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id="fig_15t", figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id="fig_30t", figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4,
            ),
        ]),

        # Second Row of Plots
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id='fig_1h', figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id="fig_4h", figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dcc.Graph(id="fig_1d", figure={}),
                        ]
                    ),
                    style={"height": "100%"},
                ),
                width=4,
            ),
        ]),
    ],
    style={
        "padding": "0px",
        # "margin": "0px"
    }
)


# Function to fetch historical data
def fetch_data(symbol, exchange):
    # TVDatafeed object
    tv = TvDatafeed()
    df_1d = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=400)
    df_4h = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_4_hour, n_bars=400)
    df_1h = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_hour, n_bars=400)
    df_30t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_30_minute, n_bars=400)
    df_15t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_15_minute, n_bars=400)
    df_5t = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_5_minute, n_bars=400)
    return df_1d, df_4h, df_1h, df_30t, df_15t, df_5t

# Function to find range breaks
def find_rangebreaks(df, freq):
    ref_date = pd.date_range(start=pd.Timestamp(df.index[0]), end=pd.Timestamp(df.index[-1]), freq=freq)
    df_index_as_date = pd.to_datetime(df.index)
    rangebreak_list = ref_date.difference(df_index_as_date)
    return rangebreak_list

# Function to create a plot with layout improvements
def create_smc_plot(df, timeframe, freq_mapping, session="New York kill zone", swing_length=5, dvalue=30, width=400, height=300):
    window_df = df.copy()

    # Compute EMAs before slicing to ensure accuracy
    window_df['EMA20'] = window_df['close'].ewm(span=20, adjust=False).mean()
    window_df['EMA50'] = window_df['close'].ewm(span=50, adjust=False).mean()
    window_df['EMA100'] = window_df['close'].ewm(span=100, adjust=False).mean()

    # Slice to last 200 data points
    window_df = window_df.tail(100)

    # Recompute SMC indicators on the reduced data
    swing_highs_lows_data = smc.swing_highs_lows(window_df, swing_length=swing_length)
    bos_choch_data = smc.bos_choch(window_df, swing_highs_lows_data)
    ob_data = smc.ob(window_df, swing_highs_lows_data)
    sessions = smc.sessions(window_df, session=session)

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
        line=dict(color='blue', width=1)
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
        line=dict(color='purple', width=1)
    ))

    # Add SMC-related overlays to the chart
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
        width=width,
        height=height,
        margin=dict(l=0, r=50, b=0, t=0, pad=0),
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

# Callback to update the graphs
@app.callback(
    [
        Output('fig_5t', 'figure'),
        Output('fig_15t', 'figure'),
        Output('fig_30t', 'figure'),
        Output('fig_1h', 'figure'),
        Output('fig_4h', 'figure'),
        Output('fig_1d', 'figure'),
    ],
    Input('submit-val', 'n_clicks'),
    [Input('symbol', 'value'), Input('exchange', 'value')]
)
def update_graph(n_clicks, symbol, exchange):
    if n_clicks > 0:
        # Fetch the data based on symbol and exchange
        df_1d, df_4h, df_1h, df_30t, df_15t, df_5t = fetch_data(symbol, exchange)

        # Adjust frequency strings for pandas
        freq_mapping = {
            '1d': '1d',
            '4h': '4h',
            '1h': '1h',
            '30min': '30min',
            '15min': '15min',
            '5min': '5min'
        }

        # Create the plots with correct dvalue (in minutes)
        fig_1d = create_smc_plot(df_1d, timeframe='1d', freq_mapping=freq_mapping, dvalue=1440)
        fig_4h = create_smc_plot(df_4h, timeframe='4h', freq_mapping=freq_mapping, dvalue=240)
        fig_1h = create_smc_plot(df_1h, timeframe='1h', freq_mapping=freq_mapping, dvalue=60)
        fig_30t = create_smc_plot(df_30t, timeframe='30min', freq_mapping=freq_mapping, dvalue=30)
        fig_15t = create_smc_plot(df_15t, timeframe='15min', freq_mapping=freq_mapping, dvalue=15)
        fig_5t = create_smc_plot(df_5t, timeframe='5min', freq_mapping=freq_mapping, dvalue=5)


        # Return all figures in the order of Outputs
        return fig_5t, fig_15t, fig_30t, fig_1h, fig_4h, fig_1d

    # If n_clicks is 0, return empty figures
    return {}, {}, {}, {}, {}, {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
