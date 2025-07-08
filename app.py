import dash as dash
from dash import dcc, html, State
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import shapiro, kstest, normaltest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# ============================ Initialization ==================================

run_type = 'LOCAL' # 'LOCAL' or 'SERVER'

my_app = dash.Dash('Jeremy\'s Chess Data Project')

if run_type == 'SERVER':
    server = my_app.server

csv_file = 'chess_data.csv'

df = pd.read_csv(csv_file, nrows=50000)

top_eco_openings = df['Opening'].value_counts().nlargest(40).index.tolist()

# =========================== Tabs ============================================

def get_eco_tab(top_eco_openings):
    tab = dcc.Tab(label='ECO Openings', value='eco', children=[
        html.Div([
            html.H4('ECO codes are a system of naming the most common openings'),
            html.H5('https://chessopenings.com/eco/'),
            html.Label('Select ECO Opening:'),
            dcc.Dropdown(
                id='eco-dropdown',
                options=[{'label': eco, 'value': eco} for eco in top_eco_openings],
                value=top_eco_openings[0]
            ),
            dcc.Loading(
                id='loading-eco-plot',
                type='circle',
                children=[
                    dcc.Graph(id='eco-violin-plot')
                ]
            )
        ])
    ])
    return tab

def get_totals_tab():
    tab = dcc.Tab(label='Totals', value='totals', children=[
        html.Div([
            html.H3('Totals: Checks and Captures'),
            html.Label('Select Game Result(s):'),
            dcc.Checklist(
                id='totals-checklist',
                options=[
                    {'label': 'White', 'value': 'white'},
                    {'label': 'Black', 'value': 'black'},
                    {'label': 'Draw', 'value': 'draw'}
                ],
                value=['white', 'black', 'draw'],
                inline=True
            ),

            dcc.Graph(id='totals-checks-graph'),
            dcc.Graph(id='totals-captures-graph'),
            dcc.Graph(id='totals-strip-plot')
        ])
    ])
    return tab

def get_image_tab():
    IMG_FOLDER = 'assets/images'
    image_divs = []

    for img_file in os.listdir(IMG_FOLDER):
        img_path = f'{IMG_FOLDER}/{img_file}'
        image_divs.append(
            html.Div([
                html.A(
                    html.Button(f"Download {img_file}", title=f"Download {img_file}"),
                    href=img_path,
                    download=img_file
                ),
                html.Br(),
                html.Img(src=img_path, style={'width': '80%', 'height': 'auto'}),
                html.Hr()
            ])
        )

    tab = dcc.Tab(label='Images', value='images', children=html.Div(image_divs))
    return tab

def get_elo_tab(df):
    min_elo = int(df['avg_elo'].min())
    max_elo = int(df['avg_elo'].max())
    tab = dcc.Tab(label='Elo Noramality Analysis', value='elo', children=[
        html.Div([
            html.Label('Select Elo Range:'),
            dcc.RangeSlider(
                id='elo-range-slider',
                min=min_elo,
                max=max_elo,
                step=50,
                value=[1200, 2400],
                marks={i: str(i) for i in range(min_elo, max_elo, 50)}
            ),
            html.Br(),
            html.Label('Select Number of Histogram Bins:'),
            dcc.Slider(
                id='elo-bin-slider',
                min=5,
                max=50,
                step=5,
                value=30,
                marks={i: str(i) for i in range(5, 51, 5)}
            ),
            dcc.Loading(
                id='loading-elo-plot',
                type='circle',
                children=[
                    dcc.Graph(id='elo-hist-kde-plot')
                ]
            ),
            html.H5("Average Elo here (and everywhere else on this site) refers to the average elo of the white and black player that played in the game entry."),
            html.Br(),
            html.Label("Select Normality Test:"),
            dcc.Dropdown(
                id='normality-test-dropdown',
                options=[
                    {'label': 'Shapiro-Wilk', 'value': 'shapiro'},
                    {'label': 'Kolmogorov-Smirnov', 'value': 'ks'},
                    {'label': "D’Agostino-Pearson", 'value': 'dagostino'}
                ],
                value='shapiro'
            ),

            html.Br(),
            html.Button("Run Normality Test", id='normality-btn', n_clicks=0),

            html.Br(), html.Br(),
            html.Div(id='normality-output')
        ])
    ])
    return tab

def get_pca_tab():
    return dcc.Tab(label='PCA', value='pca', children=[
        html.Div([
            html.H3("PCA Dimensionality Reduction"),
            html.Button("Run PCA", id="run-pca-btn", n_clicks=0),
            html.Br(), html.Br(),
            dcc.Graph(id="pca-graph")
        ])
    ])

def get_outlier_tab():
    tab = dcc.Tab(label='Outlier Removal', children=[
        html.Br(),

        html.Label("Select Outlier Detection Method:"),
        dcc.Dropdown(
            id='method-dropdown',
            options=[
                {'label': 'iqr', 'value': 'iqr'},
                {'label': 'Z-Score', 'value': 'Z'}
            ],
            value='iqr'
        ),

        html.Br(),
        html.Label("Threshold (default is 1.5):"),
        dcc.Input(
            id='threshold-input',
            type='number',
            value=1.5,
            step=0.1,
            style={'width': '100px'}
        ),

        html.Br(), html.Br(),
        html.Button("Remove Outliers", id='remove-btn', n_clicks=0),

        html.Br(), html.Br(),
        html.Div(id='summary-output'),

        html.Br(), html.Br(),
        html.Button("Reset Dataset", id='reset-btn', n_clicks=0)
    ])
    return tab

def get_first_events_tab():
    return dcc.Tab(label='First Events', value='first-events', children=[
        html.Div([
            html.H3('Timing of First Major Events in Chess Games'),

            html.H2('The next 2 input options only affect the first 3 graphs'),

            html.Label("Select Event Type:"),
            dcc.Dropdown(
                id='event-timing-dropdown',
                options=[
                    {'label': 'First Castle Turn', 'value': 'first_castle_turn'},
                    {'label': 'First Capture Turn', 'value': 'first_capture_turn'},
                    {'label': 'First Queen Move Turn', 'value': 'first_queen_move_turn'},
                ],
                value='first_castle_turn'
            ),
            html.Label("Select Display Mode:"),
            dcc.RadioItems(
                id='event-mode-radio',
                options=[
                    {'label': 'Combined', 'value': 'combined'},
                    {'label': 'By Color', 'value': 'by_color'}
                ],
                value='combined'
            ),
            dcc.Loading(
                id='loading-line',
                type='circle',
                children=[
                    dcc.Graph(id='line-first-events')
                ]
            ),
            dcc.Loading(
                id='loading-regplot',
                type='circle',
                children=[
                    dcc.Graph(id='regplot-first-events')
                ]
            ),
            dcc.Loading(
                id='loading-pie',
                type='circle',
                children=[
                    dcc.Graph(id='pie-first-events')
                ]
            ),

            html.H3("Hexbin Plot (Plotly equivalent: density_heatmap): Average Elo vs First Castle Turn"),
            dcc.Graph(id='hexbin-plot'),

            html.H3("Joint Plot: Avg Elo vs First Capture Turn"),
            dcc.Graph(id='joint-plot'),

            html.H3("Pair Plot: Elo and First Event Features"),
            dcc.Graph(id='pair-plot'),

            html.H3("Correlation Heatmap of First Events and Average Elo"),
            dcc.Graph(id='correlation-heatmap'),

            html.H3("3D Scatterplot: Elo vs Castle Turn vs Captures"),
            dcc.Graph(id='scatter-3d-plot'),
        ])
    ])

def get_more_dynamic_plots_tab():
    tab = dcc.Tab(label='More Dynamic Plots', value='more-dynamic', children=[
        html.Div([

            html.H3("Bar Plot: Castling Type by Color"),
            dcc.Graph(id='castling-bar-plot'),

            html.H3("Count Plot: Game Results"),
            dcc.Graph(id='result-countplot')
        ])
    ])
    return tab

@my_app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_correlation_heatmap(_):
    cols = ['avg_elo', 'first_queen_move_turn', 'first_capture_turn', 'first_castle_turn']
    corr_matrix = df[cols].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',  # reverse Red-Blue = coolwarm
        zmin=-1,
        zmax=1,
        labels=dict(color='Correlation'),
        aspect='auto'
    )

    fig.update_layout(height=700)

    return fig

# ============================= Dashboard =========================================

# Layout with Tabs
my_app.layout = html.Div([
    html.H1('Chess Data Dashboard - Jeremy Neale', style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='eco', children=[
        get_outlier_tab(),
        get_pca_tab(),
        get_eco_tab(top_eco_openings),
        get_totals_tab(),
        get_elo_tab(df),
        get_first_events_tab(),
        get_more_dynamic_plots_tab(),
        get_image_tab()
    ])
])

# =========================== Callback ==========================================

@my_app.callback(
    Output('eco-violin-plot', 'figure'),
    [Input('eco-dropdown', 'value'),
    Input('reset-btn', 'n_clicks')]
)
def update_violin_plot(selected_eco, _):
    subset = df[df['Opening'] == selected_eco]

    # Create the violin plot showing distribution of winner Elo by color
    fig = px.violin(
        subset,
        x='avg_elo',
        y='Result-string',
        color='Result-string',
        box=True,
        points='all',
        title=f'Distribution of Winner Elo for {selected_eco}',
        labels={'avg_elo': 'Average Elo', 'Result-string': 'Game Result'}
    )
    fig.update_layout()
    return fig


@my_app.callback(
    [Output('totals-checks-graph', 'figure'),
     Output('totals-captures-graph', 'figure')],
    [Input('totals-checklist', 'value'),
    Input('reset-btn', 'n_clicks')]
)
def update_totals_graphs(selected_results, _):
    filtered = df[df['Result-string'].isin(selected_results)]
    # Checks boxplot
    fig_checks = px.box(
        filtered,
        x='Result-string',
        y='total_checks',
        color='Result-string',
        title='Total Checks by Result'
    )
    # Captures boxplot
    fig_captures = px.box(
        filtered,
        x='Result-string',
        y='total_captures',
        color='Result-string',
        title='Total Captures by Result'
    )
    return fig_checks, fig_captures


@my_app.callback(
    Output('totals-strip-plot', 'figure'),
    [Input('totals-checklist', 'value'),
    Input('reset-btn', 'n_clicks')]
)
def update_strip_plot(selected_results, _):
    filtered = df[df['Result-string'].isin(selected_results)]

    fig = px.strip(
        filtered,
        x='Event',
        y='total_captures',
        title='Strip Plot: Total Captures by Game Type',
        labels={'total_captures': 'Total Captures', 'Event': 'Game Type'}
    )
    # Update the plot
    fig.update_traces(jitter=True, opacity=0.5)
    fig.update_layout(xaxis_tickangle=20, height=500)

    return fig

@my_app.callback(
    Output('elo-hist-kde-plot', 'figure'),
    [
        Input('elo-range-slider', 'value'),
        Input('elo-bin-slider', 'value'),
        Input('reset-btn', 'n_clicks')
    ])
def update_elo_hist_kde(elo_range, num_bins, _):
    elo_min, elo_max = elo_range
    filtered = df[(df['avg_elo'] >= elo_min) & (df['avg_elo'] <= elo_max)]

    fig = px.histogram(
        filtered,
        x='avg_elo',
        nbins=num_bins,
        opacity=0.7,
        title='Histogram with KDE Overlay'
    )
    fig.update_layout(
        xaxis_title='Average Elo',
        yaxis_title='Count',
        bargap=0.05,
        height=500
    )
    return fig

@my_app.callback(
    [Output('pie-first-events', 'figure'),
     Output('line-first-events', 'figure'),
     Output('regplot-first-events', 'figure')],
    [Input('event-timing-dropdown', 'value'),
     Input('event-mode-radio', 'value'),
     Input('reset-btn', 'n_clicks')]
)
def update_first_event_figures(selected_event, view_mode, _):
    # To view them combined with Radioitems
    if view_mode == 'combined':
        # Pie chart
        counts = df[selected_event].fillna('neither').value_counts()
        pie_fig = px.pie(
            names=counts.index,
            values=counts.values,
            title=selected_event,
            color=counts.index,
            color_discrete_map={'white': 'gray', 'black': 'black', 'neither': 'blue'}
        )

        # Line chart
        line_data = df[selected_event].value_counts().sort_index()
        line_fig = px.line(
            x=line_data.index,
            y=line_data.values,
        )
        line_fig.update_layout(
            title=f'Turn Distribution by {selected_event}',
            xaxis_title='Turn',
            yaxis_title='Number of Games'
        )

        # Regression plot
        df_event = df.dropna(subset=[selected_event])
        reg_fig = px.scatter(
            df_event,
            x='avg_elo',
            y=selected_event,
            opacity=0.5,
            title=f'{selected_event} by Elo',
            labels={'avg_elo': 'Average Elo', selected_event: 'Turn'}
        )

        # Make regression line and add it to the scatter

        x = df_event['avg_elo']
        y = df_event[selected_event]
        slope, intercept = np.polyfit(x, y, deg=1)

        # Generate line values (middle school algebra throwback)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        reg_fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines'
        ))

        return pie_fig, line_fig, reg_fig
    # To separate them by color
    elif view_mode == 'by_color':
        # Swap out the variable for the version that shows categorical color, not numerical turn number
        event_base = selected_event.replace('_turn', '')
        col = f'{event_base}_color'

        # Pie chart: distribution of which color did the event first
        color_counts = df[col].fillna('neither').value_counts()
        pie_fig = px.pie(
            names=color_counts.index,
            values=color_counts.values,
            title=event_base,
            color=color_counts.index,
            color_discrete_map={'white': 'gray', 'black': 'black', 'neither': 'blue'}
        )

        # Line plot: two lines, one for white and one for black
        df_both = df.dropna(subset=[selected_event, col])
        df_white = df_both[df_both[col] == 'white']
        df_black = df_both[df_both[col] == 'black']

        line_fig = px.line(
            title=selected_event,
            labels={'x': 'Turn', 'y': 'Number of Games'}
        )
        # Plot the white and black lines with scatter
        line_fig.add_scatter(x=df_white[selected_event].value_counts().sort_index().index,
            y=df_white[selected_event].value_counts().sort_index().values,
            mode='lines+markers', name='White', line=dict(color='gray')
        )
        line_fig.add_scatter(x=df_black[selected_event].value_counts().sort_index().index,
            y=df_black[selected_event].value_counts().sort_index().values,
            mode='lines+markers', name='Black', line=dict(color='black')
        )
        line_fig.update_layout(
            title=f'Turn Distribution by {selected_event}',
            xaxis_title='Turn',
            yaxis_title='Number of Games'
        )

        # Regression plot (color-coded)
        reg_fig = px.scatter(
            df_both,
            x='avg_elo',
            y=selected_event,
            color=col,
            color_discrete_map={'white': 'gray', 'black': 'black'},
            opacity=0.5,
            title=f'Avg Elo vs. {selected_event} by Color',
            labels={'avg_elo': 'Average Elo', selected_event: 'Turn'}
        )

        # Make regression line and add it to the scatter

        x = df_both['avg_elo']
        y = df_both[selected_event]
        slope, intercept = np.polyfit(x, y, deg=1)

        # Generate line values (middle school algebra throwback)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        reg_fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines'
        ))

        return pie_fig, line_fig, reg_fig
    else:
        print("Impossible. Logic error 001")
        raise Exception

# Only triggers when the button is pressed
@my_app.callback(
    Output('summary-output', 'children'),
    Input('remove-btn', 'n_clicks'),
    State('method-dropdown', 'value'),
    State('threshold-input', 'value'),
    prevent_initial_call=True
)
def remove_outliers(unneeded_button_press, method, threshold):
    # Use global (I know it's bad practice) to update the var across the file
    global df
    numeric_cols = df.select_dtypes(include='number').columns
    original_len = len(df)

    if method == 'iqr':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif method == 'Z':
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            df = df[z.abs() <= threshold]

    removed = original_len - len(df)
    return f"{removed} outliers removed. Current dataset has {len(df)} rows."


@my_app.callback(
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_df(_):
    global df
    df = pd.read_csv('chess_games_50k.csv')

@my_app.callback(
    Output('normality-output', 'children'),
    Input('normality-btn', 'n_clicks'),
    State('normality-test-dropdown', 'value'),
    State('elo-range-slider', 'value'),
    prevent_initial_call=True
)
def run_normality_test(_, test_type, elo_range):
    global df
    elo_min, elo_max = elo_range
    filtered = df[(df['avg_elo'] >= elo_min) & (df['avg_elo'] <= elo_max)]['avg_elo'].dropna()

    if test_type == 'shapiro':
        stat, p = shapiro(filtered)
        test_name = "Shapiro-Wilk"
    elif test_type == 'ks':
        # Normalize data
        standardized = (filtered - filtered.mean()) / filtered.std()
        stat, p = kstest(standardized, 'norm')
        test_name = "Kolmogorov-Smirnov"
    elif test_type == 'dagostino':
        stat, p = normaltest(filtered)
        test_name = "D’Agostino-Pearson"
    else:
        return "ERROR: Unknown test selected."

    alpha = 0.05
    if p > alpha:
        res = "Data IS normally distributed."
    else:
        res = "Data is NOT normally distributed."

    return html.Div([
        html.H5(f"{test_name} Test Result:"),
        html.P(f"Test Statistic: {stat:.4f}"),
        html.P(f"p-value: {p:.4f}"),
        html.P(f"Result (α = 0.05): {res}")
    ])

@my_app.callback(
    Output("pca-graph", "figure"),
    Input("run-pca-btn", "n_clicks"),
    prevent_initial_call=True
)
def run_pca(_):
    numeric = df.select_dtypes(include='number').dropna()
    scaled = MinMaxScaler().fit_transform(numeric)
    pca = PCA()
    pca.fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig = px.line(x=np.arange(1, len(cumulative)+1), y=cumulative,
        labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
        title='PCA Cumulative Explained Variance')
    fig.update_layout(xaxis=dict(tickmode='linear'))
    return fig

@my_app.callback(
    Output('scatter-3d-plot', 'figure'),
    [Input('reset-btn', 'n_clicks')]
)
def update_3d_scatter(_):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['avg_elo'],
        y=df['first_castle_turn'],
        z=df['total_captures'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['total_captures'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Average Elo',
            yaxis_title='First Castle Turn',
            zaxis_title='Total Captures'
        ),
        title='3D Scatter: Elo vs Castle Turn vs Captures',
        height=1000
    )

    return fig

@my_app.callback(
    Output('hexbin-plot', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_hexbin(_):
    fig = px.density_heatmap(
        df,
        x='avg_elo',
        y='first_castle_turn',
        nbinsx=30,
        nbinsy=30,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_title='Average Elo',
        yaxis_title='First Castle Turn'
    )
    return fig

@my_app.callback(
    Output('joint-plot', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_joint_plot(_):
    # Manually make the joint plot with plotly's graph object (go)
    valid = df[['avg_elo', 'first_capture_turn']].dropna()
    x = valid['avg_elo']
    y = valid['first_capture_turn']

    # Main scatter plot
    scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Scatter'
    )

    # Makes a KDE overlay
    contour = go.Histogram2dContour(
        x=x,
        y=y,
        opacity=0.6,
        ncontours=5,
        showscale=False,
        name='KDE'
    )

    layout = go.Layout(
        xaxis=dict(title='Average Elo'),
        yaxis=dict(title='First Capture Turn')
    )

    fig = go.Figure(data=[contour, scatter], layout=layout)
    return fig

@my_app.callback(
    Output('castling-bar-plot', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_castling_bar_plot(_):
    white_counts = df['white_castle'].value_counts(dropna=False)
    black_counts = df['black_castle'].value_counts(dropna=False)

    labels = ['kingside', 'queenside', np.nan]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=white_counts.reindex(labels),
        name='White',
        marker=dict(color='#dddddd')
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=black_counts.reindex(labels),
        name='Black',
        marker=dict(color='black')
    ))

    fig.update_layout(
        barmode='group',
        xaxis_title='Castling Type',
        yaxis_title='Number of Games',
        template='simple_white',
        legend=dict(x=0.8, y=1.0)
    )

    return fig

@my_app.callback(
    Output('pair-plot', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_pair_plot(_):
    cols = [
        'avg_elo',
        'first_queen_move_turn',
        'first_capture_turn',
        'first_castle_turn',
        'Result-string'  # hue
    ]

    df_clean = df[cols].dropna()

    fig = px.scatter_matrix(
        df_clean,
        dimensions=cols[:-1],  # drop result string
        color='Result-string',
        labels={
            'avg_elo': 'Avg Elo',
            'first_queen_move_turn': 'First Queen Move',
            'first_capture_turn': 'First Capture Turn',
            'first_castle_turn': 'First Castle Turn'
        },
        color_discrete_map={'white': 'grey', 'black': 'black', 'draw': 'blue'}
    )

    fig.update_layout(height=800)
    return fig

@my_app.callback(
    Output('result-countplot', 'figure'),
    Input('reset-btn', 'n_clicks')
)
def update_result_countplot(_):
    result_counts = df['Result-string'].value_counts().reindex(['white', 'black', 'draw'])

    fig = px.bar(
        x=result_counts.index,
        y=result_counts.values,
        color=result_counts.index,
        color_discrete_map={
            'white': '#dddddd',
            'black': 'black',
            'draw': 'blue'
        }
    )

    fig.update_layout(
        xaxis_title='Result',
        yaxis_title='Number of Games',
        template='simple_white',
        showlegend=True
    )

    return fig

# ================================================================================

if __name__ == '__main__':
    print("Running app...")
    if run_type == 'LOCAL':
        my_app.run(debug=True, port=8080)
    elif run_type == 'SERVER':
        my_app.run_server(debug=True, host='0.0.0.0', port=8080)
    else:
        print('run_type must be LOCAL or SERVER.')
