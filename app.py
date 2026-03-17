import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# === 1. CONFIGURATION ===
PREDEFINED_ZONES = [
    "Wayanad North", "Wayanad South", "Palakkad Gap", "Nilambur North", 
    "Nilambur South", "Periyar East", "Periyar West", "Idukki Central", 
    "Munnar High Range", "Agasthyavanam", "Silent Valley", "Aralam"
]

# === 2. INITIALIZATION & SERVER EXPORT ===
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP], 
    suppress_callback_exceptions=True
)
server = app.server  # CRITICAL: This allows Render/Gunicorn to host the app

# === 3. DATA ENGINE ===
def load_data():
    # Robust pathing: looks for 'data' folder in the same directory as app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'realtime_predictions.csv')
    
    if not os.path.exists(data_path):
        # Fallback dummy data for initial build testing
        print(f"⚠️ Warning: {data_path} not found. Loading fallback.")
        return pd.DataFrame({
            'lat': [10.85, 10.12], 'lon': [76.27, 76.80], 'fire_prob': [0.95, 0.65], 
            'temperature': [42.11, 38.45], 'ndvi': [0.12, 0.22], 'forest_zone': ['Palakkad Gap', 'Idukki Central']
        })
    
    try:
        df = pd.read_csv(data_path)
        
        # Handle coordinates from GEE .geo column if lat/lon aren't explicit
        if '.geo' in df.columns and ('lat' not in df.columns):
            def extract_coords(geo_str):
                try:
                    geo_data = json.loads(geo_str)
                    return geo_data['coordinates'][1], geo_data['coordinates'][0]
                except: return None, None
            df['lat'], df['lon'] = zip(*df['.geo'].apply(extract_coords))
            df = df.dropna(subset=['lat', 'lon'])

        df['temperature'] = df['temperature'].round(2)
        if 'forest_zone' not in df.columns:
            df['forest_zone'] = df['grid_id'].astype(str) if 'grid_id' in df.columns else "Sector " + df.index.astype(str)
            
        return df
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return pd.DataFrame()

# === 4. UI COMPONENTS ===
def create_metric(label, value, icon, color="#00f2ff"):
    return html.Div([
        html.Div([html.I(className=f"bi bi-{icon} me-2", style={'color': '#ff4400'}), html.Span(label)]),
        html.H2(value, style={'marginTop': '10px', 'color': color, 'fontFamily': 'Orbitron'})
    ], className="p-3", style={'border': '1px solid #444', 'borderRadius': '8px', 'background': 'rgba(255,255,255,0.05)'})

def create_alert_list(df):
    if df.empty:
        return [dbc.Alert("Scanning Sahyadri Range... No data found.", color="secondary")]
    
    df_sorted = df.sort_values(by='fire_prob', ascending=False)
    alert_elements = []

    for _, row in df_sorted.iterrows():
        prob = int(row['fire_prob'] * 100)
        if prob >= 80:
            alert_elements.append(dbc.Alert([
                html.Strong(f"CRITICAL: {row['forest_zone']}"),
                html.P(f"Extreme Risk: {prob}% | Temp: {row['temperature']}°C", className="mb-0", style={'fontSize': '0.8rem'})
            ], color="danger", className="mb-2", style={'borderLeft': '5px solid #ff0000'}))
        elif prob >= 50:
            alert_elements.append(dbc.Alert([
                html.Strong(f"ADVISORY: {row['forest_zone']}"),
                html.P(f"Moderate Risk: {prob}% | Temp: {row['temperature']}°C", className="mb-0", style={'fontSize': '0.8rem'})
            ], color="warning", className="mb-2", style={'borderLeft': '5px solid #ffc107', 'color': '#000'}))
        else:
            alert_elements.append(dbc.Alert([
                html.Span(f"NORMAL: {row['forest_zone']} ({prob}%)"),
            ], color="info", className="mb-1", style={'fontSize': '0.75rem', 'opacity': '0.7'}))
            
    return alert_elements

# === 5. LAYOUT ===
app.layout = html.Div([
    dcc.Store(id='auth-store', data={'authenticated': False}, storage_type='session'),
    dcc.Store(id='page-state', data='map'),
    html.Div(id='master-render-engine')
])

def render_dashboard(current_page):
    df = load_data()
    prediction_time = (datetime.now() + timedelta(hours=48)).strftime('%d %b %Y, %H:%M')
    critical_count = len(df[df['fire_prob'] >= 0.8]) if not df.empty else 0
    
    sidebar = html.Div([
        html.H4("SAHYADRI AI", style={'color': '#ff4400', 'fontFamily': 'Orbitron', 'marginBottom': '50px'}),
        dbc.Nav([
            dbc.NavLink("INCIDENT MAP", id="nav-map", active=(current_page == 'map'), n_clicks=0, href="#"),
            dbc.NavLink("FIELD REPORTING", id="nav-field", active=(current_page == 'field'), n_clicks=0, href="#"),
            dbc.NavLink("SATELLITE LOGS", id="nav-logs", active=(current_page == 'logs'), n_clicks=0, href="#"),
        ], vertical=True, pills=True),
        dbc.Button("LOGOUT", id="logout-btn", color="danger", size="sm", className="mt-auto")
    ], style={'width': '250px', 'position': 'fixed', 'height': '100vh', 'padding': '20px', 'borderRight': '1px solid #444', 'display': 'flex', 'flexDirection': 'column'})

    if current_page == "field":
        content = html.Div([
            dbc.Alert("✅ DATABASE SYNCED", id="sync-alert", is_open=False, duration=3000, color="success"),
            html.H2("FIELD REPORTING", style={'fontFamily': 'Orbitron'}),
            html.Div([
                html.P("RANGE OFFICER LOG", style={'color': '#ff4400', 'fontWeight': 'bold'}),
                dcc.Dropdown(id='zone-sel', options=[{'label': z, 'value': z} for z in PREDEFINED_ZONES], placeholder="SELECT ZONE", style={'color': '#000'}),
                dbc.RadioItems(id='obs-radio', options=[{"label": "ACTIVE FIRE", "value": 1}, {"label": "SMOKE", "value": 2}, {"label": "CLEAR", "value": 3}], className="my-4", style={'color': '#00f2ff'}),
                dbc.Button("SYNC TO STATE", id="submit-val", color="warning", className="w-100")
            ], className="p-4 mt-4", style={'background': 'rgba(255,255,255,0.05)', 'borderRadius': '10px', 'maxWidth': '500px'})
        ])
    elif current_page == "logs":
        log_rows = [html.P(f"> [SCAN] {r['forest_zone']}: {r['temperature']}°C | NDVI: {r['ndvi']}") for _, r in df.iterrows()]
        content = html.Div([
            html.H2("TELEMETRY LOGS", style={'fontFamily': 'Orbitron'}),
            html.Div([html.P(f"> [OK] Uplink Active. TARGET: {prediction_time}"), *log_rows], 
                     style={'background': '#000', 'color': '#00ff00', 'fontFamily': 'monospace', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '20px'})
        ])
    else: # Map View
        fig = px.density_map(df, lat='lat', lon='lon', z='fire_prob', radius=20, zoom=7, center=dict(lat=10.5, lon=76.5), color_continuous_scale="Reds", map_style="carto-darkmatter")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        content = html.Div([
            html.H1("KERALA FOREST COMMAND", style={'fontFamily': 'Orbitron'}),
            html.P(f"PREDICTION WINDOW: {prediction_time} (48H ADVANCE)", style={'color': '#ff4400', 'fontWeight': 'bold'}),
            dbc.Row([
                dbc.Col(create_metric("ACTIVE BEATS", len(df), "radar")),
                dbc.Col(create_metric("MAX TEMP", f"{df['temperature'].max():.2f}°C" if not df.empty else "N/A", "thermometer-sun")),
                dbc.Col(create_metric("CRITICAL ALERTS", critical_count, "exclamation-octagon", color="#ff0000")),
            ], className="my-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig, style={'height': '500px'}, config={'displayModeBar': False}), width=8),
                dbc.Col([
                    html.H5("PREDICTED ALERTS (ALL)", style={'color': '#ff4400', 'fontSize': '0.9rem'}),
                    html.Div(create_alert_list(df), style={'maxHeight': '500px', 'overflowY': 'auto'})
                ], width=4)
            ])
        ])

    return html.Div([sidebar, html.Div(content, style={'marginLeft': '270px', 'padding': '40px'})])

# === 6. CALLBACKS ===
@app.callback(Output('master-render-engine', 'children'), [Input('auth-store', 'data'), Input('page-state', 'data')])
def render_engine(auth, page):
    if not auth or not auth.get('authenticated'):
        return html.Div([
            html.Div([
                html.H1("SAHYADRI AI", style={'fontFamily': 'Orbitron', 'color': 'white'}),
                html.P("SECURE UPLINK REQUIRED", style={'color': '#ff4400', 'fontSize': '0.8rem'}),
                dbc.Input(id="pwd-input", type="password", placeholder="ENTER KEY", className="mb-3 text-center"),
                dbc.Button("AUTHORIZE", id="login-btn", n_clicks=0, className="w-100", style={'background': '#ff4400', 'border': 'none'}),
                html.Div(id="login-error-output", className="mt-3", style={'color': '#ff4400'})
            ], style={'maxWidth': '400px', 'margin': '150px auto', 'textAlign': 'center', 'padding': '40px', 'background': '#0a0a0a', 'border': '1px solid #333'})
        ])
    return render_dashboard(page)

@app.callback([Output('page-state', 'data'), Output('auth-store', 'data', allow_duplicate=True)],
    [Input('nav-map', 'n_clicks'), Input('nav-field', 'n_clicks'), Input('nav-logs', 'n_clicks'), Input('logout-btn', 'n_clicks')],
    prevent_initial_call=True)
def nav_handler(n1, n2, n3, n_logout):
    ctx = callback_context
    if not ctx.triggered: return 'map', no_update
    tid = ctx.triggered[0]['prop_id'].split('.')[0]
    if tid == 'logout-btn': return 'map', {'authenticated': False}
    return tid.split('-')[1], no_update

@app.callback(
    [Output('auth-store', 'data'), Output('login-error-output', 'children')],
    [Input('login-btn', 'n_clicks'), Input('pwd-input', 'n_submit')],
    State('pwd-input', 'value'),
    prevent_initial_call=True
)
def auth_process(n, pwd):
    # If the button hasn't been clicked yet, don't show anything
    if n == 0 or n is None:
        return no_update, ""
    
    # If password is correct
    if pwd == "kerala_forest_2026":
        return {'authenticated': True}, ""
    
    # If password field is empty but button was clicked
    if not pwd:
        return no_update, "PLEASE ENTER PASSWORD"
    
    # If password was entered but is incorrect
    return no_update, "❌ ACCESS DENIED: INVALID PASSWORD"

@app.callback(Output("sync-alert", "is_open"), Input("submit-val", "n_clicks"), [State("zone-sel", "value"), State("obs-radio", "value")], prevent_initial_call=True)
def sync_process(n, z, o): return True if n and z and o else False

if __name__ == '__main__':
    # Production entry point
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
