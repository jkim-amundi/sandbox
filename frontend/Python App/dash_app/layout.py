
"""
Main Layout Module

This module defines the Dash application layout components including:
- App initialization
- Header components
- Sidebar components
- Main content containers
- Modal dialogs
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_table.Format import Format, Scheme
import pandas as pd

# Import configuration and data
from .config_and_data import (
    config, 
    country_options, 
    dropdown_texts,
    metric_options,
    metric_descriptions
)

# Get style settings from config
style_config = config["app_settings"]["style"]

# Initialize the Dash app
external_stylesheets = [
    getattr(dbc.themes, style_config["theme"]),
    style_config["icons_url"]
]

app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets, 
    assets_folder="assets", 
    suppress_callback_exceptions=True
)

# ===================
# Layout Components
# ===================

# Amundi logo image
fixed_image = html.Img(
    src=app.get_asset_url("Amundi.jpg"),
    style={
        "position": "relative",
        "top": "10px",
        "right": "10px",
        "height": style_config["header_height"]
    }
)

# Header row with logo, title, and navigation buttons
header_row = dbc.Row(
    [
        dbc.Col(fixed_image, width="auto"),
        dbc.Col(
            html.H1(config["app_settings"]["header_title"], className="text-center"),
            className="d-flex align-items-center justify-content-center",
        ),
        dbc.Col(
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        dropdown_texts["btn_actual"], 
                        id="btn-actual", 
                        style={"background-color": style_config["primary_color"], "color": "rgb(255,255,255)"}, 
                        className="m-1"
                    ),
                    dbc.Button(
                        dropdown_texts["btn_historical"], 
                        id="btn-historical", 
                        style={"background-color": style_config["secondary_color"], "color": "rgb(255,255,255)"}, 
                        className="m-1"
                    ),
                    dbc.Button(
                        dropdown_texts["btn_about"], 
                        id="btn-about", 
                        style={"background-color": style_config["tertiary_color"], "color": "rgb(255,255,255)"}, 
                        className="m-1"
                    ),
                ]
            ),
            width="auto",
            className="d-flex align-items-center justify-content-end",
        ),
    ],
    align="center",
    className="mb-3"
)

# Title row
title_row = dbc.Row(
    dbc.Col(
        html.H1(config["app_settings"]["header_title"], className="text-center mt-3 mb-4")
    )
)

# Right sidebar with country selection and data
country_info_style = config["layout"]["country_info"]
accordion_config = config["layout"]["accordion"]

right_sidebar = dbc.Col(
    [
        html.Label(
            dropdown_texts["dropdown_countries_label"], 
            style={"fontWeight": "bold"}
        ),
        dcc.Dropdown(
            id="dropdown-country",
            options=country_options,
            placeholder="Select a country...",
            value="All",
            style={"width": "100%"}
        ),
        html.Br(),
        html.Div(
            [
                html.Img(
                    id="country-flag", 
                    style={"height": country_info_style["flag_height"], 
                           "margin-right": country_info_style["margin_right"]}
                ),
                html.P(id="country-weight", className="mt-2 mb-0"),
                html.P(id="country-sectors", className="mt-0"),
                html.P(id="country-scores", className="mt-0") 
            ],
            style={"textAlign": country_info_style["text_align"]}
        ),
        html.Hr(),
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [html.P("", id="pop-data")],
                    title="Country Data",
                    id="country-data-accordion",
                    item_id="country_data"
                ),
                dbc.AccordionItem(
                    [html.Div(id="economic-table")],
                    title="Economic Data",
                    id="economic-data-accordion",
                    item_id="economic"
                ),
                dbc.AccordionItem(
                    [html.Div(id="forecast-table")],
                    title="Institute Forecast",
                    id="forecast-accordion",
                    item_id="forecast"
                ),
                dbc.AccordionItem(
                    html.Div(id="financial-data-container"),
                    title="Financial Data",
                    item_id="financial"
                )
            ],
            always_open=accordion_config["always_open"],
            active_item=accordion_config["default_active"]
        ),
    ],
    id="right-sidebar",
    width=config["layout"]["sidebar"]["width"],
    className=config["layout"]["sidebar"]["className"],
)

# Store components for selected Metric and Scenario
store_components = [
    dcc.Store(id="selected-metric", data="All"),
    dcc.Store(id="selected-scenario", data="standard")
]

# Row with clickable links for Metric and Scenario selection
selection_row = dbc.Row(
    [
        dbc.Col(
            html.H4("Macro Positioning", className="mb-0"),
            width="auto"
        ),
        dbc.Col(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Metric:", className="mb-0 me-2"),
                            dbc.Button(
                                "All", 
                                id="metric-link", 
                                n_clicks=0, 
                                color="link", 
                                style={"padding": "0", "fontWeight": "normal"}
                            )
                        ],
                        width="auto",
                        className="d-flex align-items-center"
                    ),
                    dbc.Col(
                        [
                            html.Label("Scenario:", className="mb-0 me-2"),
                            dbc.Button(
                                "Standard", 
                                id="scenario-link", 
                                n_clicks=0, 
                                color="link", 
                                style={"padding": "0", "fontWeight": "normal"}
                            )
                        ],
                        width="auto",
                        className="d-flex align-items-center"
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Extract Data", 
                            id="extract-data-button", 
                            n_clicks=0, 
                            color="link", 
                            style={"padding": "0", "fontWeight": "normal"}
                        ),
                        width="auto",
                        className="d-flex align-items-center"
                    )
                ],
                className="g-2"
            ),
            width=True,
            className="d-flex justify-content-end align-items-center"
        )
    ],
    className="mb-3",
    id="selection-row"  # Aggiungi un ID al selection_row
)

# Modifica il about_content per includere il selection_row ma nasconderlo
about_content = html.Div(
    [
        # Includi il selection_row ma con stile display:none
        html.Div(
            selection_row,
            style={"display": "none"}
        ),
        dcc.Markdown(id="institute-view-content", 
                    children="", 
                    style={"padding": "20px"})
    ]
)

# Modifica il historical_content per includere il selection_row ma nasconderlo
historical_content = html.Div(
    [
        # Includi il selection_row ma con stile display:none
        html.Div(
            selection_row,
            style={"display": "none"}
        ),
        dcc.Markdown(id="frv-content", 
                    children="", 
                    style={"padding": "20px"})
    ]
)

# Modifica il actual_content per rendere esplicito il selection_row
actual_content = dbc.Container([
    selection_row,
    dbc.Row([
         dbc.Col(
             dcc.Graph(id="polar-plot", style={"height": "400px", "width": "100%"}), 
             width=6, 
             id="polar-plot-col"
         ),
         dbc.Col(
             dcc.Graph(id="bar-plot", style={"height": "400px", "width": "100%"}), 
             width=6, 
             id="bar-plot-col"
         )
    ]),
    dbc.Row([
         dbc.Col(html.Div(id="dashboard-table-container"), width=12)
    ]),
    dbc.Col(html.Hr(), width=12),
    dbc.Row([dbc.Col(html.H4("Index overview", className="text-left"), width=12)]),
    dbc.Row([dbc.Col(html.H5("Total Return Decomposition", className="text-left"), width=12)]),
    dbc.Row([dbc.Col(html.Div(id="total-return-decomposition"), width=12)]),
    dbc.Col(html.Hr(), width=12),
    dbc.Row([dbc.Col(html.Div(id="flow-analysis-section"), width=12)]),
])

# Modal for Metric selection
metric_modal = dbc.Modal(
    [
        dbc.ModalHeader(dropdown_texts["modal_metric_title"]),
        dbc.ModalBody([
            dbc.ListGroup([
                dbc.ListGroupItem(
                    [
                        html.Div("All", style={"fontWeight": "bold"}),
                        html.Div(metric_descriptions["All"], style={"fontSize": "smaller", "color": "gray"})
                    ],
                    id="metric-option-all", action=True
                ),
                dbc.ListGroupItem(
                    [
                        html.Div("Fundamentals", style={"fontWeight": "bold"}),
                        html.Div(metric_descriptions["Fundamentals"], style={"fontSize": "smaller", "color": "gray"})
                    ],
                    id="metric-option-fundamentals", action=True
                ),
                dbc.ListGroupItem(
                    [
                        html.Div("Risks", style={"fontWeight": "bold"}),
                        html.Div(metric_descriptions["Risks"], style={"fontSize": "smaller", "color": "gray"})
                    ],
                    id="metric-option-risks", action=True
                ),
                dbc.ListGroupItem(
                    [
                        html.Div("Valuations", style={"fontWeight": "bold"}),
                        html.Div(metric_descriptions["Valuations"], style={"fontSize": "smaller", "color": "gray"})
                    ],
                    id="metric-option-valuations", action=True
                )
            ])
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-metric-modal", className="ml-auto"))
    ],
    id="metric-modal",
    is_open=False
)

# Modal for Scenario selection
scenario_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            [
                html.Span(dropdown_texts["modal_scenario_title"]),
                dbc.Button(
                    html.I(className="bi bi-info-circle"),
                    id="scenario-info-btn",
                    color="link",
                    style={"marginLeft": "10px"}
                )
            ]
        ),
        dbc.ModalBody([
            dbc.ListGroup([
                dbc.ListGroupItem(
                    [
                        html.Div(scenario["name"], style={"fontWeight": "bold"}),
                        html.Div(scenario["description"], style={"fontSize": "smaller", "color": "gray"})
                    ],
                    id=f"scenario-option-{scenario['id']}", action=True
                ) 
                for scenario in config["scenarios"]["list"]
            ])
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-scenario-modal", className="ml-auto"))
    ],
    id="scenario-modal",
    is_open=False
)

# Modal for PPT extraction
extract_modal = dbc.Modal(
    [
        dbc.ModalHeader("Extract Data"),
        dbc.ModalBody([
            html.Label("Select Countries:"),
            dcc.Checklist(
                id="extract-country-checklist",
                options=[
                    {"label": c, "value": c} 
                    for c in ["All"] + sorted([opt["value"] for opt in country_options if opt["value"] != "All"])
                ],
                value=[],
                labelStyle={"display": "block", "margin": "5px 0"}
            ),
            html.Br(),
            dbc.Button("Download PPT", id="download-ppt-button", color="primary")
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-extract-modal", className="ml-auto"))
    ],
    id="extract-modal",
    is_open=False
)

# Download component for PPT download
download_component = dcc.Download(id="download-ppt")

# Assemble the complete app layout
app_layout = dbc.Container(
    store_components +
    [
        header_row,
        dbc.Row([
            dbc.Col(
                html.Div(id="content", children=actual_content), 
                width=config["layout"]["content"]["width"]
            ),
            right_sidebar
        ]),
        extract_modal,
        download_component,
        metric_modal,
        scenario_modal
    ],
    fluid=config["layout"]["containers"]["main"]["fluid"],
)

# Attach the layout to the app
app.layout = app_layout


# ===============================
# Functions for chart generation
# ===============================

def make_stacked_bar_for_all_indices():
    """
    Create a stacked bar chart for all global indices.
    
    Returns:
        plotly.graph_objects.Figure: Stacked bar chart
    """
    import plotly.graph_objects as go
    from .utils import cagr
    import math
    
    skip_rows = config["file_paths"]["bloomberg_skiprows"]
    
    # Labels and colors for chart components
    stacked_config = config["charts"]["stacked_bar"]["variables"]
    labels = {var["id"]: var["label"] for var in stacked_config}
    colors = {var["id"]: var["color"] for var in stacked_config}
    
    # Add dividend configuration
    dividend_config = config["charts"]["stacked_bar"]["dividend_config"]
    labels[dividend_config["id"]] = dividend_config["label"]
    colors[dividend_config["id"]] = dividend_config["color"]

    # List of global indices
    index_list = config["charts"]["financial"]["stacked"]
    
    # Time calculation constant
    year_days = config["data_processing"]["calculations"]["year_days"]

    # Data container for chart
    data_attr = {}
    
    # Process each index
    for entry in index_list:
        label, sheet_name = entry["name"], entry["sheet_name"]
        
        try:
            # Load data for this index
            df = pd.read_excel(config["file_paths"]["bloomberg"], sheet_name=sheet_name, skiprows=skip_rows)
        except Exception:
            data_attr[label] = None
            continue
            
        if df.empty:
            data_attr[label] = None
            continue
            
        # Get date column
        date_col = df.columns[0]
        
        # Required columns for calculations
        required_cols = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", 
                       "INDX_ADJ_PE", "PX_LAST", "TOTAL_RETURN"]
                       
        if any(col not in df.columns for col in required_cols):
            data_attr[label] = None
            continue
            
        # Filter to required columns and remove rows with missing dates
        df = df[[date_col] + required_cols].dropna(subset=[date_col])
        
        if len(df) < 2:
            data_attr[label] = None
            continue
            
        # Get first and last rows for calculations
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        # Calculate time difference
        if isinstance(first_row[date_col], pd.Timestamp) and isinstance(last_row[date_col], pd.Timestamp):
            days = (last_row[date_col] - first_row[date_col]).days
        else:
            days = 0
            
        year_frac = days / year_days if days > 0 else 1.0

        # Calculate CAGR for each component
        annual_sales = cagr(first_row["TRAIL_12M_SALES_PER_SH"], last_row["TRAIL_12M_SALES_PER_SH"], year_frac)
        annual_margin = cagr(first_row["TRAIL_12M_PROF_MARGIN"], last_row["TRAIL_12M_PROF_MARGIN"], year_frac)
        annual_pe = cagr(first_row["INDX_ADJ_PE"], last_row["INDX_ADJ_PE"], year_frac)
        annual_px_last = cagr(first_row["PX_LAST"], last_row["PX_LAST"], year_frac)
        annual_total = cagr(first_row["TOTAL_RETURN"], last_row["TOTAL_RETURN"], year_frac)
        
        # Calculate log returns
        log_sales = math.log(1 + annual_sales) if annual_sales is not None else 0
        log_margin = math.log(1 + annual_margin) if annual_margin is not None else 0
        log_pe = math.log(1 + annual_pe) if annual_pe is not None else 0
        log_px_last = math.log(1 + annual_px_last) if annual_px_last is not None else 0
        log_total = math.log(1 + annual_total) if annual_total is not None else 0
        
        # For dividends, use difference between total return and price
        log_dividends = (annual_total - annual_px_last) if (annual_total is not None and annual_px_last is not None) else 0

        # Calculate ratio for rebasing
        ratio = (annual_total / log_total) if (log_total != 0) else 0

        # Calculate rebased annualized return for each component
        rebased_sales = log_sales * ratio
        rebased_margin = log_margin * ratio
        rebased_pe = log_pe * ratio
        rebased_dividends = log_dividends * ratio

        # Store rebased values for this index
        comp_rebased = {
            "TRAIL_12M_SALES_PER_SH": rebased_sales,
            "TRAIL_12M_PROF_MARGIN": rebased_margin,
            "INDX_ADJ_PE": rebased_pe,
            "DIVIDENDS": rebased_dividends
        }
        data_attr[label] = comp_rebased

    # Create stacked bar chart
    fig = go.Figure()
    x_indices = [entry["name"] for entry in index_list]
    components = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "DIVIDENDS"]
    
    # Add trace for each component
    for comp in components:
        y_vals = []
        for idx in x_indices:
            if data_attr.get(idx):
                y_vals.append(data_attr[idx][comp] * 100)  # Convert to percentage
            else:
                y_vals.append(0)
                
        fig.add_trace(go.Bar(
            x=x_indices,
            y=y_vals,
            name=labels.get(comp, comp),
            marker_color=colors.get(comp, "gray")
        ))
    
    # Get chart layout settings
    common_layout = config["charts"]["layout"]["common"]
    default_title = config["charts"]["financial"]["default_titles"]["stacked_bar_all"]
    
    # Update layout
    fig.update_layout(
        barmode="stack",
        title=default_title,
        xaxis_title="Index",
        yaxis_title="Rebased Annualized Return (%)",
        plot_bgcolor=common_layout["plot_bgcolor"],
        legend=dict(
            orientation=common_layout["legend_position"]["orientation"],
            yanchor=common_layout["legend_position"]["yanchor"],
            y=common_layout["legend_position"]["y"],
            xanchor=common_layout["legend_position"]["xanchor"],
            x=common_layout["legend_position"]["x"]
        )
    )
    
    return dcc.Graph(figure=fig)


def make_stacked_bar_for_country(selected_country):
    """
    Create a stacked bar chart comparing a country to MXEF.
    
    Args:
        selected_country: Country to compare with MXEF
        
    Returns:
        plotly.graph_objects.Figure: Stacked bar chart
    """
    import plotly.graph_objects as go
    from .utils import cagr
    import math
    
    skip_rows = config["file_paths"]["bloomberg_skiprows"]
    
    # Labels and colors for chart components
    stacked_config = config["charts"]["stacked_bar"]["variables"]
    labels = {var["id"]: var["label"] for var in stacked_config}
    colors = {var["id"]: var["color"] for var in stacked_config}
    
    # Add dividend configuration
    dividend_config = config["charts"]["stacked_bar"]["dividend_config"]
    labels[dividend_config["id"]] = dividend_config["label"]
    colors[dividend_config["id"]] = dividend_config["color"]

    # Get sheet name for selected country
    mapping = config["countries"]["mapping"].get(selected_country, {})
    country_sheet = mapping.get("sheet_name", "")
    
    # Define indices to compare
    index_list = [
        {"name": "MXEF", "sheet_name": "MXEF Index"},
        {"name": selected_country, "sheet_name": country_sheet}
    ]
    
    # Time calculation constant
    year_days = config["data_processing"]["calculations"]["year_days"]

    # Data container for chart
    data_attr = {}
    
    # Process each index
    for entry in index_list:
        label, sheet_name = entry["name"], entry["sheet_name"]
        
        try:
            # Load data for this index
            df = pd.read_excel(config["file_paths"]["bloomberg"], sheet_name=sheet_name, skiprows=skip_rows)
        except Exception:
            data_attr[label] = None
            continue
            
        if df.empty:
            data_attr[label] = None
            continue
            
        # Get date column
        date_col = df.columns[0]
        
        # Required columns for calculations
        required_cols = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", 
                       "INDX_ADJ_PE", "PX_LAST", "TOTAL_RETURN"]
                       
        if any(col not in df.columns for col in required_cols):
            data_attr[label] = None
            continue
            
        # Filter to required columns and remove rows with missing dates
        df = df[[date_col] + required_cols].dropna(subset=[date_col])
        
        if len(df) < 2:
            data_attr[label] = None
            continue
            
        # Get first and last rows for calculations
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        # Calculate time difference
        if isinstance(first_row[date_col], pd.Timestamp) and isinstance(last_row[date_col], pd.Timestamp):
            days = (last_row[date_col] - first_row[date_col]).days
        else:
            days = 0
            
        year_frac = days / year_days if days > 0 else 1.0

        # Calculate CAGR for each component
        annual_sales = cagr(first_row["TRAIL_12M_SALES_PER_SH"], last_row["TRAIL_12M_SALES_PER_SH"], year_frac)
        annual_margin = cagr(first_row["TRAIL_12M_PROF_MARGIN"], last_row["TRAIL_12M_PROF_MARGIN"], year_frac)
        annual_pe = cagr(first_row["INDX_ADJ_PE"], last_row["INDX_ADJ_PE"], year_frac)
        annual_px_last = cagr(first_row["PX_LAST"], last_row["PX_LAST"], year_frac)
        annual_total = cagr(first_row["TOTAL_RETURN"], last_row["TOTAL_RETURN"], year_frac)
        
        # Calculate log returns
        log_sales = math.log(1 + annual_sales) if annual_sales is not None else 0
        log_margin = math.log(1 + annual_margin) if annual_margin is not None else 0
        log_pe = math.log(1 + annual_pe) if annual_pe is not None else 0
        log_px_last = math.log(1 + annual_px_last) if annual_px_last is not None else 0
        log_total = math.log(1 + annual_total) if annual_total is not None else 0
        
        # For dividends, use difference between total return and price
        log_dividends = (annual_total - annual_px_last) if (annual_total is not None and annual_px_last is not None) else 0

        # Calculate ratio for rebasing
        ratio = (annual_total / log_total) if (log_total != 0) else 0

        # Calculate rebased annualized return for each component
        rebased_sales = log_sales * ratio
        rebased_margin = log_margin * ratio
        rebased_pe = log_pe * ratio
        rebased_dividends = log_dividends * ratio

        # Store rebased values for this index
        comp_rebased = {
            "TRAIL_12M_SALES_PER_SH": rebased_sales,
            "TRAIL_12M_PROF_MARGIN": rebased_margin,
            "INDX_ADJ_PE": rebased_pe,
            "DIVIDENDS": rebased_dividends
        }
        data_attr[label] = comp_rebased

    # Create stacked bar chart
    fig = go.Figure()
    x_indices = [entry["name"] for entry in index_list]
    components = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "DIVIDENDS"]
    
    # Add trace for each component
    for comp in components:
        y_vals = []
        for idx in x_indices:
            if data_attr.get(idx):
                y_vals.append(data_attr[idx][comp] * 100)  # Convert to percentage
            else:
                y_vals.append(0)
                
        fig.add_trace(go.Bar(
            x=x_indices,
            y=y_vals,
            name=labels.get(comp, comp),
            marker_color=colors.get(comp, "gray")
        ))
    
    # Get chart layout settings
    common_layout = config["charts"]["layout"]["common"]
    default_title = config["charts"]["financial"]["default_titles"]["stacked_bar_country"].format(country=selected_country)
    
    # Update layout
    fig.update_layout(
        barmode="stack",
        title=default_title,
        xaxis_title="Index",
        yaxis_title="Rebased Annualized Return (%)",
        plot_bgcolor=common_layout["plot_bgcolor"],
        legend=dict(
            orientation=common_layout["legend_position"]["orientation"],
            yanchor=common_layout["legend_position"]["yanchor"],
            y=common_layout["legend_position"]["y"],
            xanchor=common_layout["legend_position"]["xanchor"],
            x=common_layout["legend_position"]["x"]
        )
    )
    
    return dcc.Graph(figure=fig)


def make_total_return_for_country(selected_country):
    """
    Generate a table with total return decomposition for the selected country.
    
    Args:
        selected_country: Country to analyze
        
    Returns:
        dash_table.DataTable: Data table with return decomposition
    """
    from dash import dash_table, html
    from .utils import cagr, growth, format_value
    import math
    
    skip_rows = config["file_paths"]["bloomberg_skiprows"]
    mapping = config["countries"]["mapping"].get(selected_country, {})
    sheet_name = mapping.get("sheet_name", "")
    
    if not sheet_name:
        return html.Div(f"No sheet found for {selected_country}")

    try:
        df_bbg = pd.read_excel(config["file_paths"]["bloomberg"], sheet_name=sheet_name, skiprows=skip_rows)
    except Exception as e:
        return html.Div(f"Error reading data for {selected_country}: {str(e)}")

    if df_bbg.empty:
        return html.Div(f"No data in sheet '{sheet_name}' for {selected_country}")

    date_col = df_bbg.columns[0]
    
    # Define columns in the desired order
    col_list = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "DIVIDENDS", "PX_LAST", "TOTAL_RETURN"]
    
    # Check for missing columns
    missing = [c for c in col_list if c not in df_bbg.columns and c != "DIVIDENDS"]
    if missing:
        return html.Div(f"Missing columns in {sheet_name}: {missing}")

    # Filter and clean data
    df_bbg = df_bbg[[date_col] + [c for c in col_list if c != "DIVIDENDS"]].dropna(subset=[date_col])
    if len(df_bbg) < 2:
        return html.Div("Not enough data to compute decomposition.")

    first_row = df_bbg.iloc[0]
    last_row = df_bbg.iloc[-1]

    # Calculate time period
    if isinstance(first_row[date_col], pd.Timestamp) and isinstance(last_row[date_col], pd.Timestamp):
        days = (last_row[date_col] - first_row[date_col]).days
    else:
        days = 0
    
    # Get year days constant from config
    year_days = config["data_processing"]["calculations"]["year_days"]
    year_frac = days / year_days if days > 0 else 1

    # Calculate growth and CAGR for original columns
    growth_dic = {}
    annual_dic = {}
    for c in ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "PX_LAST", "TOTAL_RETURN"]:
        v0 = first_row[c]
        v1 = last_row[c]
        if c == "INDX_DIVISOR":
            growth_dic[c] = -growth(v0, v1)
            annual_dic[c] = -cagr(v0, v1, year_frac)
        else:
            growth_dic[c] = growth(v0, v1)
            annual_dic[c] = cagr(v0, v1, year_frac)
    
    # Calculate DIVIDENDS column
    growth_div = None
    annual_div = None
    if first_row["PX_LAST"] and first_row["TOTAL_RETURN"]:
        growth_div = growth(first_row["TOTAL_RETURN"], last_row["TOTAL_RETURN"]) - growth(first_row["PX_LAST"], last_row["PX_LAST"]) 
        annual_div = (1+cagr(first_row["TOTAL_RETURN"], last_row["TOTAL_RETURN"], year_frac)) / (1+cagr(first_row["PX_LAST"], last_row["PX_LAST"], year_frac)) - 1 
        
    growth_dic["DIVIDENDS"] = growth_div
    annual_dic["DIVIDENDS"] = annual_div

    # Calculate Return Attribution Annualized
    cols_for_attribution = ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "DIVIDENDS"]
    sum_annual_attr = sum(annual_dic[c] for c in cols_for_attribution if annual_dic[c] is not None)
    
    # Get minimum denominator value from config
    min_denom = config["data_processing"]["calculations"]["minimum_cagr_denom"]
    if abs(sum_annual_attr) < min_denom:
        sum_annual_attr = min_denom

    return_attr = {}
    return_attr_norm = {}
    for c in cols_for_attribution:
        val = annual_dic[c]
        if val is not None:
            return_attr[c] = (val * annual_dic["TOTAL_RETURN"]) / sum_annual_attr
            return_attr_norm[c] = return_attr[c] / annual_dic["TOTAL_RETURN"] if annual_dic["TOTAL_RETURN"] else None
        else:
            return_attr[c] = None
            return_attr_norm[c] = None

    def fmt_val(x, pct=False):
        """Format a value for display."""
        if x is None:
            return "N/A"
        try:
            if pct:
                return f"{x * 100:.1f}%"
            else:
                return f"{x:.2f}"
        except:
            return str(x)

    # Prepare table rows
    rows_data = []
    
    # Initial date row
    r1 = {"Label": f"Initial date: {str(df_bbg.iloc[0][date_col])[:10]}"}
    
    # Final date row
    r2 = {"Label": f"Final date: {str(df_bbg.iloc[-1][date_col])[:10]}"}
    
    # Growth percentage row
    r3 = {"Label": "Growth (%)"}
    
    # Annual Average (CAGR) row
    r4 = {"Label": "Annual Avg (CAGR)"}
    
    # Log Return row
    r5 = {"Label": "Log Return (Annual Avg)"}
    
    # Rebased Annualized Return row
    r6 = {"Label": "Rebased Annualized Return"}

    # Calculate log returns
    log_row = {}
    for c in col_list:
        if c in ["TRAIL_12M_SALES_PER_SH", "TRAIL_12M_PROF_MARGIN", "INDX_ADJ_PE", "PX_LAST", "TOTAL_RETURN"]:
            val = annual_dic[c]
            log_row[c] = math.log(1 + val) if val is not None else None
        elif c == "DIVIDENDS":
            # For DIVIDENDS, log return is the simple difference
            if annual_dic["TOTAL_RETURN"] is not None and annual_dic["PX_LAST"] is not None:
                log_row[c] = annual_dic["TOTAL_RETURN"] - annual_dic["PX_LAST"]
            else:
                log_row[c] = None

    # Calculate ratio for rebasing
    total_log = math.log(1 + annual_dic["TOTAL_RETURN"]) if annual_dic["TOTAL_RETURN"] is not None else None
    ratio = (annual_dic["TOTAL_RETURN"] / total_log) if (total_log not in [None, 0]) else None

    # Calculate rebased annualized return
    rebased_row = {}
    for c in col_list:
        if log_row.get(c) is not None and ratio is not None:
            rebased_row[c] = log_row[c] * ratio
        else:
            rebased_row[c] = None

    # Populate table rows
    for c in col_list:
        r1[c] = fmt_val(first_row[c]) if c != "DIVIDENDS" else ""
        r2[c] = fmt_val(last_row[c]) if c != "DIVIDENDS" else ""
        r3[c] = fmt_val(growth_dic[c], pct=True)
        r4[c] = fmt_val(annual_dic[c], pct=True)
        r5[c] = fmt_val(log_row[c], pct=True)
        r6[c] = fmt_val(rebased_row[c], pct=True)
    
    rows_data.extend([r1, r2, r3, r4, r5, r6])
    
    # Define table columns
    columns = [{"name": str("Label"), "id": str("Label")}] + [{"name": str(c), "id": str(c)} for c in col_list]
    
    # Get table styles from config
    table_styles = config["tables"]["return_decomposition_table"]["styles"]
    
    # Create data table
    dt = dash_table.DataTable(
        data=rows_data,
        columns=columns,
        style_cell=table_styles["cell"],
        style_header=table_styles["header"],
        style_table=table_styles["table"],
        page_action="none"
    )
    
    return dt




