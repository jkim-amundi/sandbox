"""
Ristrutturazione del file callback_plots.py per rendere le funzioni accessibili a livello di modulo
"""

"""
Plot and Visualization Callbacks Module

This module contains all callbacks related to generating and updating plots:
- Polar plot
- Bar plot
- Financial charts 
- Multiple box plot
- Stacked bar charts for total return decomposition
"""

import dash
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import os
from datetime import datetime
import dash_table

# Import configuration and utilities
from ..config_and_data import (
    config,
    df_variables, 
    metric_colors, 
    metric_labels,
    get_dashboard_file_path,
    valid_fund,
    valid_risk,
    valid_val
)

from ..utils import lighten_color, order_dataframe_by_country
from ..layout import make_stacked_bar_for_all_indices, make_stacked_bar_for_country, make_total_return_for_country

# Definire le funzioni a livello di modulo anziché all'interno di register_plot_callbacks

def create_polar_plot(selected_country, selected_metric):
    """
    Generate polar plot visualization based on selected country and metric.
    
    Returns:
        dict: Plotly figure object
    """
    if not selected_country or selected_country == "All":
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_country_selected"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    dashboard_file = get_dashboard_file_path()
    
    if not os.path.exists(dashboard_file):
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["dashboard_file_not_found"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    df_dash = pd.read_excel(dashboard_file, sheet_name="Dashboard")
    df_country = df_dash[df_dash["Country"] == selected_country]
    
    if df_country.empty:
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_data_for_country"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    # Exclude certain columns and melt the dataframe for plotting
    exclude_cols = config["data_processing"]["excluded_columns"]
    df_plot = df_country.drop(columns=exclude_cols, errors="ignore")
    df_melt = df_plot.melt(var_name="Variable", value_name="Value")
    
    if selected_metric and selected_metric != "All":
        valid_vars = df_variables[df_variables["F/R/V"] == selected_metric]["What"].unique().tolist()
        df_melt = df_melt[df_melt["Variable"].isin(valid_vars)]
    
    if df_melt.empty:
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_valid_data"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    df_merged = df_melt.merge(
        df_variables[["What", "F/R/V"]],
        left_on="Variable",
        right_on="What",
        how="left"
    )
    
    # Get polar chart settings
    polar_config = config["charts"]["layout"]["polar"]
    
    unique_vars = df_merged["Variable"].unique().tolist()
    n_vars = len(unique_vars)
    angle_map = {var: i * (360 / n_vars) for i, var in enumerate(unique_vars)}
    bar_width = polar_config["bar_width_factor"] * (360 / n_vars)
    
    fig = go.Figure()
    for cat in df_merged["F/R/V"].dropna().unique():
        sub = df_merged[df_merged["F/R/V"] == cat]
        if sub.empty:
            continue
        r_values = []
        theta_values = []
        hover_text = []
        for _, row in sub.iterrows():
            r_values.append(row["Value"])
            theta_values.append(angle_map[row["Variable"]])
            hover_text.append(f"{row['Variable']}: {row['Value']}")
        fig.add_trace(go.Barpolar(
            r=r_values,
            theta=theta_values,
            width=[bar_width] * len(r_values),
            name=metric_labels.get(cat, cat),
            marker_color=metric_colors.get(cat, "gray"),
            marker_line_color=polar_config["marker_line"]["color"],
            marker_line_width=polar_config["marker_line"]["width"],
            opacity=polar_config["opacity"],
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>"
        ))
    
    #setting up he axis labels
    #angular position where label must be places
    #text the variable name

    tickvals = [angle_map[var] for var in unique_vars]
    ticktext = unique_vars
    
    # Use the Polar layout settings from the config
    default_title = polar_config["default_title"].format(country=selected_country)
    common_layout = config["charts"]["layout"]["common"]
    
    #appearance of the chart
    fig.update_layout(
        title=default_title,
        showlegend=False,
        polar=dict(
            radialaxis=polar_config["radialaxis"],
            angularaxis=dict(
                showticklabels=polar_config["angularaxis"]["showticklabels"],
                ticks=polar_config["angularaxis"]["ticks"],
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                rotation=polar_config["angularaxis"]["rotation"],
                direction=polar_config["angularaxis"]["direction"]
            )
        ),
        template=None,
        plot_bgcolor=common_layout["plot_bgcolor"],
        paper_bgcolor=common_layout["paper_bgcolor"]
    )
    
    return fig

def create_bar_plot(selected_country, scenario):
    """
    Generate bar plot visualization comparing country scores with averages.
    
    Returns:
        dict: Plotly figure object
    """
    if not selected_country or selected_country == "All":
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_country_selected"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    dashboard_file = get_dashboard_file_path()
    
    if not os.path.exists(dashboard_file):
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["dashboard_file_not_found"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    df_dash = pd.read_excel(dashboard_file, sheet_name="Dashboard")
    df_country = df_dash[df_dash["Country"] == selected_country]
    
    # Handle empty dataframe case
    if df_country.empty:
        selected_values = {"Fundamental Score": 0, "Risk Score": 0, "Valuation Score": 0, "TotalScore": 0}
    else:
        row = df_country.iloc[0]
        fundamental = row.get("Fundamental Score", 0)
        risk = row.get("Risk Score", 0)
        valuation = row.get("Valuation Score", 0)
        
        # Apply scenario weights if not standard
        scenario_weights = config["scenarios"]["weights"]
        if scenario not in scenario_weights:
            scenario = "standard"
            
        weights = scenario_weights[scenario]
        
        if scenario == "standard":
            selected_values = {
                "Fundamental Score": fundamental, 
                "Risk Score": risk,
                "Valuation Score": valuation, 
                "TotalScore": row.get("TotalScore", 0)
            }
        else:
            # Calculate weighted total based on scenario
            total = (fundamental * (weights["fundamental"] / 100) +
                     risk * (weights["risk"] / 100) +
                     valuation * (weights["valuation"] / 100))
            selected_values = {
                "Fundamental Score": fundamental, 
                "Risk Score": risk,
                "Valuation Score": valuation, 
                "TotalScore": total
            }
    
    # Calculate averages
    if scenario == "standard":
        averages = {
            "Fundamental Score": df_dash["Fundamental Score"].mean(),
            "Risk Score": df_dash["Risk Score"].mean(),
            "Valuation Score": df_dash["Valuation Score"].mean(),
            "TotalScore": df_dash["TotalScore"].mean()
        }
    else:
        # Calculate weighted averages based on scenario
        weights = scenario_weights[scenario]
        averages = {
            "Fundamental Score": df_dash["Fundamental Score"].mean(),
            "Risk Score": df_dash["Risk Score"].mean(),
            "Valuation Score": df_dash["Valuation Score"].mean(),
            "TotalScore": df_dash.apply(
                lambda r: r["Fundamental Score"] * (weights["fundamental"] / 100) +
                          r["Risk Score"] * (weights["risk"] / 100) +
                          r["Valuation Score"] * (weights["valuation"] / 100), axis=1
            ).mean()
        }
    
    # Get bar chart configuration
    bar_config = config["charts"]["layout"]["bar"]
    bar_colors = bar_config["colors"]
    
    # Create the selected country bars
    trace_selected = go.Bar(
        x=list(selected_values.keys()),
        y=[selected_values[m] for m in selected_values],
        name=f"{selected_country}",
        marker_color=[bar_colors[m] for m in selected_values]
    )
    
    # Create the average bars with lightened colors and pattern
    trace_average = go.Bar(
        x=list(averages.keys()),
        y=[averages[m] for m in averages],
        name="Average",
        marker=dict(
            color=[lighten_color(bar_colors[m], bar_config["lighten_factor"]) for m in averages],
            pattern=dict(
                shape=bar_config["average_pattern"]["shape"]
            )
        )
    )
    
    # Create the figure
    fig = go.Figure(data=[trace_selected, trace_average])
    
    # Get common layout settings
    common_layout = config["charts"]["layout"]["common"]
    legend_position = common_layout["legend_position"]
    
    # Use Bar plot layout settings from config
    default_title = bar_config["default_title"].format(country=selected_country)
    fig.update_layout(
        barmode="group", 
        title=default_title,
        showlegend=False,
        legend=dict(
            orientation=legend_position["orientation"],
            yanchor=legend_position["yanchor"],
            y=legend_position["y"],
            xanchor=legend_position["xanchor"],
            x=legend_position["x"]
        ),
        plot_bgcolor=common_layout["plot_bgcolor"],
        paper_bgcolor=common_layout["paper_bgcolor"]
    )
    
    return fig

def create_financial_chart(selected_country):
    """
    Generate financial chart comparing country index with MXEF index.
    
    Returns:
        dict: Plotly figure object
    """
    if not selected_country or selected_country == "All":
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_country_selected"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    try:
        mapping = config["countries"]["mapping"].get(selected_country, {})
        sheet_name = mapping.get("sheet_name", "")
        if not sheet_name:
            return go.Figure(layout={'annotations': [{
                'text': config["messages"]["errors"]["msci_index_not_found"],
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]})
            
        df_country = pd.read_excel(
            config["file_paths"]["bloomberg"],
            sheet_name=sheet_name,
            skiprows=config["file_paths"]["bloomberg_skiprows"]
        )
        date_col = df_country.columns[0]
        if "PX_LAST" not in df_country.columns:
            return go.Figure(layout={'annotations': [{
                'text': config["messages"]["errors"]["missing_columns"],
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]})
            
        df_country = df_country[[date_col, "PX_LAST"]].dropna()
        df_country.columns = ["Date", "CountryPX"]

        df_mxef = pd.read_excel(
            config["file_paths"]["bloomberg"],
            sheet_name=config["charts"]["financial"]["default_indices"]["mxef_index"],
            skiprows=config["file_paths"]["bloomberg_skiprows"]
        )
        if "PX_LAST" not in df_mxef.columns:
            return go.Figure(layout={'annotations': [{
                'text': config["messages"]["errors"]["missing_mxef_index"],
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]})
            
        df_mxef = df_mxef[[date_col, "PX_LAST"]].dropna()
        df_mxef.columns = ["Date", "MXEF"]

        df_merged = pd.merge(df_mxef, df_country, on="Date", how="inner")
        
        # Base 100 normalization
        base_mxef = df_merged["MXEF"].iloc[0]
        if base_mxef != 0:
            df_merged["MXEF"] = df_merged["MXEF"] / base_mxef * 100
        
        base_country = df_merged["CountryPX"].iloc[0]
        if base_country != 0:
            df_merged["CountryPX"] = df_merged["CountryPX"] / base_country * 100

        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_merged["Date"],
            y=df_merged["MXEF"],
            mode="lines",
            name="MXEF Index",
            line=dict(color=config["app_settings"]["style"]["secondary_color"])
        ))
        fig.add_trace(go.Scatter(
            x=df_merged["Date"],
            y=df_merged["CountryPX"],
            mode="lines",
            name=selected_country,
            line=dict(color=config["app_settings"]["style"]["primary_color"])
        ))
        
        # Get common layout settings
        common_layout = config["charts"]["layout"]["common"]
        legend_position = common_layout["legend_position"]
        
        # Use financial chart title from config
        default_title = config["charts"]["financial"]["default_titles"]["index_comparison"].format(
            country=selected_country
        )
        
        fig.update_layout(
            title=default_title,
            xaxis_title="Date",
            yaxis_title="Base 100",
            autosize=True,
            paper_bgcolor=common_layout["paper_bgcolor"],
            plot_bgcolor=common_layout["plot_bgcolor"],
            legend=dict(
                orientation=legend_position["orientation"],
                yanchor=legend_position["yanchor"],
                y=legend_position["y"],
                xanchor=legend_position["xanchor"],
                x=legend_position["x"]
            )
        )
        
        return fig
        
    except Exception as e:
        return go.Figure(layout={'annotations': [{
            'text': f"{config['messages']['errors']['read_financial_data_error']} {str(e)}",
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})

def create_multiple_box_plot(selected_country):
    """
    Generate multiple box plots for different metrics of the selected country.
    
    Returns:
        dict: Plotly figure object
    """
    if not selected_country or selected_country == "All":
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_country_selected"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
        
    try:
        mapping = config["countries"]["mapping"].get(selected_country, {})
        sheet_name = mapping.get("sheet_name", "")
        if not sheet_name:
            return go.Figure(layout={'annotations': [{
                'text': config["messages"]["errors"]["msci_index_not_found"],
                'xref': 'paper', 'yref': 'paper',
                'showarrow': False, 'font': {'size': 16}
            }]})
            
        df_bbg = pd.read_excel(
            config["file_paths"]["bloomberg"],
            sheet_name=sheet_name,
            skiprows=config["file_paths"]["bloomberg_skiprows"]
        )
    except Exception as e:
        return go.Figure(layout={'annotations': [{
            'text': f"{config['messages']['errors']['read_financial_data_error']} {str(e)}",
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
        
    # Get columns and colors configuration for box plots
    box_plot_config = config["charts"]["box_plot"]
    multi_cols = list(box_plot_config["labels"].keys())
    colors = box_plot_config["colors"]

    # Filter columns with valid data
    valid_cols = []
    for col in multi_cols:
        if col in df_bbg.columns and not df_bbg[col].dropna().empty:
            valid_cols.append(col)
            
    if not valid_cols:
        return go.Figure(layout={'annotations': [{
            'text': config["messages"]["errors"]["no_valid_data"],
            'xref': 'paper', 'yref': 'paper',
            'showarrow': False, 'font': {'size': 16}
        }]})
    
    # Create a subplot with one row and multiple columns
    fig = make_subplots(rows=1, cols=len(valid_cols), shared_yaxes=False)

    # Get actual marker settings
    actual_marker = box_plot_config["actual_marker"]

    # Add box plots for each metric
    for i, col in enumerate(valid_cols):
        series_data = df_bbg[col].dropna()
        trace_name = box_plot_config["labels"].get(col, col)
        
        # Add box plot
        fig.add_trace(
            go.Box(
                x=[trace_name]*len(series_data),
                y=series_data,
                name=trace_name,
                boxmean='sd',
                boxpoints="outliers",
                whiskerwidth=0.3,
                marker_color=colors.get(col, "gray"),
                fillcolor=colors.get(col, "gray"),
                line=dict(color=colors.get(col, "gray"))
            ),
            row=1, col=i+1
        )
        
        # Add scatter for the latest value
        last_val = series_data.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[trace_name],
                y=[last_val],
                name=f"Actual {col}",
                mode='markers+text',
                text=[actual_marker["text"]],
                textposition="top center",
                marker=dict(
                    color=actual_marker["color"],
                    size=actual_marker["size"],
                    symbol=actual_marker["symbol"]
                )
            ),
            row=1, col=i+1
        )
        
        # Hide y-axis for this subplot
        fig.update_yaxes(visible=False, row=1, col=i+1)
    
    # Update layout
    fig.update_layout(
        title=f"Box Plots for {selected_country}",
        template='plotly_white',
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def register_plot_callbacks(app, cache):
    """
    Register all plot-related callbacks with the app.
    
    Args:
        app: The Dash application instance
        cache: The Flask-Cache instance
    """
    
    # Callback to update the Polar Plot
    @app.callback(
        Output("polar-plot", "figure"),
        [Input("btn-actual", "n_clicks"),
         Input("dropdown-country", "value"),
         Input("selected-metric", "data")]
    )
    @cache.memoize(timeout=config["app_settings"]["cache"]["thresholds"]["plots"])
    def update_polar_plot(btn_actual, selected_country, selected_metric):
        """Wrapper around create_polar_plot that adds caching"""
        return create_polar_plot(selected_country, selected_metric)
    
    # Callback to update the Bar Plot
    @app.callback(
        Output("bar-plot", "figure"),
        [Input("btn-actual", "n_clicks"),
         Input("dropdown-country", "value"),
         Input("selected-scenario", "data")]
    )
    @cache.memoize(timeout=config["app_settings"]["cache"]["thresholds"]["plots"])
    def update_bar_plot(btn_actual, selected_country, scenario):
        """Wrapper around create_bar_plot that adds caching"""
        return create_bar_plot(selected_country, scenario)
    
    # Callback to update the Financial Chart
    @app.callback(
        Output("financial-chart", "figure"),
        [Input("dropdown-country", "value")]
    )
    @cache.memoize(timeout=config["app_settings"]["cache"]["thresholds"]["plots"])
    def update_financial_chart(selected_country):
        """Wrapper around create_financial_chart that adds caching"""
        return create_financial_chart(selected_country)

    # Callback to update the Multiple Box Plot
    @app.callback(
        Output("multiple-box-plot", "figure"),
        [Input("dropdown-country", "value")]
    )
    @cache.memoize(timeout=config["app_settings"]["cache"]["thresholds"]["plots"])
    def update_multiple_box_plot(selected_country):
        """Wrapper around create_multiple_box_plot that adds caching"""
        return create_multiple_box_plot(selected_country)
        
    # Callback to update the Financial Data container
    @app.callback(
        Output("financial-data-container", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_financial_data_container(selected_country):
        """
        Update the financial data container with appropriate charts.
        For "All", show regional indices. For specific country, show financial chart and box plot.
        
        Returns:
            list: List of Dash components
        """
        try:
            # If "All" is selected, show regional charts
            if not selected_country or selected_country == "All":
                index_list = config["charts"]["financial"]["indices"]
                figures = []
                
                for item in index_list:
                    region = item["name"]
                    sheet_name = item["sheet_name"]
                    color = item["color"]

                    # Load region data
                    df_region = pd.read_excel(
                        config["file_paths"]["bloomberg"],
                        sheet_name=sheet_name,
                        skiprows=config["file_paths"]["bloomberg_skiprows"]
                    )
                    date_col = df_region.columns[0]
                    df_region = df_region[[date_col, "PX_LAST"]].dropna()
                    df_region.columns = ["Date", region]

                    # Load MXEF data
                    mxef_sheet = config["charts"]["financial"]["default_indices"]["mxef_index"]
                    df_mxef = pd.read_excel(
                        config["file_paths"]["bloomberg"],
                        sheet_name=mxef_sheet,
                        skiprows=config["file_paths"]["bloomberg_skiprows"]
                    )
                    df_mxef = df_mxef[[date_col, "PX_LAST"]].dropna()
                    df_mxef.columns = ["Date", "MXEF Index"]

                    # Merge and normalize to base 100
                    df_merged = pd.merge(df_mxef, df_region, on="Date", how="inner")
                    base_mxef = df_merged["MXEF Index"].iloc[0]
                    df_merged["MXEF Index"] = df_merged["MXEF Index"] / base_mxef * 100 if base_mxef != 0 else df_merged["MXEF Index"]

                    base_region = df_merged[region].iloc[0]
                    df_merged[region] = df_merged[region] / base_region * 100 if base_region != 0 else df_merged[region]

                    # Get common layout settings
                    common_layout = config["charts"]["layout"]["common"]
                    legend_position = common_layout["legend_position"]

                    # Create figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_merged["Date"],
                        y=df_merged["MXEF Index"],
                        mode="lines",
                        name="MXEF Index",
                        line=dict(color=config["app_settings"]["style"]["secondary_color"])
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_merged["Date"],
                        y=df_merged[region],
                        mode="lines",
                        name=region,
                        line=dict(color=color)
                    ))
                    fig.update_layout(
                        title=region,
                        xaxis_title="Date",
                        yaxis_title="Base 100",
                        autosize=True,
                        paper_bgcolor=common_layout["paper_bgcolor"],
                        plot_bgcolor=common_layout["plot_bgcolor"],
                        legend=dict(
                            orientation=legend_position["orientation"],
                            yanchor=legend_position["yanchor"],
                            y=legend_position["y"],
                            xanchor=legend_position["xanchor"],
                            x=legend_position["x"]
                        )
                    )
                    figures.append(dcc.Graph(figure=fig))
                    
                return figures
            else:
                # For specific country, show financial chart and box plot
                return [
                    dcc.Graph(id="financial-chart"),
                    dcc.Graph(id="multiple-box-plot")
                ]
        except Exception as e:
            return html.Div(f"Error loading Bloomberg data: {str(e)}")
            
    # Callback to update the Total Return Decomposition section
    @app.callback(
        Output("total-return-decomposition", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_total_return_decomposition(selected_country):
        """
        Update the total return decomposition section.
        For "All", show stacked bar chart for global indices.
        For specific country, show return decomposition table and stacked bar chart.
        
        Returns:
            dash.html.Div: Dash component with return decomposition
        """
        if not selected_country or selected_country == "All":
            return make_stacked_bar_for_all_indices()
        else:
            return html.Div([
                make_total_return_for_country(selected_country),
                make_stacked_bar_for_country(selected_country)
            ])
            
    # Callback to update the Flow Analysis section
    @app.callback(
        Output("flow-analysis-section", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_flow_analysis_section(selected_country):
        """
        Update the flow analysis section visibility and content.
        Only shown when "All" is selected.
        
        Returns:
            dash.html.Div: Dash component with flow analysis
        """
        if selected_country == "All":
            header = html.H5("Flow Analysis", className="text-left")
            # The callback for "flow-analysis-table" will handle loading the table
            table = html.Div(id="flow-analysis-table")
            return html.Div([header, table])
        else:
            return ""

    # Callback to update the Flow Analysis table
    @app.callback(
        Output("flow-analysis-table", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_flow_analysis_table(selected_country):
        """
        Update the flow analysis table with Bloomberg flow data.
        Only shown when "All" is selected.
        
        Returns:
            dash.html.Div: Table component with flow data
        """
        if selected_country != "All":
            return ""
            
        try:
            df_flow = pd.read_excel(
                config["file_paths"]["bloomberg"], 
                sheet_name="Flows",
                skiprows=config["file_paths"]["bloomberg_skiprows"]
            )
            columns = [{"name": str(col), "id": str(col)} for col in df_flow.columns]
            table = dash_table.DataTable(
                data=df_flow.to_dict("records"),
                columns=columns,
                page_size=10,
                style_table={'overflowX': 'auto'},
            )
            return table
        except Exception as e:
            return html.Div(f"Error reading Bloomberg flow data: {str(e)}")