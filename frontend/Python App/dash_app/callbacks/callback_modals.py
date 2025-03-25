"""
Modal Dialog Callbacks Module

This module contains all callbacks related to modal dialogs:
- Metric selection modal
- Scenario selection modal
- Extract data modal
- Content and sidebar visibility management
"""

import dash
from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate

# Import configuration and data
from ..config_and_data import (
    config, 
    df_index,
    institute_view_text, 
    frv_text,
    get_dashboard_file_path
)

from ..layout import (
    actual_content, 
    historical_content, 
    about_content
)
import pandas as pd
# Import utility functions
from ..utils import create_score_display


def register_modal_callbacks(app, cache):
    """
    Register all modal-related callbacks with the app.
    
    Args:
        app: The Dash application instance
        cache: The Flask-Cache instance
    """
    
    # Callback to toggle visibility of selection row based on active view
    @app.callback(
        Output("selection-row", "style"),
        [Input("btn-actual", "n_clicks"),
         Input("btn-historical", "n_clicks"),
         Input("btn-about", "n_clicks")]
    )
    def toggle_selection_row(actual_clicks, historical_clicks, about_clicks):
        """
        Hide/show the selection row (metric, scenario, extract) based on active view.
        """
        ctx = dash.callback_context
        
        # If no trigger (first execution), assume we're in Actual view
        if not ctx.triggered:
            return {}
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Show row only in Actual view
        if triggered_id == "btn-actual":
            return {}
        else:
            return {"display": "none"}
    
    # Callback to update content on tab change and handle sidebar visibility
    @app.callback(
        [Output("content", "children"),
         Output("country-weight", "children"),
         Output("country-sectors", "children"),
         Output("country-scores", "children"),
         Output("right-sidebar", "style")],
        [Input("btn-actual", "n_clicks"),
         Input("btn-historical", "n_clicks"),
         Input("btn-about", "n_clicks"),
         Input("dropdown-country", "value"),
         Input("scenario-info-btn", "n_clicks")]
    )
    def update_content(btn_actual, btn_historical, btn_about, selected_country, scenario_info):
        """
        Update the main content area based on the selected navigation tab.
        Hides the sidebar when on About or Historical tabs.
        
        Returns:
            tuple: (content, country_weight, country_sectors, country_scores, sidebar_style)
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            return actual_content, "", "", "", {}

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # For About, Historical or Info tabs, hide the sidebar
        if triggered_id in ["btn-about", "btn-historical", "scenario-info-btn"]:
            if triggered_id in ["btn-about", "scenario-info-btn"]:
                # When About button is pressed, load content directly here
                about_with_md = about_content
                # Update markdown content directly
                for child in about_with_md.children:
                    if hasattr(child, 'id') and child.id == 'institute-view-content':
                        child.children = institute_view_text
                return about_with_md, "", "", "", {"display": "none"}
            elif triggered_id == "btn-historical":
                # When Historical button is pressed, load content directly here
                hist_with_md = historical_content
                # Update markdown content directly
                for child in hist_with_md.children:
                    if hasattr(child, 'id') and child.id == 'frv-content':
                        child.children = frv_text
                return hist_with_md, "", "", "", {"display": "none"}
        
        # Handle Actual tab or country dropdown change
        if triggered_id in ["btn-actual", "dropdown-country"]:
            if selected_country:
                if selected_country == "All":
                    # For "All", don't show weight, stocks and scores
                    country_weight = ""
                    country_sectors = ""
                    country_scores = ""
                else:
                    # Get weight and stocks from index data
                    row_idx = df_index[df_index["Country"] == selected_country]
                    if not row_idx.empty:
                        weight_value = row_idx.iloc[0].get("Weight")
                        if weight_value is not None:
                            weight_value = weight_value * 100
                        stocks_value = row_idx.iloc[0].get("Number of stocks")
                        country_weight = f"Weight of {selected_country}: {weight_value:.1f}%" if weight_value is not None else "Weight: N/A"
                        country_sectors = f"Number of stocks: {stocks_value:.0f}" if stocks_value is not None else "Number of stocks: N/A"
                    else:
                        country_weight = "Weight of the Country: N/A"
                        country_sectors = "Number of stocks: N/A"
                    
                    # Read dashboard for scores
                    try:
                        dashboard_file = get_dashboard_file_path()
                        df_dash = pd.read_excel(dashboard_file, sheet_name="Dashboard")
                        df_country_dash = df_dash[df_dash["Country"] == selected_country]
                        if not df_country_dash.empty:
                            row_dash = df_country_dash.iloc[0]
                            score_total = row_dash.get("TotalScore")
                            score_val = row_dash.get("Valuation Score")
                            score_risk = row_dash.get("Risk Score")
                            score_fund = row_dash.get("Fundamental Score")
                        else:
                            score_total = score_val = score_risk = score_fund = None
                    except Exception as e:
                        score_total = score_val = score_risk = score_fund = None

                    # Create score components with color coding
                    p_total = create_score_display("Total Score", score_total)
                    p_val = create_score_display("Valuation Score", score_val)
                    p_risk = create_score_display("Risk Score", score_risk)
                    p_fund = create_score_display("Fundamental Score", score_fund)
                    country_scores = html.Div([p_total, p_val, p_risk, p_fund])
            else:
                country_weight = ""
                country_sectors = ""
                country_scores = ""
            return actual_content, country_weight, country_sectors, country_scores, {}
        
        # Default return 
        return actual_content, "", "", "", {}

    # Callback to toggle visibility of charts based on country selection
    @app.callback(
        [Output("polar-plot-col", "style"),
         Output("bar-plot-col", "style")],
        [Input("dropdown-country", "value")]
    )
    def hide_plots_if_all(selected_country):
        """
        Hide the polar and bar plots when 'All' is selected.
        """
        if not selected_country or selected_country == "All":
            return ({"display": "none"}, {"display": "none"})
        else:
            return ({}, {})

    # Callback to update accordion visibility in the sidebar
    @app.callback(
        [Output("country-data-accordion", "style"),
         Output("economic-data-accordion", "style"),
         Output("forecast-accordion", "style")],
        [Input("dropdown-country", "value")]
    )
    def update_accordion_visibility(selected_country):
        """
        Hide country data accordions when 'All' is selected.
        """
        if selected_country == "All":
            return [{"display": "none"}, {"display": "none"}, {"display": "none"}]
        else:
            return [{}, {}, {}]
    
    # Callback for Metric selection modal
    @app.callback(
        [Output("selected-metric", "data"),
         Output("metric-modal", "is_open")],
        [Input("metric-link", "n_clicks"),
         Input("close-metric-modal", "n_clicks"),
         Input("metric-option-all", "n_clicks"),
         Input("metric-option-fundamentals", "n_clicks"),
         Input("metric-option-risks", "n_clicks"),
         Input("metric-option-valuations", "n_clicks"),
         Input("dropdown-country", "value"),
         Input("btn-about", "n_clicks"),
         Input("btn-historical", "n_clicks")],
        [State("selected-metric", "data"),
         State("metric-modal", "is_open")]
    )
    def update_metric_modal(n_link, n_close, n_all, n_fund, n_risks, n_val, 
                        country_value, about_clicks, historical_clicks, 
                        current_metric, is_open):
        """
        Handle the metric selection modal interactions.
        """
        ctx = dash.callback_context
        
        # If no trigger (first execution), don't update
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Close modal when switching views
        if triggered_id in ["btn-about", "btn-historical"]:
            return current_metric, False
        
        # When country changes, reset to "All" but keep modal closed
        if triggered_id == "dropdown-country":
            return "All", False
        
        # Only explicit click on metric button should open modal
        if triggered_id == "metric-link" and n_link is not None and n_link > 0:
            return current_metric, True
        
        # Close modal and update value when option selected
        if triggered_id == "close-metric-modal":
            return current_metric, False
        if triggered_id == "metric-option-all":
            return "All", False
        if triggered_id == "metric-option-fundamentals":
            return "Fundamentals", False
        if triggered_id == "metric-option-risks":
            return "Risks", False
        if triggered_id == "metric-option-valuations":
            return "Valuations", False
        
        # For all other cases, don't update
        raise PreventUpdate

    # Callback for Scenario selection modal
    @app.callback(
        [Output("selected-scenario", "data"),
         Output("scenario-modal", "is_open")],
        [Input("scenario-link", "n_clicks"),
         Input("close-scenario-modal", "n_clicks")] +
        [Input(f"scenario-option-{scenario['id']}", "n_clicks") for scenario in config["scenarios"]["list"]] +
        [Input("btn-about", "n_clicks"),
         Input("btn-historical", "n_clicks"),
         Input("dropdown-country", "value")],
        [State("selected-scenario", "data"),
         State("scenario-modal", "is_open")]
    )
    def update_scenario_modal(*args):
        """
        Handle the scenario selection modal interactions.
        """
        # Extract arguments
        num_options = len(config["scenarios"]["list"])
        n_link, n_close = args[0:2]
        option_clicks = args[2:2+num_options]
        about_clicks, historical_clicks, country_value = args[2+num_options:5+num_options]
        current_scenario, is_open = args[-2:]
        
        ctx = dash.callback_context
        
        # If no trigger (first execution), don't update
        if not ctx.triggered:
            raise PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Close modal when switching views or changing country
        if triggered_id in ["btn-about", "btn-historical", "dropdown-country"]:
            return current_scenario, False
        
        # Only explicit click on scenario button should open modal
        if triggered_id == "scenario-link" and n_link is not None and n_link > 0:
            return current_scenario, True
        
        # Close modal when Close button clicked
        if triggered_id == "close-scenario-modal":
            return current_scenario, False
        
        # Handle scenario options
        scenario_ids = [scenario["id"] for scenario in config["scenarios"]["list"]]
        for i, scenario_id in enumerate(scenario_ids):
            if triggered_id == f"scenario-option-{scenario_id}" and option_clicks[i] is not None and option_clicks[i] > 0:
                return scenario_id, False
        
        # For all other triggers, don't update
        raise PreventUpdate

    # Callback to update the Metric link button text
    @app.callback(
        Output("metric-link", "children"),
        [Input("selected-metric", "data")]
    )
    def update_metric_link(selected_metric):
        """Update the metric link button text to show the selected metric."""
        return selected_metric

    # Callback to update the Scenario link button text
    @app.callback(
        Output("scenario-link", "children"),
        [Input("selected-scenario", "data")]
    )
    def update_scenario_link(selected_scenario):
        """Update the scenario link button text to show the selected scenario."""
        # Find the scenario name from the ID
        for scenario in config["scenarios"]["list"]:
            if scenario["id"] == selected_scenario:
                return scenario["name"]
                
        # Default fallback
        return "Standard" if not selected_scenario else selected_scenario.capitalize()

    # Callback to toggle the Extract Data modal
    @app.callback(
        Output("extract-modal", "is_open"),
        [Input("extract-data-button", "n_clicks"), 
         Input("close-extract-modal", "n_clicks"),
         Input("btn-about", "n_clicks"),
         Input("btn-historical", "n_clicks")],
        [State("extract-modal", "is_open")]
    )
    def toggle_extract_modal(n_open, n_close, about_clicks, historical_clicks, is_open):
        """Toggle the extract data modal visibility."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Close modal when switching views
        if triggered_id in ["btn-about", "btn-historical"]:
            return False
            
        # Open/close modal only when needed
        if triggered_id == "extract-data-button" and n_open:
            return True
        elif triggered_id == "close-extract-modal" and n_close:
            return False
            
        # In all other cases, don't modify state
        raise PreventUpdate

    # Callback to update the country flag image
    @app.callback(
        Output("country-flag", "src"),
        [Input("dropdown-country", "value")]
    )
    def update_country_flag(selected_country):
        """Update the country flag image based on selected country."""
        if not selected_country or selected_country == "All":
            return ""
        mapping = config["countries"]["mapping"].get(selected_country, {})
        return mapping.get("flag_url", "")