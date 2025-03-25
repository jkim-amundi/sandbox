"""
Table and Data Display Callbacks Module

This module contains all callbacks related to tables and data displays:
- Dashboard table
- Economic data table
- Forecast data table
- Country data display
"""

import dash
from dash import Input, Output, html, dash_table
from dash.dependencies import State
from dash_table.Format import Format, Scheme
import pandas as pd
import os
from datetime import datetime
import dash_bootstrap_components as dbc
import dash_table

# Import configuration and utilities
from ..config_and_data import (
    config, 
    df_variables, 
    df_index,
    get_dashboard_file_path,
    valid_fund,
    valid_risk,
    valid_val
)

from ..utils import (
    apply_scenario_weights,
    order_dataframe_by_country,
    format_percentage,
    format_with_commas
)


def register_table_callbacks(app, cache):
    """
    Register all table-related callbacks with the app.
    
    Args:
        app: The Dash application instance
        cache: The Flask-Cache instance
    """
    
    # Callback to update the Dashboard table
    @app.callback(
        Output("dashboard-table-container", "children"),
        [Input("btn-actual", "n_clicks"),
         Input("dropdown-country", "value"),
         Input("selected-metric", "data"),
         Input("selected-scenario", "data")]
    )
    @cache.memoize(timeout=config["app_settings"]["cache"]["thresholds"]["tables"])
    def update_dashboard_table(btn_actual, selected_country, selected_metric, scenario):
        """
        Update the main dashboard table with filtered and formatted data.
        
        Returns:
            dash.html.Div: Table component
        """
        dashboard_file = get_dashboard_file_path()
        
        if not os.path.exists(dashboard_file):
            return html.Div(config["messages"]["errors"]["dashboard_file_not_found"])
        
        df_full = pd.read_excel(dashboard_file, sheet_name="Dashboard")
        df_small = df_full.copy()
        df_main = df_full.copy()
        
        # Use standard scenario if none specified
        if scenario is None:
            scenario = "standard"
        
        # Apply scenario weights if not standard
        if scenario != "standard":
            df_small = apply_scenario_weights(df_small, scenario)
            df_main = apply_scenario_weights(df_main, scenario)
        
        # Filter for specific metric if requested
        if selected_metric and selected_metric != "All":
            valid_vars = df_variables[df_variables["F/R/V"] == selected_metric]["What"].unique().tolist()
            columns_to_keep = ["Country"] + [col for col in df_main.columns if col in valid_vars]
            # Always include score columns
            for score in ["Fundamental Score", "Risk Score", "Valuation Score", "TotalScore"]:
                if score not in columns_to_keep:
                    columns_to_keep.append(score)
            df_main = df_main[columns_to_keep]
        
        # Prioritize selected country if specified
        if selected_country and selected_country != "All":
            df_main = order_dataframe_by_country(df_main, selected_country)
            df_small = order_dataframe_by_country(df_small, selected_country)
        
        # Reorder columns for better display
        desired_order = ["Country", "TotalScore", "Fundamental Score", "Valuation Score", "Risk Score"]
        new_order = [col for col in desired_order if col in df_main.columns]
        new_order += [col for col in df_main.columns if col not in new_order]
        df_main = df_main[new_order]
        
        # Get table styling configuration
        table_styles = config["tables"]["dashboard_table"]["styles"]
        quantile_thresholds = config["metrics"]["quantile_thresholds"]
        color_ranges = config["metrics"]["color_ranges"]
        
        # Define conditional styling for numeric data
        style_data_conditional = []
        numeric_cols = [c for c in df_main.columns if c != "Country" and pd.api.types.is_numeric_dtype(df_main[c])]
        
        for col in numeric_cols:
            lower_bound = df_main[col].quantile(quantile_thresholds["lower"])
            upper_bound = df_main[col].quantile(quantile_thresholds["upper"])
            
            # Style for low values
            style_data_conditional.append({
                'if': {'filter_query': f'{{{col}}} <= {lower_bound}', 'column_id': col},
                'backgroundColor': color_ranges["low"]["background"],
                'color': color_ranges["low"]["text"]
            })
            
            # Style for high values
            style_data_conditional.append({
                'if': {'filter_query': f'{{{col}}} >= {upper_bound}', 'column_id': col},
                'backgroundColor': color_ranges["high"]["background"],
                'color': color_ranges["high"]["text"]
            })
        
        # Style headers based on metric category
        header_styles = []
        for col in df_main.columns:
            # For aggregate score columns
            if col == config["metrics"]["mapping"].get("Fundamentals"):
                header_styles.append({
                    "if": {"column_id": col},
                    "backgroundColor": config["metrics"]["colors"]["Fundamentals"],
                    "color": "white"
                })
            elif col == config["metrics"]["mapping"].get("Risks"):
                header_styles.append({
                    "if": {"column_id": col},
                    "backgroundColor": config["metrics"]["colors"]["Risks"],
                    "color": "white"
                })
            elif col == config["metrics"]["mapping"].get("Valuations"):
                header_styles.append({
                    "if": {"column_id": col},
                    "backgroundColor": config["metrics"]["colors"]["Valuations"],
                    "color": "white"
                })
            else:
                # For individual metric columns
                if col in valid_fund:
                    header_styles.append({
                        "if": {"column_id": col},
                        "backgroundColor": config["metrics"]["colors"]["Fundamentals"],
                        "color": "white"
                    })
                elif col in valid_risk:
                    header_styles.append({
                        "if": {"column_id": col},
                        "backgroundColor": config["metrics"]["colors"]["Risks"],
                        "color": "white"
                    })
                elif col in valid_val:
                    header_styles.append({
                        "if": {"column_id": col},
                        "backgroundColor": config["metrics"]["colors"]["Valuations"],
                        "color": "white"
                    })
        
        # Configure columns for the DataTable
        main_columns = []
        for col in df_main.columns:
            if pd.api.types.is_numeric_dtype(df_main[col]):
                main_columns.append({
                    "name": str(col),
                    "id": col,
                    "type": "numeric",
                    "format": Format(precision=0, scheme=Scheme.fixed)
                })
            else:
                main_columns.append({"name": str(col), "id": str(col)})
        
        # Get table settings from config
        page_size = config["tables"]["dashboard_table"]["default_page_size"]
        
        # Create the main table component
        main_table = dash_table.DataTable(
            data=df_main.to_dict("records"),
            columns=main_columns,
            page_size=page_size,
            style_table=table_styles["table"],
            style_cell=table_styles["cell"],
            style_header=table_styles["header"],
            fixed_columns={'headers': True, 'data': 1},
            sort_action="native",
            filter_action="none",
            style_data_conditional=style_data_conditional,
            style_header_conditional=header_styles  
        )

        # For "All" country selection, show Leaders and Laggards tables
        if selected_country in (None, "All"):
            # Get summary table configuration
            summary_tables_config = config["tables"]["dashboard_table"]["summary_tables"]
            summary_styles = summary_tables_config["styles"]
            num_rows = summary_tables_config["num_rows"]
            
            base_cols = ["Country", "Fundamental Score", "Risk Score", "Valuation Score", "TotalScore"]
            for col in base_cols:
                if col not in df_small.columns:
                    df_small[col] = "N/A"
                    
            # Determine which column to use for ranking
            ranking_col = config["metrics"]["mapping"].get(selected_metric, "TotalScore") if selected_metric and selected_metric != "All" else "TotalScore"
            remaining_cols = [col for col in base_cols if col not in ("Country", ranking_col)]
            small_cols_order = ["Country", ranking_col] + remaining_cols
            
            # Configure columns for the small tables
            small_columns = []
            for col in small_cols_order:
                if pd.api.types.is_numeric_dtype(df_small[col]):
                    small_columns.append({
                        "name": str(col),
                        "id": col,
                        "type": "numeric",
                        "format": Format(precision=0, scheme=Scheme.fixed)
                    })
                else:
                    small_columns.append({"name": str(col), "id": str(col)})
            
            # Sort for leaders (high scores) and laggards (low scores)
            df_sorted_desc = df_small.sort_values(by=ranking_col, ascending=False)
            df_sorted_asc = df_small.sort_values(by=ranking_col, ascending=True)
            df_best = df_sorted_desc.head(num_rows)[small_cols_order]
            df_worst = df_sorted_asc.head(num_rows)[small_cols_order]
            
            # Create Leaders table
            best_table = dash_table.DataTable(
                data=df_best.to_dict("records"),
                columns=small_columns,
                style_table=summary_styles["table"],
                style_cell=summary_styles["cell"],
                style_header=summary_styles["header"],
                page_action="none",
                style_as_list_view=True
            )
            
            # Create Laggards table
            worst_table = dash_table.DataTable(
                data=df_worst.to_dict("records"),
                columns=small_columns,
                style_table=summary_styles["table"],
                style_cell=summary_styles["cell"],
                style_header=summary_styles["header"],
                page_action="none",
                style_as_list_view=True
            )
            
            # Combine Leaders and Laggards tables in a row
            small_tables = dash.html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5(summary_tables_config["title_leaders"], className="text-center"), 
                        best_table
                    ], width=6),
                    dbc.Col([
                        html.H5(summary_tables_config["title_laggards"], className="text-center"), 
                        worst_table
                    ], width=6)
                ], className="mb-4")
            ])
            
            # Return Leaders, Laggards, and main table
            return html.Div([small_tables, main_table])
        else:
            # For specific country, just return the main table
            return main_table

    # Callback to update Country Data
    @app.callback(
        Output("pop-data", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_country_data(selected_country):
        """
        Update the country demographic data display in the sidebar.
        
        Returns:
            dash.html.Div: Formatted country data
        """
        if not selected_country or selected_country == "All":
            return ""
            
        row = df_index[df_index["Country"] == selected_country]
        if row.empty:
            return html.Div("Population: N/A")
            
        row = row.iloc[0]
        pop = row.get("Population", "N/A")
        ppa = row.get("PPA", "N/A")
        gdp = row.get("GDP", "N/A")
        ltg = row.get("LTG", "N/A")
        
        # Format population with commas
        pop_formatted = format_with_commas(pop)
            
        return html.Div([
            html.P(f"Population: {pop_formatted}"),
            html.P(f"PPA: {ppa}"),
            html.P(f"GDP: {gdp}"),
            html.P(f"LTG: {ltg:.2}" if isinstance(ltg, (int, float)) else f"LTG: {ltg}")
        ])

    # Callback to update Economic Data table
    @app.callback(
        Output("economic-table", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_economic_data(selected_country):
        """
        Update the economic data table for the selected country.
        
        Returns:
            dash.html.Table: Economic data table
        """
        if not selected_country or selected_country == "All":
            return ""
            
        row = df_index[df_index["Country"] == selected_country]
        if row.empty:
            return ""
        
        # Get economic table configuration
        economic_config = config["tables"]["economic_table"]
        headers = economic_config["headers"]
        indicators = economic_config["indicators"]
        styles = economic_config["styles"]
        
        # Create header row
        header_row_elem = html.Tr([
            html.Th(
                col, 
                style={
                    "backgroundColor": styles["header"]["background"],
                    "color": styles["header"]["color"], 
                    "padding": styles["header"]["padding"]
                }
            ) for col in headers
        ])
        
        # Create data rows
        data_rows = []
        row_data = row.iloc[0]
        for indicator in indicators:
            indicator_name = indicator["name"]
            cols = indicator["columns"]
            
            row_cells = [html.Td(indicator_name, style={"padding": styles["cell"]["padding"]})]
            for col in cols:
                value = format_percentage(row_data.get(col, "N/A"))
                row_cells.append(html.Td(value, style={"padding": styles["cell"]["padding"]}))
                
            data_rows.append(html.Tr(row_cells))
        
        # Create the table
        table = html.Table(
            [header_row_elem] + data_rows,
            style=styles["table"]
        )
        
        return table

    # Callback to update Forecast Data table
    @app.callback(
        Output("forecast-table", "children"),
        [Input("dropdown-country", "value")]
    )
    def update_forecast_data(selected_country):
        """
        Update the forecast data tables for the selected country.
        
        Returns:
            dash.html.Div: Container with forecast tables
        """
        if not selected_country or selected_country == "All":
            return ""
            
        row = df_index[df_index["Country"] == selected_country]
        if row.empty:
            return ""
            
        row_data = row.iloc[0]

        # --- Institute Forecast table ---
        forecast_config = config["tables"]["forecast_table"]
        fore_headers = forecast_config["headers"]
        fore_indicators = forecast_config["indicators"]
        fore_styles = forecast_config["styles"]
        
        # Create header row
        fore_header_row = html.Tr([
            html.Th(
                col, 
                style={
                    "backgroundColor": fore_styles["header"]["background"], 
                    "color": fore_styles["header"]["color"], 
                    "padding": fore_styles["header"]["padding"]
                }
            )
            for col in fore_headers
        ])
        
        # Create data rows
        data_rows = []
        for indicator in fore_indicators:
            indicator_name = indicator["name"]
            cols = indicator["columns"]
            
            row_cells = [html.Td(indicator_name, style={"padding": fore_styles["cell"]["padding"]})]
            for col in cols:
                value = format_percentage(row_data.get(col, "N/A"))
                row_cells.append(html.Td(value, style={"padding": fore_styles["cell"]["padding"]}))
                
            data_rows.append(html.Tr(row_cells))
        
        # Create the forecast table
        forecast_table = html.Table(
            [fore_header_row] + data_rows,
            style=fore_styles["table"]
        )
        
        # --- GDP Growth and Inflation table ---
        index_forecast_config = config["tables"]["index_forecast_table"]
        index_headers = index_forecast_config["headers"]
        index_indicators = index_forecast_config["indicators"]
        index_styles = index_forecast_config["styles"]
        
        indicator_header = html.Tr([
            html.Th(
                col, 
                style={
                    "backgroundColor": index_styles["header"]["background"],
                    "color": index_styles["header"]["color"],
                    "padding": index_styles["header"]["padding"]
                }
            ) for col in index_headers
        ])
        
        # Create rows for each indicator
        index_rows = []
        for indicator in index_indicators:
            indicator_name = indicator["name"]
            cols = indicator["columns"]
            
            cells = [html.Td(indicator_name, style={"padding": index_styles["cell"]["padding"]})]
            for col in cols:
                value = format_percentage(row_data.get(col, "N/A"))
                cells.append(html.Td(value, style={"padding": index_styles["cell"]["padding"]}))
                
            index_rows.append(html.Tr(cells))
        
        # Create the indicator table
        indicator_table = html.Table(
            [indicator_header] + index_rows,
            style=index_styles["table"]
        )
        
        # Return both tables in a container
        return html.Div([forecast_table, indicator_table])