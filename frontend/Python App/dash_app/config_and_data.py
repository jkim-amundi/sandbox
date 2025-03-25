"""
Configuration and Data Loading Module

This module handles:
1. Loading the configuration from JSON file
2. Loading static content like markdown files
3. Loading base data from Excel files
4. Preparing dropdown options and other static data structures
"""

import os
import json
import pandas as pd
from datetime import datetime


def load_config():
    """
    Load the configuration from the JSON file.
    
    Returns:
        dict: Configuration dictionary
    """
    with open("config.json", "r") as f:
        return json.load(f)


# Load configuration
config = load_config()


def load_markdown_content():
    """
    Load markdown content from files.
    
    Returns:
        tuple: (institute_view_text, frv_text)
    """
    # Load Institute view markdown
    md_path = os.path.join("assets", "Institute_View.md")
    with open(md_path, "r", encoding="utf-8") as f:
        institute_view_text = f.read()
        
    # Load FRV markdown
    frv_path = os.path.join("assets", "FRV.md")
    with open(frv_path, "r", encoding="utf-8") as f:
        frv_text = f.read()
        
    return institute_view_text, frv_text


# Load markdown content
institute_view_text, frv_text = load_markdown_content()


# Extract metric configuration from config file
mapping_metric = config["metrics"]["mapping"]
metric_labels = config["metrics"]["labels"]
metric_colors = config["metrics"]["colors"]
metric_descriptions = config["metrics"]["descriptions"]

def load_base_data():
    """
    Load the base data from the MacroEdge Excel file.
    
    Returns:
        tuple: (df_names, df_variables, df_index) DataFrames
    """
    file_paths = config["file_paths"]
    macroedge_file = file_paths["macroedge_file"]
    
    df_names = pd.read_excel(macroedge_file, sheet_name=file_paths["names_sheet"])
    df_variables = pd.read_excel(macroedge_file, sheet_name=file_paths["variables_sheet"])
    df_index = pd.read_excel(macroedge_file, sheet_name=file_paths["indexdata_sheet"])
    
    return df_names, df_variables, df_index


# Load the data once at module import time
df_names, df_variables, df_index = load_base_data()


def build_dropdown_options():
    """
    Build dropdown options for countries and metrics.
    
    Returns:
        tuple: (country_options, metric_options)
    """
    # Build country options
    country_options = [{"label": "All", "value": "All"}] + [
        {"label": name, "value": name} for name in sorted(df_names["Country"].dropna().unique())
    ]
    
    # Build metric options (for modal dropdown)
    metric_options = (
        [{"label": "All", "value": "All"}] +
        [{"label": str(x), "value": str(x)} for x in sorted(df_variables["F/R/V"].dropna().unique())]
    )
    
    return country_options, metric_options


# Build dropdown options
country_options, metric_options = build_dropdown_options()


# Extract dropdown texts from configuration
dropdown_texts = config["dropdown_texts"]


def get_dashboard_file_path():
    """
    Constructs the path to the current month's dashboard file.
    
    Returns:
        str: Path to the dashboard file
    """
    month_format = config["data_processing"]["date_format"]["month_pattern"]
    month_name = datetime.now().strftime(month_format)
    folder_path = os.path.dirname(os.path.abspath(config["file_paths"]["macroedge_file"])) + os.sep
    dashboard_file = folder_path + config["file_paths"]["dashboard_file_pattern"].replace("{month_name}", month_name)
    return dashboard_file


def load_dashboard_data():
    """
    Loads the dashboard data from the current month's file.
    
    Returns:
        pandas.DataFrame: Dashboard data or None if file not found
    """
    dashboard_file = get_dashboard_file_path()
    if not os.path.exists(dashboard_file):
        return None
    
    return pd.read_excel(dashboard_file, sheet_name="Dashboard")


def get_valid_column_sets():
    """
    Get sets of valid columns for each metric category.
    
    Returns:
        tuple: (valid_fund, valid_risk, valid_val) Sets of column names
    """
    valid_fund = set(df_variables[df_variables["F/R/V"].str.lower() == "fundamentals"]["What"].str.strip())
    valid_risk = set(df_variables[df_variables["F/R/V"].str.lower() == "risks"]["What"].str.strip())
    valid_val = set(df_variables[df_variables["F/R/V"].str.lower() == "valuations"]["What"].str.strip())
    
    return valid_fund, valid_risk, valid_val


# Get valid column sets
valid_fund, valid_risk, valid_val = get_valid_column_sets()
