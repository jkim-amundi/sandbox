
"""
Utility Functions Module

This module contains helper functions that are used across the application.
"""

import math
import pandas as pd
from dash import html

# Import configuration
from .config_and_data import config


def cagr(first_val, last_val, year_frac):
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        first_val: Initial value
        last_val: Final value
        year_frac: Time period in years
        
    Returns:
        float: CAGR value or None if calculation is not possible
    """
    if first_val in [None, 0] or year_frac <= 0:
        return None
    return (last_val / first_val) ** (1 / year_frac) - 1


def growth(first_val, last_val):
    """
    Calculate simple growth percentage.
    
    Args:
        first_val: Initial value
        last_val: Final value
        
    Returns:
        float: Growth percentage or None if calculation is not possible
    """
    if first_val is None or first_val == 0:
        return None
    return (last_val / first_val) - 1


def format_percentage(val, precision=None):
    """
    Format a value as a percentage string.
    
    Args:
        val: Value to format
        precision: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    if precision is None:
        precision = config["data_processing"]["calculations"]["numeric_precision"]["percentage"]
        
    try:
        return f"{float(val):.{precision}f}%"
    except (ValueError, TypeError):
        return str(val)


def lighten_color(color, factor=None):
    """
    Lightens an RGB color by interpolating towards white.
    
    Args:
        color: RGB color string in format "rgb(R, G, B)"
        factor: Interpolation factor (0 = original color, 1 = white)
        
    Returns:
        str: Lightened RGB color string
    """
    if factor is None:
        factor = config["charts"]["layout"]["bar"]["lighten_factor"]
        
    # Convert "rgb(R, G, B)" to a list of integers
    rgb = list(map(int, color.replace("rgb(", "").replace(")", "").split(",")))
    # Interpolate towards white (255, 255, 255)
    lighter = [int(c + (255 - c) * factor) for c in rgb]
    return f"rgb({lighter[0]}, {lighter[1]}, {lighter[2]})"


def get_color_for_score(score):
    """
    Determine color based on a score value.
    
    Args:
        score: Numeric score (0-100)
        
    Returns:
        str: Color name (red, green, gold, or black for None)
    """
    if score is None:
        return "black"
        
    thresholds = config["data_processing"]["calculations"]["score_thresholds"]
    
    if score > thresholds["high"]:
        return "green"
    elif score < thresholds["low"]:
        return "red"
    else:
        return "gold"


def create_score_display(label, score):
    """
    Create a colored score display component.
    
    Args:
        label: Label text
        score: Score value
        
    Returns:
        html.P: Dash HTML component with colored score
    """
    score_int = int(round(score)) if score is not None else "N/A"
    return html.P([
        html.Span(f"{label}: ", style={'color': 'black'}),
        html.Span(f"{score_int}", style={'color': get_color_for_score(score)})
    ], style={'textAlign': 'center'})


def filter_valid_columns(df, valid_cols_set):
    """
    Filter a DataFrame to keep only columns that are in the valid set.
    
    Args:
        df: DataFrame to filter
        valid_cols_set: Set of valid column names
        
    Returns:
        list: List of column names that are in both the DataFrame and valid set
    """
    return [col for col in df.columns if col in valid_cols_set]


def order_dataframe_by_country(df, country):
    """
    Reorders a DataFrame to put a specific country first.
    
    Args:
        df: DataFrame to reorder
        country: Country to prioritize
        
    Returns:
        pandas.DataFrame: Reordered DataFrame
    """
    df_country = df[df["Country"] == country]
    df_other = df[df["Country"] != country]
    return pd.concat([df_country, df_other])


def format_value(value, is_percentage=False, precision=None):
    """
    Format a value for display, handling None values.
    
    Args:
        value: Value to format
        is_percentage: Whether to format as percentage
        precision: Number of decimal places
        
    Returns:
        str: Formatted value
    """
    if value is None:
        return "N/A"
        
    if precision is None:
        if is_percentage:
            precision = config["data_processing"]["calculations"]["numeric_precision"]["percentage"]
        else:
            precision = config["data_processing"]["calculations"]["numeric_precision"]["decimal"]
    
    try:
        if is_percentage:
            return f"{value * 100:.{precision}f}%"
        else:
            return f"{value:.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


def format_with_commas(value):
    """
    Format a numeric value with commas for thousands.
    
    Args:
        value: Numeric value to format
        
    Returns:
        str: Formatted value with commas
    """
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


def apply_scenario_weights(df, scenario):
    """
    Apply scenario weights to calculate total score.
    
    Args:
        df: DataFrame with base scores
        scenario: Scenario name
        
    Returns:
        pandas.DataFrame: DataFrame with updated total score
    """
    from .config_and_data import valid_fund, valid_risk, valid_val
    
    # Use standard scenario if none specified
    if scenario is None or scenario not in config["scenarios"]["weights"]:
        scenario = "standard"
    
    # Get weights from configuration
    weights = config["scenarios"]["weights"][scenario]
    
    # Calculate scores based on column means
    df["Fundamental Score"] = df[[col for col in df.columns if col in valid_fund]].mean(axis=1)
    df["Risk Score"] = df[[col for col in df.columns if col in valid_risk]].mean(axis=1)
    df["Valuation Score"] = df[[col for col in df.columns if col in valid_val]].mean(axis=1)
    
    # Apply weighted total score
    df["TotalScore"] = (
        df["Fundamental Score"] * (weights["fundamental"] / 100) +
        df["Risk Score"] * (weights["risk"] / 100) +
        df["Valuation Score"] * (weights["valuation"] / 100)
    )
    
    return df
