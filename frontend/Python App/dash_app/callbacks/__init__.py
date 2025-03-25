"""
Callbacks Initialization Module

This module imports and registers all callback functions from the various callback modules.
"""

def register_callbacks(app, cache):
    """
    Register all callbacks with the Dash application.
    
    Args:
        app: The Dash application instance
        cache: The Flask-Cache instance for caching expensive operations
    """
    # Import all callback modules
    from .callback_modals import register_modal_callbacks
    from .callback_plots import register_plot_callbacks
    from .callback_tables import register_table_callbacks
    from .callback_ppt import register_ppt_callbacks
    
    # Register callbacks from each module
    register_modal_callbacks(app, cache)
    register_plot_callbacks(app, cache)
    register_table_callbacks(app, cache)
    register_ppt_callbacks(app, cache)
    
    print("All callbacks registered successfully.")
