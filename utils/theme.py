import plotly.io as pio


def marimo_theme(mo, default: str = "light") -> str:
    """Return the active marimo theme or *default* outside marimo."""
    try:
        return mo.app_meta().theme
    except Exception:
        return default


def apply_plotly_theme(mo):
    """Set plotly theme based on the current marimo theme."""
    template_map = {"dark": "plotly_dark", "light": "plotly_white"}
    template = template_map[marimo_theme(mo)]
    pio.templates.default = template
    return template
