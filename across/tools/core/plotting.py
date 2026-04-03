from typing import Any

import plotly.graph_objects as go

from ..core.schemas.custom_types import AstropyDateTime


def plot_visibility_windows(
    visibility_windows: list[dict[str, Any]],
    observatory_name: str | None = None,
    fig: go.Figure | None = None,
    offset: int | float = 0,
) -> go.Figure:
    """
    Method to visualize visibility windows using plotly.

    Parameters
    ----------
    visibility_windows: list[dict[str, Any]]
        A list of dictionaries containing the JSON-serialized
        visibility windows.
    observatory_name: str, optional
        The name of the observatory for these window, by default None.
    fig : go.Figure, optional
        An existing plotly figure to add to, by default None
    offset : int | float, optional
        The x-axis offset to plot new visibility windows, by default 0

    Returns
    -------
    go.Figure
        The plotly figure containing the visibility plot
    """
    if fig is None:
        fig = go.Figure()

    for window in visibility_windows:
        window_starttime = AstropyDateTime(window["window"]["begin"]["datetime"]).to_datetime()
        window_endtime = AstropyDateTime(window["window"]["end"]["datetime"]).to_datetime()

        fig.add_trace(
            go.Scatter(
                x=[offset - 0.35, offset + 0.35, offset + 0.35, offset - 0.35, offset - 0.35],
                y=[window_starttime, window_starttime, window_endtime, window_endtime, window_starttime],
                fill="toself",
                mode="lines",
                hoveron="fills",
                line=dict(
                    width=1,
                    color="black",
                ),
                marker=dict(size=0, opacity=0),
                fillcolor="salmon",
                opacity=0.7,
                hoverinfo="text",
                text=(
                    f"<b>{observatory_name}</b><br>"
                    f"Start: {window_starttime}<br>"
                    f"Start Reason: {window['constraint_reason']['start_reason']}<br>"
                    f"End: {window_endtime}<br>"
                    f"End Reason: {window['constraint_reason']['end_reason']}"
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_color="black",
                ),
                showlegend=False,
            )
        )

    return fig


def plot_joint_visibility_windows(
    visibility_windows: list[dict[str, Any]],
    min_extent: float = 0.0,
    max_extent: float = 0.0,
    fig: go.Figure | None = None,
) -> go.Figure:
    """
    Method to visualize joint visibility windows using plotly.
    Plots the individual instrument visibility windows and the regions
    of joint visibility on one figure.

    Parameters
    ----------
    visibility_windows: list[dict[str, Any]]
        A list of dictionaries containing the JSON-serialized
        visibility windows.
    min_extent: float, optional
        The leftmost extent of the visibility windows,
        for overlap with the joint window. By default 0.
    max_extent: float, optional
        The rightmost extent of the visibility windows,
        for overlap with the joint window. By default 0.
    fig : go.Figure, optional
        An existing plotly figure to add to, by default None

    Returns
    -------
    go.Figure
        The plotly figure containing the visibility plot
    """
    if fig is None:
        fig = go.Figure()

    for window in visibility_windows:
        window_starttime = AstropyDateTime(window["window"]["begin"]["datetime"]).to_datetime()
        window_endtime = AstropyDateTime(window["window"]["end"]["datetime"]).to_datetime()

        fig.add_trace(
            go.Scatter(
                x=[
                    min_extent - 0.5,
                    max_extent + 0.5,
                    max_extent + 0.5,
                    min_extent - 0.5,
                    min_extent - 0.5,
                ],
                y=[window_starttime, window_starttime, window_endtime, window_endtime, window_starttime],
                fill="toself",
                mode="lines",
                hoveron="fills",
                line=dict(
                    width=1,
                    color="black",
                ),
                marker=dict(size=0, opacity=0),
                fillcolor="pink",
                opacity=0.5,
                hoverinfo="text",
                text=(
                    "<b>Joint Window</b><br>"
                    f"Start: {window_starttime}<br>"
                    f"Start Reason: {window['constraint_reason']['start_reason']}<br>"
                    f"End: {window_endtime}<br>"
                    f"End Reason: {window['constraint_reason']['end_reason']}<br>"
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_color="black",
                ),
                showlegend=False,
                zorder=-1,
            )
        )

    return fig


def plot_footprint(
    detectors: list[dict[str, Any]],
    fig: go.Figure | None = None,
    name: str | None = None,
    color: str | None = None,
) -> go.Figure:
    """
    Method to plot a footprint using plotly.

    Parameters:
    -------------
    detectors: list[dict[str, Any]]: The detectors to plot.
        Assumes a list of dictionaries containing a list of coordinates
        as dictionaries.
    fig : go.Figure, optional
        An existing plotly figure to add to, by default None
    name : str | None, optional
        The name to assign to the detector traces, by default None
    color : str | None, optional
        The color to assign to the detector traces, by default None

    Returns
    -------
    go.Figure
        The plotly figure containing the footprint plot
    """
    if fig is None:
        fig = go.Figure()

    for i, detector in enumerate(detectors):
        ra_values = [coord["ra"] for coord in detector["coordinates"]] + [detector["coordinates"][0]["ra"]]
        dec_values = [coord["dec"] for coord in detector["coordinates"]] + [detector["coordinates"][0]["dec"]]

        # Check if trace with same name already exists
        name_exists = False
        if name:
            for trace in fig.data:
                name_exists = trace.name == name  # type: ignore[attr-defined]
                break

        # Only show legend for first detector if name exists
        show_legend = i == 0 and not name_exists

        # Set legend group to name or unique id
        legend_group = name if name else None

        fig.add_trace(
            go.Scattergeo(
                lon=ra_values,
                lat=dec_values,
                mode="lines",
                fill="none",
                name=name if name else legend_group,
                legendgroup=legend_group,
                line=dict(color=color) if color else None,
                showlegend=show_legend,
            )
        )
    return fig
