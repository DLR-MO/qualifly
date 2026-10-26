# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import pandas as pd
from matplotlib.figure import Figure

from qualifly_lib.data_fetching import local_file_in, local_public_ref
from qualifly_lib.visualization import plot_time_series_overview, plot_time_series_overview_interactive


def _frames():
    index = pd.to_datetime([0, 1, 2], unit="s", utc=True)
    source_df = pd.DataFrame({"lat": [52.0, 52.0, 52.1], "lon": [10.0, 10.1, 10.2],
                              "geo_alt": [1000.0, 1001.0, 1002.0]}, index=index)
    return source_df, source_df.copy()


def test_interactive_overview_skips_a_parameter_the_source_lacks():
    index = pd.to_datetime([0, 1], unit="s", utc=True)
    source_df = pd.DataFrame({"lat": [52.0, 52.0], "lon": [10.0, 10.1], "geo_alt": [1000.0, 1001.0]},
                             index=index)
    # the target carries a ground speed (velocity) the source does not
    target_df = source_df.assign(velocity=[120.0, 121.0])
    config = {"source": local_file_in.Config, "target": local_public_ref.Config}

    fig = plot_time_series_overview_interactive(source_df, target_df, config, None, "f1", None)

    titles = {axis.title.text for axis in fig.select_yaxes()}
    assert "geo_alt" in titles and "ground speed [m/s]" not in titles


def test_time_series_overview_renders_an_svg_without_pyplot(tmp_path):
    source_df, target_df = _frames()
    config = {"source": local_file_in.Config, "target": local_public_ref.Config}

    fig = plot_time_series_overview(source_df, target_df, config, filtered_results=None,
                                    score={"original": {"alt_rmse": 2.0}})

    # plain figure object: no global pyplot figure state, own Agg canvas
    assert isinstance(fig, Figure)
    out = tmp_path / "time_series.svg"
    fig.savefig(out, format="svg")
    assert "<svg" in out.read_text(encoding="utf-8")
