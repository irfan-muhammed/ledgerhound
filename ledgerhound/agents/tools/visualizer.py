"""
ledgerhound.agents.tools.visualizer
=================================

Chart generation tool for the EDA agent.  Saves charts as PNG files
and returns the file path so the agent can reference them.

Chart types: histogram, bar, line, scatter, heatmap, box
"""

from __future__ import annotations

import io, os, time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChartResult:
    success: bool
    path: str = ""
    error: str = ""
    chart_type: str = ""
    title: str = ""


class Visualizer:
    """Generate charts from DataFrames.  Used inside the agent sandbox."""

    def __init__(self, output_dir: str = "outputs/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._charts: list[ChartResult] = []

    @property
    def charts(self) -> list[ChartResult]:
        return self._charts

    def generate(self, df: pd.DataFrame, chart_type: str, *,
                 x: str | None = None, y: str | None = None,
                 hue: str | None = None, title: str = "",
                 figsize: tuple = (10, 6), top_n: int = 20) -> ChartResult:
        """Generate a chart and save to disk."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            if chart_type == "histogram":
                col = x or df.select_dtypes(include=[np.number]).columns[0]
                data = df[col].dropna()
                if len(data) > 100_000:
                    data = data.sample(100_000, random_state=42)
                ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")

            elif chart_type == "bar":
                col = x or df.columns[0]
                if y:
                    grouped = df.groupby(col)[y].mean().sort_values(ascending=False).head(top_n)
                else:
                    grouped = df[col].value_counts().head(top_n)
                grouped.plot(kind="bar", ax=ax, edgecolor="black")
                ax.set_ylabel(y or "Count")

            elif chart_type == "line":
                xcol = x or df.columns[0]
                ycol = y or df.columns[1]
                plot_df = df.sort_values(xcol)
                ax.plot(plot_df[xcol], plot_df[ycol])
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)

            elif chart_type == "scatter":
                xcol = x or df.select_dtypes(include=[np.number]).columns[0]
                ycol = y or df.select_dtypes(include=[np.number]).columns[1]
                sample = df.sample(min(5000, len(df)), random_state=42)
                if hue and hue in sample.columns:
                    for val, grp in sample.groupby(hue):
                        ax.scatter(grp[xcol], grp[ycol], label=str(val), alpha=0.5, s=10)
                    ax.legend()
                else:
                    ax.scatter(sample[xcol], sample[ycol], alpha=0.3, s=10)
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)

            elif chart_type == "heatmap":
                numeric = df.select_dtypes(include=[np.number])
                if len(numeric.columns) > 30:
                    numeric = numeric.iloc[:, :30]
                corr = numeric.corr()
                im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
                ax.set_yticklabels(corr.columns, fontsize=6)
                fig.colorbar(im)

            elif chart_type == "box":
                col = x or df.select_dtypes(include=[np.number]).columns[0]
                if hue and hue in df.columns:
                    groups = [grp[col].dropna() for _, grp in df.groupby(hue)]
                    labels = [str(v) for v in df[hue].unique()]
                    ax.boxplot(groups, labels=labels[:10])
                else:
                    ax.boxplot(df[col].dropna())
                ax.set_ylabel(col)

            else:
                return ChartResult(False, error=f"Unknown chart type: {chart_type}")

            ax.set_title(title or f"{chart_type}: {x or ''} {y or ''}")
            plt.tight_layout()

            filename = f"{chart_type}_{int(time.time())}.png"
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            result = ChartResult(True, str(path), chart_type=chart_type, title=title)
            self._charts.append(result)
            return result

        except Exception as e:
            return ChartResult(False, error=str(e))
