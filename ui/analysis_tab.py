"""
ui/analysis_tab.py

Holds the two dataset-analysis views as sub-tabs:

  Physical Features  interpretable quantities (distance, speed) with CSV export
  PCA Comparison     Doppler-domain vs range-domain features projected to 2D

Both read the same raw sample files and answer the same underlying question —
which description of a capture separates the gestures — so they belong
together rather than as two entries in the main tab bar.
"""

from PyQt6 import QtWidgets

from ui import app_colors
from ui.features_tab import FeaturesTab
from ui.pca_tab import PcaTab


def _style(c: dict) -> str:
    return f"""
    QWidget#analysis_root {{ background: {c['bg']}; }}
    QTabWidget::pane {{
        border: none;
        border-top: 1px solid {c['border']};
        background: {c['bg']};
    }}
    QTabBar {{ background: {c['panel']}; }}
    QTabBar::tab {{
        background: {c['panel']};
        color: {c['subtext']};
        padding: 9px 22px;
        font-size: 13px;
        border: none;
        border-bottom: 3px solid transparent;
        min-width: 150px;
    }}
    QTabBar::tab:selected {{
        color: {c['accent']};
        font-weight: bold;
        border-bottom: 3px solid {c['accent']};
    }}
    QTabBar::tab:hover {{ color: {c['accent']}; background: {c['tab_hover']}; }}
    """


class AnalysisTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._c = app_colors()
        self.setObjectName("analysis_root")
        self.setStyleSheet(_style(self._c))

        self.features = FeaturesTab()
        self.pca = PcaTab()

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self.features, "📈   Physical Features")
        self._tabs.addTab(self.pca, "🔬   PCA Comparison")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tabs)

    def refresh(self):
        self.features.refresh()
        self.pca.refresh()

    def stop_if_running(self):
        """Stop whichever sub-tab has a worker running."""
        self.features.stop_if_running()
        self.pca.stop_if_running()
