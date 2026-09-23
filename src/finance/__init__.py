"""Cross-sectional finance returns/risk pipeline stages - parallel to the
tabular stages in the top-level src/ modules and to src/forecasting/'s
single-series stages, dispatched to by src/dags/dag_factory.py when a
pipeline's problem_type is "finance". See config/m6_returns_risk/ for the
first pipeline using this path.
"""
