"""Time-series forecasting pipeline stages - parallel to the tabular stages
in the top-level src/ modules, dispatched to by src/dags/dag_factory.py when
a pipeline's problem_type is "forecasting". See config/pjm_load_forecast/
for the first pipeline using this path.
"""
