"""Install the selected Time-JEPA modules; install requirements.txt separately."""
from setuptools import find_namespace_packages, setup

setup(
    name="time-jepa",
    version="0.1.0",
    description="Grounding video world models with physical time series",
    python_requires=">=3.11",
    packages=find_namespace_packages(include=[
        "app*", "src*", "covla_jepa_rjob*", "covla_probe_rjob*", "covla_law_rjob*",
        "grasp_eval_rjob*", "physion_ocp_rjob*", "intphys2_eval_rjob*",
        "world_model_probe_rjob*", "physics_consistency*", "external_baselines*",
    ], exclude=["*.tests", "*.tests.*"]),
)
