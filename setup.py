from setuptools import setup, find_packages

setup(
    name="ml_engine",
    version="0.1.0",
    description="Modular ML pipeline (data loading, preprocessing, training, evaluation, explainability)",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "shap"
    ],
    python_requires=">=3.8",
)