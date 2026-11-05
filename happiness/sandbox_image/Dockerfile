# The analysis sandbox: where the Python that Claude writes runs, with no network.
# Happiness builds this image by itself the first time it needs it.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
      "numpy>=2,<3" "pandas>=2,<3" "scipy>=1.13,<2" "scikit-learn>=1.5,<2" \
      "statsmodels>=0.14,<1" "matplotlib>=3.9,<4"
