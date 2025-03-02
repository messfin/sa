## Basic Analysis
```python
from zmtech import ZMTechAnalysis

# Initialize analyzer
analyzer = ZMTechAnalysis()

# Run basic analysis
results = analyzer.analyze_stocks('AAPL', 'MSFT')
print(results)

# Custom analysis parameters
results = analyzer.analyze_stocks(
    'AAPL',
    'MSFT',
    days_before=10,
    days_after=10,
    include_volume=True
)

# Generate report
report = analyzer.generate_report(results, 'pdf')
```
