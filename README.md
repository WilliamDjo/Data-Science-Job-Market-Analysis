# 📊 Data Science Job Market Analysis

A comprehensive data analysis project that examines the global data science job market, combining salary data with cost of living metrics to identify the most attractive markets for data science professionals.

## 🎯 Project Overview

This project analyzes data science job opportunities across different countries, considering both salary potential and cost of living to help professionals make informed decisions about where to work or relocate. The analysis combines multiple data sources and provides interactive visualizations to explore market trends.

### Key Features

- **Multi-source Data Integration**: Combines job listings, salary data, cost of living indices, and currency exchange rates
- **Intelligent Data Processing**: Automated web scraping with caching, fuzzy string matching for country names, and currency conversion
- **Comprehensive Analysis**: Data quality assessment, salary benchmarking, and market comparison
- **Professional Visualizations**: High-quality charts and plots for data storytelling
- **Automated Reporting**: JSON-formatted insights and findings

## 🔧 Technology Stack

- **Python 3.11+**: Core programming language
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **requests**: Web scraping
- **thefuzz**: Fuzzy string matching for country names
- **numpy**: Numerical computations

## 📋 Prerequisites

- Python 3.11 or 3.12
- pip package manager
- Internet connection (for initial data scraping)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/data-science-job-analysis.git
cd data-science-job-analysis
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv job_analysis_env

# Activate virtual environment
# On macOS/Linux:
source job_analysis_env/bin/activate
# On Windows:
job_analysis_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Core Dataset

Download the `ds_jobs.csv` file and place it in the `data/` directory (or the script will prompt you for the location).

### 5. Run Analysis

```bash
python job_market_analyzer.py
```

## 📊 Data Sources

1. **Job Listings**: Data science positions with salary information (user-provided CSV)
2. **Cost of Living**: [Numbeo Cost of Living Index](https://www.numbeo.com/cost-of-living/)
3. **Exchange Rates**: Australian Taxation Office currency conversion rates
4. **Country Codes**: ISO 3166-1 alpha-2 country code mappings

_Note: This project uses educational mirrors of these data sources to ensure reproducibility and reduce load on original servers._

## 🎯 Key Analysis Features

### Data Processing Pipeline

1. **Data Ingestion**: Loads job data and scrapes supporting datasets
2. **Data Cleaning**: Standardizes column names, handles missing values
3. **Data Enhancement**: Adds experience ratings, country names, currency conversions
4. **Data Integration**: Fuzzy matches countries across datasets
5. **Analysis & Visualization**: Creates insights and professional charts

### Generated Outputs

- **Salary Analysis**: Average salaries by country and experience level
- **Cost-Benefit Visualization**: Scatter plot showing salary vs. cost of living
- **Data Quality Report**: Missing values, data completeness statistics
- **Market Insights**: Top paying countries, experience level distribution

## 📈 Usage Examples

### Basic Analysis

```python
from job_market_analyzer import JobMarketAnalyzer

# Initialize analyzer
analyzer = JobMarketAnalyzer()

# Run complete analysis
results = analyzer.run_full_analysis()

# Access results
jobs_data = results["jobs_data"]
insights = results["insights_report"]
```

### Custom Analysis

```python
# Initialize with custom directories
analyzer = JobMarketAnalyzer(data_dir="my_data", output_dir="my_results")

# Run step-by-step
analyzer.load_jobs_data("my_jobs.csv")
analyzer.scrape_cost_of_living_data()
analyzer.add_experience_rating()
analyzer.convert_to_aud(target_year=2024)

# Create custom visualization
viz_data = analyzer.create_market_visualization(top_n_countries=15)
```

## 🔍 Key Insights

The analysis typically reveals:

- **Geographic Salary Variations**: Significant differences in data science salaries across countries
- **Experience Premium**: Clear salary progression with experience levels
- **Cost-Adjusted Rankings**: Countries that offer the best value when considering cost of living
- **Market Opportunities**: Emerging markets with good salary-to-cost ratios

## 📋 Requirements File

Create a `requirements.txt` file with:

```txt
matplotlib==3.8.2
numpy==1.26.0
pandas==2.2.0
thefuzz==0.22.1
requests==2.31.0
```

## 🤝 Contributing

Contributions are welcome! Here are some ways to contribute:

1. **Data Sources**: Add new data sources or improve existing scrapers
2. **Analysis**: Implement additional analytical methods or metrics
3. **Visualizations**: Create new chart types or improve existing plots
4. **Documentation**: Improve documentation or add examples
5. **Bug Fixes**: Report and fix issues

### Development Setup

```bash
# Fork the repository and clone your fork
git clone https://github.com/yourusername/data-science-job-analysis.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

## ⚠️ Disclaimer

- This analysis is for educational and informational purposes only
- Salary data may not reflect current market conditions
- Cost of living indices are approximations and may vary by lifestyle
- Always verify data independently before making career decisions

## 📞 Contact

- **Author**: William Djong
- **Email**: william.djong01@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/william-djong)

## 🙏 Acknowledgments

- Job data sourced from AI-Jobs.net
- Cost of living data from Numbeo
- Exchange rates from Australian Taxation Office
- Country codes from ISO 3166-1 standard

---
