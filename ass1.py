#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Science Job Market Analysis
================================

A comprehensive analysis of data science job opportunities, salaries, and cost of living
across different countries to help inform career decisions and relocation choices.

This project combines multiple data sources:
- Data science job listings with salary information
- Cost of living indices by country
- Currency exchange rates
- Country code mappings

Author: William Djong
Date: 2025
"""

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from thefuzz import fuzz
from thefuzz import process
import os
import requests
from pathlib import Path

# Standard libraries
import json
from datetime import datetime


class JobMarketAnalyzer:
    """Main class for analyzing data science job market data."""
    
    def __init__(self, data_dir="data", output_dir="output"):
        """Initialize the analyzer with data and output directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.jobs_df = None
        self.cost_df = None
        self.currency_df = None
        self.country_df = None
        
    def _log_progress(self, step, dataframe=None, message=None):
        """Log progress with optional dataframe info."""
        print(f"✓ {step}")
        if message:
            print(f"  {message}")
        if dataframe is not None:
            print(f"  Shape: {dataframe.shape}")
            if len(dataframe.columns) <= 10:
                print(f"  Columns: {list(dataframe.columns)}")
    
    def load_jobs_data(self, csv_file="ds_jobs.csv"):
        """Load the primary data science jobs dataset."""
        file_path = self.data_dir / csv_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Jobs data file not found: {file_path}")
            
        self.jobs_df = pd.read_csv(file_path)
        self._log_progress("Loaded jobs dataset", self.jobs_df)
        return self.jobs_df
    
    def scrape_cost_of_living_data(self, csv_file="cost_of_living.csv", 
                                  url="https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/cost_of_living.html"):
        """Scrape and cache cost of living data."""
        file_path = self.data_dir / csv_file
        
        if file_path.exists():
            self.cost_df = pd.read_csv(file_path)
            self._log_progress("Loaded cached cost of living data", self.cost_df)
        else:
            self._log_progress("Scraping cost of living data from web...")
            response = requests.get(url)
            tables = pd.read_html(response.text)
            
            if tables:
                self.cost_df = tables[0]
                # Clean column names
                self.cost_df.columns = self.cost_df.columns.str.lower().str.replace(" ", "_")
                
                # Cache the data
                self.cost_df.to_csv(file_path, index=False)
                self._log_progress("Scraped and cached cost of living data", self.cost_df)
            else:
                raise ValueError("No tables found on cost of living page")
                
        return self.cost_df
    
    def scrape_currency_data(self, csv_file="exchange_rates.csv",
                           url="https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/exchange_rates.html"):
        """Scrape and cache currency exchange rate data."""
        file_path = self.data_dir / csv_file
        
        if file_path.exists():
            self.currency_df = pd.read_csv(file_path)
            self._log_progress("Loaded cached currency data", self.currency_df)
        else:
            self._log_progress("Scraping currency exchange rates from web...")
            response = requests.get(url)
            tables = pd.read_html(response.text, header=[0, 1])
            
            if tables:
                df = tables[0]
                # Remove unwanted columns and clean headers
                df = df.loc[:, df.columns.get_level_values(0) != "Nearest actual exchange rate"]
                df.columns = df.columns.droplevel(0)
                df.columns = df.columns.str.replace("\xa0", " ")
                df = df.drop("30 Jun 23", axis=1, errors="ignore")
                df = df.rename(columns={"31 Dec 23": "rate"})
                df.columns = df.columns.str.lower()
                
                self.currency_df = df
                self.currency_df.to_csv(file_path, index=False)
                self._log_progress("Scraped and cached currency data", self.currency_df)
            else:
                raise ValueError("No tables found on currency exchange page")
                
        return self.currency_df
    
    def scrape_country_codes(self, csv_file="country_codes.csv",
                           url="https://www.cse.unsw.edu.au/~cs9321/24T1/ass1/country_codes.html"):
        """Scrape and cache country code mappings."""
        file_path = self.data_dir / csv_file
        
        if file_path.exists():
            self.country_df = pd.read_csv(file_path)
            self._log_progress("Loaded cached country codes", self.country_df)
        else:
            self._log_progress("Scraping country codes from web...")
            response = requests.get(url)
            tables = pd.read_html(response.text)
            
            if tables:
                df = tables[0]
                # Remove unwanted columns and rename
                columns_to_remove = ["Year", "ccTLD", "Notes"]
                df = df.drop(columns=columns_to_remove, errors="ignore")
                
                df = df.rename(columns={
                    "Country name (using title case)": "country",
                    "Code": "code"
                })
                
                self.country_df = df
                self.country_df.to_csv(file_path, index=False)
                self._log_progress("Scraped and cached country codes", self.country_df)
            else:
                raise ValueError("No tables found on country codes page")
                
        return self.country_df
    
    def analyze_data_quality(self):
        """Generate a data quality summary for the jobs dataset."""
        if self.jobs_df is None:
            raise ValueError("Jobs data not loaded. Call load_jobs_data() first.")
            
        summary_df = pd.DataFrame(
            index=self.jobs_df.columns, 
            columns=["observations", "distinct", "missing"]
        )
        
        for column in self.jobs_df.columns:
            observations = self.jobs_df[column].count()
            distinct = self.jobs_df[column].nunique()
            missing = len(self.jobs_df) - observations
            
            summary_df.loc[column, "observations"] = observations
            summary_df.loc[column, "distinct"] = distinct
            summary_df.loc[column, "missing"] = missing
        
        summary_df = summary_df.apply(pd.to_numeric, errors="ignore")
        self._log_progress("Generated data quality summary", summary_df)
        return summary_df
    
    def add_experience_rating(self):
        """Add numerical experience rating based on experience level."""
        if self.jobs_df is None:
            raise ValueError("Jobs data not loaded.")
            
        experience_map = {"EN": 1, "MI": 2, "SE": 3, "EX": 4}
        self.jobs_df["experience_rating"] = self.jobs_df["experience_level"].map(experience_map)
        self._log_progress("Added experience rating column")
        return self.jobs_df
    
    def add_country_names(self):
        """Add full country names based on country codes."""
        if self.jobs_df is None or self.country_df is None:
            raise ValueError("Jobs and country data must be loaded first.")
            
        self.jobs_df = pd.merge(
            self.jobs_df,
            self.country_df,
            left_on="employee_residence",
            right_on="code",
            how="left"
        ).drop("code", axis=1)
        
        self._log_progress("Added country names")
        return self.jobs_df
    
    def convert_to_aud(self, target_year=2023):
        """Convert salaries to Australian dollars for specified year."""
        if self.jobs_df is None or self.currency_df is None:
            raise ValueError("Jobs and currency data must be loaded first.")
            
        # Filter for target year
        self.jobs_df = self.jobs_df[self.jobs_df["work_year"] == target_year].copy()
        
        # Get USD to AUD conversion rate
        usd_rate = self.currency_df[
            self.currency_df["currency"] == "United States dollar"
        ]["rate"].iloc[0]
        
        # Convert salaries
        self.jobs_df["salary_in_aud"] = (
            self.jobs_df["salary_in_usd"] * float(usd_rate)
        ).astype(int)
        
        self._log_progress(f"Converted salaries to AUD for {target_year}")
        return self.jobs_df
    
    def normalize_cost_of_living(self):
        """Normalize cost of living indices relative to Australia."""
        if self.cost_df is None:
            raise ValueError("Cost of living data must be loaded first.")
            
        # Keep only relevant columns
        self.cost_df = self.cost_df[["country", "cost_of_living_plus_rent_index"]].copy()
        
        # Get Australia's index
        aus_index = self.cost_df[
            self.cost_df["country"] == "Australia"
        ]["cost_of_living_plus_rent_index"].iloc[0]
        
        # Normalize relative to Australia
        self.cost_df["cost_of_living_plus_rent_index"] = (
            (self.cost_df["cost_of_living_plus_rent_index"] / aus_index) * 100
        ).round(1)
        
        self.cost_df = self.cost_df.sort_values("cost_of_living_plus_rent_index")
        self._log_progress("Normalized cost of living indices relative to Australia")
        return self.cost_df
    
    def fuzzy_match_countries(self, threshold=90):
        """Match countries between jobs and cost of living data using fuzzy matching."""
        if self.jobs_df is None or self.cost_df is None:
            raise ValueError("Both jobs and cost data must be loaded first.")
            
        matches = []
        for _, job_row in self.jobs_df.iterrows():
            if pd.notna(job_row["country"]):
                best_match, score, _ = process.extractOne(
                    job_row["country"], self.cost_df["country"]
                )
                if score >= threshold:
                    matches.append((job_row.name, best_match))
        
        # Create mapping and add cost of living data
        country_mapping = {}
        for job_index, cost_country in matches:
            cost_value = self.cost_df[
                self.cost_df["country"] == cost_country
            ]["cost_of_living_plus_rent_index"].iloc[0]
            country_mapping[job_index] = cost_value
        
        self.jobs_df["cost_of_living"] = self.jobs_df.index.map(country_mapping)
        self.jobs_df = self.jobs_df.dropna(subset=["cost_of_living"])
        
        self._log_progress(f"Matched countries with {threshold}% threshold", 
                          message=f"Matched {len(matches)} countries")
        return self.jobs_df
    
    def create_salary_pivot_table(self):
        """Create pivot table of average salaries by country and experience."""
        if self.jobs_df is None:
            raise ValueError("Jobs data must be processed first.")
            
        pivot_table = pd.pivot_table(
            self.jobs_df,
            values="salary_in_aud",
            index="country",
            columns="experience_rating",
            aggfunc="mean",
            fill_value=0
        ).astype(int)
        
        # Create multi-level columns
        columns = [(f"salary_in_aud", i) for i in range(1, 5)]
        pivot_table.columns = pd.MultiIndex.from_tuples(columns)
        
        # Sort by average salary across all experience levels
        pivot_table = pivot_table.sort_values(
            by=[("salary_in_aud", 1), ("salary_in_aud", 2), 
                ("salary_in_aud", 3), ("salary_in_aud", 4)],
            ascending=False
        )
        
        self._log_progress("Created salary pivot table", pivot_table)
        return pivot_table
    
    def create_market_visualization(self, top_n_countries=10):
        """Create comprehensive visualization of job markets."""
        if self.jobs_df is None:
            raise ValueError("Jobs data must be processed first.")
            
        # Select top countries by cost of living
        top_countries = (
            self.jobs_df.groupby("country")["cost_of_living"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n_countries)
            .index
        )
        
        filtered_df = self.jobs_df[self.jobs_df["country"].isin(top_countries)]
        
        # Aggregate data
        grouped_df = (
            filtered_df.groupby(["experience_rating", "country"])
            .agg({"cost_of_living": "mean", "salary_in_aud": "mean"})
            .reset_index()
        )
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        colors = {1: "#FF6B6B", 2: "#4ECDC4", 3: "#45B7D1", 4: "#96CEB4"}
        experience_labels = {
            1: "Entry Level (EN)", 
            2: "Mid Level (MI)", 
            3: "Senior Level (SE)", 
            4: "Executive Level (EX)"
        }
        
        for exp_level, group in grouped_df.groupby("experience_rating"):
            plt.scatter(
                group["cost_of_living"],
                group["salary_in_aud"],
                label=experience_labels[exp_level],
                color=colors[exp_level],
                alpha=0.7,
                s=100,
                edgecolors='white',
                linewidth=2
            )
            
            # Annotate points
            for _, row in group.iterrows():
                plt.annotate(
                    row["country"],
                    (row["cost_of_living"], row["salary_in_aud"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    alpha=0.8
                )
        
        # Styling
        plt.title("Data Science Job Market Analysis\nSalary vs Cost of Living by Experience Level", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Cost of Living Index (Relative to Australia)", fontsize=14)
        plt.ylabel("Average Salary (AUD)", fontsize=14)
        
        plt.legend(title="Experience Level", loc="upper right", frameon=True, 
                  fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "job_market_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        self._log_progress(f"Saved visualization to {output_file}")
        
        return grouped_df
    
    def generate_insights_report(self):
        """Generate a comprehensive insights report."""
        if self.jobs_df is None:
            raise ValueError("Data must be processed first.")
            
        # Calculate key metrics
        total_jobs = len(self.jobs_df)
        countries_count = self.jobs_df["country"].nunique()
        avg_salary = self.jobs_df["salary_in_aud"].mean()
        salary_range = (self.jobs_df["salary_in_aud"].min(), self.jobs_df["salary_in_aud"].max())
        
        # Top paying countries
        top_countries = (
            self.jobs_df.groupby("country")["salary_in_aud"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        
        # Experience level distribution
        exp_distribution = self.jobs_df["experience_level"].value_counts()
        
        report = {
            "analysis_date": datetime.now().isoformat(),
            "dataset_summary": {
                "total_jobs_analyzed": total_jobs,
                "countries_represented": countries_count,
                "average_salary_aud": round(avg_salary, 2),
                "salary_range_aud": salary_range
            },
            "top_paying_countries": top_countries.to_dict(),
            "experience_level_distribution": exp_distribution.to_dict(),
            "key_insights": [
                "Analysis focused on 2023 job market data",
                "Salaries converted to AUD using official exchange rates",
                "Cost of living normalized relative to Australia",
                f"Dataset covers {countries_count} countries with {total_jobs} job listings"
            ]
        }
        
        # Save report
        report_file = self.output_dir / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self._log_progress(f"Generated insights report: {report_file}")
        return report
    
    def run_full_analysis(self):
        """Execute the complete analysis pipeline."""
        print("🚀 Starting Data Science Job Market Analysis")
        print("=" * 50)
        
        try:
            # Data ingestion
            print("\n📊 Data Ingestion Phase")
            self.load_jobs_data()
            self.scrape_cost_of_living_data()
            self.scrape_currency_data()
            self.scrape_country_codes()
            
            # Data exploration
            print("\n🔍 Data Exploration Phase")
            quality_summary = self.analyze_data_quality()
            
            # Data processing
            print("\n⚙️ Data Processing Phase")
            self.add_experience_rating()
            self.add_country_names()
            self.convert_to_aud()
            self.normalize_cost_of_living()
            self.fuzzy_match_countries()
            
            # Analysis
            print("\n📈 Analysis Phase")
            pivot_table = self.create_salary_pivot_table()
            visualization_data = self.create_market_visualization()
            report = self.generate_insights_report()
            
            print("\n✅ Analysis Complete!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📊 Total jobs analyzed: {len(self.jobs_df)}")
            print(f"🌍 Countries represented: {self.jobs_df['country'].nunique()}")
            print(f"💰 Average salary (AUD): ${self.jobs_df['salary_in_aud'].mean():,.0f}")
            
            return {
                "jobs_data": self.jobs_df,
                "pivot_table": pivot_table,
                "visualization_data": visualization_data,
                "quality_summary": quality_summary,
                "insights_report": report
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            raise


def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = JobMarketAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_full_analysis()
    
    # Display some key findings
    print("\n🔑 Key Findings:")
    top_countries = (
        results["jobs_data"].groupby("country")["salary_in_aud"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
    )
    
    for i, (country, salary) in enumerate(top_countries.items(), 1):
        print(f"{i}. {country}: ${salary:,.0f} AUD average")


if __name__ == "__main__":
    main()