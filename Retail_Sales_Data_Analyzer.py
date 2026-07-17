import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class RetailAnalyzer:
    """
    Retail Sales Data Analyzer
    """

    def __init__(self):
        self.df = None

    # ===============================
    # Load and Clean Dataset
    # ===============================
    def load_data(self, file_path):

        # Check file extension
        if not file_path.lower().endswith(".csv"):
            print("\nError: Please provide a CSV file.")
            return False

        try:
            self.df = pd.read_csv(file_path)

            print("\n===================================")
            print("DATASET LOADED SUCCESSFULLY")
            print("===================================")

            print(f"Rows    : {self.df.shape[0]}")
            print(f"Columns : {self.df.shape[1]}")

            print("\nColumn Names:")
            print(self.df.columns.tolist())

            # -----------------------------
            # Missing Values
            # -----------------------------
            print("\nChecking Missing Values...")

            missing = self.df.isnull().sum()

            print(missing)

            if missing.sum() > 0:
                print("\nMissing values found.")
                print("Applying Forward Fill...")

                self.df.fillna(method="ffill", inplace=True)

                print("Missing values handled.")

            else:
                print("No missing values found.")

            # -----------------------------
            # Required Columns
            # -----------------------------
            required_columns = [
                "Date",
                "Category",
                "Product",
                "Price",
                "Quantity Sold"
            ]

            for col in required_columns:
                if col not in self.df.columns:
                    print(f"\nError: '{col}' column is missing.")
                    return False

            # -----------------------------
            # Date Conversion
            # -----------------------------
            self.df["Date"] = pd.to_datetime(
                self.df["Date"],
                errors="coerce"
            )

            invalid_dates = self.df["Date"].isnull().sum()

            if invalid_dates > 0:
                print(f"\n{invalid_dates} invalid date(s) found.")
                self.df.dropna(subset=["Date"], inplace=True)

            # -----------------------------
            # Numeric Conversion
            # -----------------------------
            self.df["Price"] = pd.to_numeric(
                self.df["Price"],
                errors="coerce"
            )

            self.df["Quantity Sold"] = pd.to_numeric(
                self.df["Quantity Sold"],
                errors="coerce"
            )

            self.df.dropna(
                subset=["Price", "Quantity Sold"],
                inplace=True
            )

            # -----------------------------
            # Invalid Values
            # -----------------------------
            if (self.df["Price"] < 0).any():
                print("\nNegative prices found.")
                self.df = self.df[self.df["Price"] >= 0]

            if (self.df["Quantity Sold"] < 0).any():
                print("\nNegative quantities found.")
                self.df = self.df[self.df["Quantity Sold"] >= 0]

            # -----------------------------
            # Total Sales
            # -----------------------------
            if "Total Sales" not in self.df.columns:

                self.df["Total Sales"] = (
                    self.df["Price"] *
                    self.df["Quantity Sold"]
                )

                print("\n'Total Sales' column created.")

            # -----------------------------
            # Sort by Date
            # -----------------------------
            self.df.sort_values(
                by="Date",
                inplace=True
            )

            self.df.reset_index(
                drop=True,
                inplace=True
            )

            print("\nDataset is ready for analysis.")

            return True

        except FileNotFoundError:
            print("\nError: File not found.")
            return False

        except PermissionError:
            print("\nError: File is open in another program.")
            return False

        except Exception as e:
            print("\nUnexpected Error")
            print(e)
            return False

    # ===============================
    # Display Dataset Summary
    # ===============================
    def display_summary(self):

        if self.df is None:
            print("No dataset loaded.")
            return

        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)

        print("\nFirst 5 Records")
        print(self.df.head())

        print("\nLast 5 Records")
        print(self.df.tail())

        print("\nDataset Shape")
        print(self.df.shape)

        print("\nColumn Names")
        print(self.df.columns.tolist())

        print("\nDataset Information")
        self.df.info()

        print("\nStatistical Summary")
        print(self.df.describe(include="all"))

        print("\nSales by Category")
        print(
            self.df.groupby("Category")["Total Sales"].sum()
        )

        print("\nSales by Product")
        print(
            self.df.groupby("Product")["Total Sales"].sum()
        )

        print("\nSales by Date")
        print(
            self.df.groupby("Date")["Total Sales"].sum()
        )

        print("\nTop 5 Highest Sales")
        print(
            self.df.nlargest(
                5,
                "Total Sales"
            )
        )

        print("\nBottom 5 Lowest Sales")
        print(
            self.df.nsmallest(
                5,
                "Total Sales"
            )
        )

    # ===============================
    # Calculate Business Metrics
    # ===============================
    def calculate_metrics(self):

        if self.df is None:
            print("No dataset loaded.")
            return

        print("\n" + "=" * 50)
        print("SALES METRICS")
        print("=" * 50)

        total_sales = self.df["Total Sales"].sum()

        average_sales = self.df["Total Sales"].mean()

        maximum_sale = self.df["Total Sales"].max()

        minimum_sale = self.df["Total Sales"].min()

        total_quantity = self.df["Quantity Sold"].sum()

        average_price = self.df["Price"].mean()

        most_popular_product = (
            self.df.groupby("Product")["Quantity Sold"]
            .sum()
            .idxmax()
        )

        highest_category = (
            self.df.groupby("Category")["Total Sales"]
            .sum()
            .idxmax()
        )

        print(f"Total Sales           : {total_sales:.2f}")
        print(f"Average Sale          : {average_sales:.2f}")
        print(f"Maximum Sale          : {maximum_sale:.2f}")
        print(f"Minimum Sale          : {minimum_sale:.2f}")
        print(f"Total Quantity Sold   : {total_quantity}")
        print(f"Average Product Price : {average_price:.2f}")
        print(f"Most Popular Product  : {most_popular_product}")
        print(f"Best Category         : {highest_category}")

        # ---------------------------------
        # NumPy Operations
        # ---------------------------------
        print("\nUsing NumPy")

        sales_array = np.array(
            self.df["Total Sales"]
        )

        print("Mean    :", np.mean(sales_array))
        print("Median  :", np.median(sales_array))
        print("Maximum :", np.max(sales_array))
        print("Minimum :", np.min(sales_array))
        print("Std Dev :", np.std(sales_array))
        print("Variance:", np.var(sales_array))

        # ---------------------------------
        # Growth Percentage
        # ---------------------------------
        self.df["Growth %"] = (
            self.df["Total Sales"]
            .pct_change()
            * 100
        )

        self.df["Growth %"] = (
            self.df["Growth %"]
            .fillna(0)
            .round(2)
        )

        print("\nGrowth % Column Added Successfully")

        print("\nFirst 10 Growth Values")
        print(
            self.df[
                [
                    "Date",
                    "Total Sales",
                    "Growth %"
                ]
            ].head(10)
        )

        print("\nMonthly Sales Summary")

        monthly_sales = (
            self.df
            .groupby(
                self.df["Date"].dt.to_period("M")
            )["Total Sales"]
            .sum()
        )

        print(monthly_sales)

    # ===============================
    # Filter Data
    # ===============================
    def filter_data(self):

        if self.df is None:
            print("No dataset loaded.")
            return

        while True:

            print("\n========== FILTER MENU ==========")
            print("1. Filter by Category")
            print("2. Filter by Date Range")
            print("3. Back to Main Menu")

            choice = input("Enter your choice: ")

            if choice == "1":

                categories = self.df["Category"].unique()

                print("\nAvailable Categories:")
                for c in categories:
                    print("-", c)

                category = input("\nEnter Category: ").strip()

                result = self.df[
                    self.df["Category"].str.lower()
                    == category.lower()
                ]

                if result.empty:
                    print("\nNo records found.")
                else:
                    print("\nFiltered Records")
                    print(result)

            elif choice == "2":

                start = input("Enter Start Date (YYYY-MM-DD): ")
                end = input("Enter End Date (YYYY-MM-DD): ")

                try:

                    start = pd.to_datetime(start)
                    end = pd.to_datetime(end)

                    result = self.df[
                        (self.df["Date"] >= start)
                        &
                        (self.df["Date"] <= end)
                    ]

                    if result.empty:
                        print("\nNo records found.")
                    else:
                        print(result)

                except:
                    print("\nInvalid Date Format.")

            elif choice == "3":
                break

            else:
                print("Invalid Choice.")

    # ===============================
    # Visualizations
    # ===============================
    def visualize_data(self):

        if self.df is None:
            print("No dataset loaded.")
            return

        # ------------------------------
        # Bar Chart
        # ------------------------------
        category_sales = (
            self.df.groupby("Category")["Total Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(8,5))
        plt.bar(
            category_sales.index,
            category_sales.values
        )

        plt.title("Total Sales by Category")
        plt.xlabel("Category")
        plt.ylabel("Total Sales")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

        # ------------------------------
        # Line Chart
        # ------------------------------
        trend = (
            self.df.groupby("Date")["Total Sales"]
            .sum()
        )

        plt.figure(figsize=(10,5))

        plt.plot(
            trend.index,
            trend.values,
            marker="o",
            linewidth=2,
            label="Sales"
        )

        plt.title("Sales Trend")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ------------------------------
        # Heatmap
        # ------------------------------
        plt.figure(figsize=(6,4))

        corr = self.df[
            [
                "Price",
                "Quantity Sold",
                "Total Sales"
            ]
        ].corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()


import os

# =====================================
# Main Program
# =====================================

def main():

    print("=" * 50)
    print(" RETAIL SALES DATA ANALYZER ")
    print("=" * 50)

    file_name = input("\nEnter CSV File Name: ")

    # Get the folder where this Python file is stored
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # Create the complete path of the CSV file
    file_name = os.path.join(current_folder, file_name)

    analyzer = RetailAnalyzer()

    if analyzer.load_data(file_name):

        while True:

            print("\n========== MAIN MENU ==========")
            print("1. Display Summary")
            print("2. Calculate Metrics")
            print("3. Filter Data")
            print("4. Visualize Data")
            print("5. Exit")

            choice = input("Enter your choice: ")

            if choice == "1":
                analyzer.display_summary()

            elif choice == "2":
                analyzer.calculate_metrics()

            elif choice == "3":
                analyzer.filter_data()

            elif choice == "4":
                analyzer.visualize_data()    # Change this if your function name is visualize()

            elif choice == "5":

                print("\nThank You!")
                print("Program Closed Successfully.")
                break

            else:
                print("\nInvalid Choice. Please Try Again.")

    else:
        print("\nUnable to Load Dataset.")


if __name__ == "__main__":
    main()
