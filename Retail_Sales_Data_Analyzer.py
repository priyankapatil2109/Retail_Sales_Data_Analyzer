import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RetailAnalyzer:

    def __init__(self):
        self.df = None

    # Load CSV
    def load_data(self, file_path):

        if not file_path.endswith(".csv"):
            print("Please enter a valid CSV file.")
            return False

        try:
            self.df = pd.read_csv(file_path)
            print("\nDataset Loaded Successfully!")

            print("\nChecking Missing Values")
            for column in self.df.columns:
                print(column, ":", self.df[column].isnull().sum())

            # Fill missing values
            self.df.fillna(method="ffill", inplace=True)

            # Convert Date column
            self.df["Date"] = pd.to_datetime(self.df["Date"])

            return True

        except Exception as e:
            print("Error:", e)
            return False

    # Metrics
    def calculate_metrics(self):

        print("\n===== SALES METRICS =====")

        total_sales = self.df["Total Sales"].sum()
        average_sales = self.df["Total Sales"].mean()

        popular_product = self.df.groupby("Product")["Quantity Sold"].sum().idxmax()

        print("Total Sales :", total_sales)
        print("Average Sales :", round(average_sales, 2))
        print("Most Popular Product :", popular_product)

        # NumPy Array
        sales_array = np.array(self.df["Total Sales"])

        print("\nUsing NumPy")
        print("Maximum Sale :", np.max(sales_array))
        print("Minimum Sale :", np.min(sales_array))
        print("Average :", np.mean(sales_array))

        # Growth %
        self.df["Growth %"] = (
            self.df["Total Sales"].pct_change() * 100
        )

        print("\nGrowth % column added.")

    # Summary
    def display_summary(self):

        print("\n===== DATASET SUMMARY =====")

        print("\nFirst 5 Rows")
        print(self.df.head())

        print("\nDataset Info")
        print(self.df.info())

        print("\nStatistics")
        print(self.df.describe())

        print("\nSales by Category")
        print(self.df.groupby("Category")["Total Sales"].sum())

        print("\nSales by Product")
        print(self.df.groupby("Product")["Total Sales"].sum())

        print("\nSales by Date")
        print(self.df.groupby("Date")["Total Sales"].sum())

    # Filter
    def filter_data(self):

        print("\n1. Filter by Category")
        print("2. Filter by Date")

        choice = input("Enter choice: ")

        if choice == "1":

            category = input("Enter Category: ")

            result = self.df[self.df["Category"] == category]

            if result.empty:
                print("No records found.")
            else:
                print(result)

        elif choice == "2":

            date = input("Enter Date (YYYY-MM-DD): ")

            result = self.df[
                self.df["Date"] == pd.to_datetime(date)
            ]

            if result.empty:
                print("No records found.")
            else:
                print(result)

        else:
            print("Invalid Choice")

    # Visualization
    def visualize(self):

        # Bar Chart
        plt.figure(figsize=(8,5))

        category_sales = self.df.groupby("Category")["Total Sales"].sum()

        plt.bar(category_sales.index,
                category_sales.values)

        plt.title("Total Sales by Category")
        plt.xlabel("Category")
        plt.ylabel("Total Sales")
        plt.show()

        # Line Graph
        plt.figure(figsize=(10,5))

        trend = self.df.groupby("Date")["Total Sales"].sum()

        plt.plot(trend.index,
                 trend.values,
                 marker="o",
                 label="Sales")

        plt.title("Sales Trend")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.legend()
        plt.grid(True)

        plt.show()

        # Heatmap
        plt.figure(figsize=(6,4))

        corr = self.df[
            ["Price", "Quantity Sold", "Total Sales"]
        ].corr()

        sns.heatmap(corr,
                    annot=True,
                    cmap="coolwarm")

        plt.title("Correlation Heatmap")

        plt.show()


# ---------------- MAIN PROGRAM ---------------- #

file = input("Enter CSV File Name : ")

analyzer = RetailAnalyzer()

if analyzer.load_data(file):

    while True:

        print("\n========== MENU ==========")
        print("1. Display Summary")
        print("2. Calculate Metrics")
        print("3. Filter Data")
        print("4. Visualize Data")
        print("5. Exit")

        choice = input("Enter your choice : ")

        if choice == "1":
            analyzer.display_summary()

        elif choice == "2":
            analyzer.calculate_metrics()

        elif choice == "3":
            analyzer.filter_data()

        elif choice == "4":
            analyzer.visualize()

        elif choice == "5":
            print("Thank You!")
            break

        else:
            print("Invalid Choice")