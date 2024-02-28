import sys
from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow, 
                             QMessageBox, QPushButton, QVBoxLayout, QWidget, QComboBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class SectorPerformanceGUI(QMainWindow):
    def __init__(
        self,
        stocks_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> None:
        """
        Initialize the Sector Performance Analysis GUI.

        Args:
            stocks_df (pd.DataFrame): DataFrame containing stock data.
            companies_df (pd.DataFrame): DataFrame containing company sectors.
            index_df (pd.DataFrame): DataFrame containing index data.
        """
        super().__init__()
        self.stocks_df = stocks_df
        self.companies_df = companies_df
        self.index_df = index_df
        self.unique_sectors = sorted(companies_df["Sector"].unique())
        self.init_ui()

    def init_ui(self) -> None:
        """
        Sets up the user interface for the Sector Performance Analysis GUI.
        """
        self.setWindowTitle("Sector Performance Analysis")
        self.setGeometry(100, 100, 1000, 800)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.sector_combo = self.create_sector_combo()
        layout.addWidget(QLabel("Select Sector:"), alignment=Qt.AlignTop)
        layout.addWidget(self.sector_combo)

        self.from_year_input, self.to_year_input = self.create_year_inputs()
        self.year_error_label = QLabel("")
        self.year_error_label.setStyleSheet("color: red;")

        for widget in [
            QLabel("From Year:"),
            self.from_year_input,
            QLabel("To Year:"),
            self.to_year_input,
            self.year_error_label,
            QLabel("Note: Year from 2014-2024"),
        ]:
            layout.addWidget(widget)

        self.create_analyze_button(layout)

    def create_sector_combo(self) -> QComboBox:
        """
        Creates a combo box for selecting a sector.

        Returns:
            QComboBox: The sector selection combo box.
        """
        sector_combo = QComboBox(self)
        sector_combo.addItems(self.unique_sectors)
        return sector_combo

    def create_year_inputs(self) -> Tuple[QLineEdit, QLineEdit]:
        """
        Creates input fields for the user to enter start and end years for analysis.

        Returns:
            A tuple containing QLineEdit objects for the start year (`from_year_input`) and
            the end year (`to_year_input`).
        """
        from_year_input = QLineEdit(self)
        to_year_input = QLineEdit(self)
        from_year_input.textChanged.connect(self.validate_year_input)
        to_year_input.textChanged.connect(self.validate_year_input)
        return from_year_input, to_year_input

    def create_analyze_button(self, layout: QVBoxLayout) -> None:
        """
        Creates an "Analyze" button and adds it to the provided layout.

        Args:
            layout (QVBoxLayout): The layout to which the analyze button will be added.
        """
        analyze_button = QPushButton("Analyze", self)
        analyze_button.clicked.connect(self.perform_sector_analysis)
        layout.addWidget(analyze_button)

    def validate_year_input(self) -> None:
        """
        Validates the year inputs from the user.
        If the input is invalid, an appropriate error message is displayed to the user.
        """
        from_year = self.from_year_input.text()
        to_year = self.to_year_input.text()
        message = self.validate_years(from_year, to_year)
        self.year_error_label.setText(message)

    def validate_years(self, from_year: str, to_year: str) -> str:
        """
        Validate the 'from' and 'to' year strings for correctness and logical order.
        Args:
        from_year (str): The starting year as a string.
        to_year (str): The ending year as a string.

        Returns:
            str: An error message if validation fails, or an empty string if validation is successful.
        """
        if len(from_year) != 4 or len(to_year) != 4:
            return "'From Year' and 'To Year' must be 4-digit numbers."
        try:
            from_year_int, to_year_int = int(from_year), int(to_year)
            if not (2014 <= from_year_int <= 2024) or not (2014 <= to_year_int <= 2024):
                return "Years must be between 2014 and 2024."
            if from_year_int > to_year_int:
                return "'From Year' should be <= 'To Year'."
        except ValueError:
            return "Please enter valid 4-digit years."
        return ""

    def perform_sector_analysis(self) -> None:
        """
        Perform and visualize the analysis of a selected sector's performance compared to the S&P 500 index.
        """
        selected_sector = self.sector_combo.currentText()
        from_year = self.from_year_input.text()
        to_year = self.to_year_input.text()

        if self.validate_years(from_year, to_year):
            QMessageBox.warning(
                self, "Input Error", self.validate_years(from_year, to_year)
            )
            return

        sector_performance, index_performance = self.filter_data_by_sector_and_year(
            selected_sector, int(from_year), int(to_year)
        )
        self.plot_sector_analysis(
            sector_performance,
            index_performance,
            selected_sector,
            int(from_year),
            int(to_year),
        )

    def filter_data_by_sector_and_year(
        self, selected_sector: str, from_year: int, to_year: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Filters the stock and index data for a specific sector and year range.
        Parameters:
        - selected_sector (str): The sector to analyze.
        - from_year (int): The starting year of the analysis period.
        - to_year (int): The ending year of the analysis period.

        Returns:
        - Tuple[pd.Series, pd.Series]: A tuple containing two pandas Series. The first Series contains the
        average daily closing price for the selected sector, indexed by date. The second Series contains
        the daily closing price for the S&P 500 index, also indexed by date.
        """

        self.stocks_df["Date"] = pd.to_datetime(self.stocks_df["Date"])
        self.index_df["Date"] = pd.to_datetime(self.index_df["Date"])

        merged_df = pd.merge(
            self.stocks_df,
            self.companies_df[["Symbol", "Sector", "Longname"]],
            on="Symbol",
            how="left",
        )
        date_mask = (merged_df["Date"].dt.year >= from_year) & (
            merged_df["Date"].dt.year <= to_year
        )
        sector_df = merged_df[(merged_df["Sector"] == selected_sector) & date_mask]
        sector_performance = sector_df.groupby("Date")["Close"].mean()

        index_mask = (self.index_df["Date"].dt.year >= from_year) & (
            self.index_df["Date"].dt.year <= to_year
        )
        index_performance = self.index_df[index_mask].set_index("Date")["S&P500"]

        return sector_performance, index_performance

    def plot_sector_analysis(
        self,
        sector_performance: pd.Series,
        index_performance: pd.Series,
        selected_sector: str,
        from_year: int,
        to_year: int,
    ) -> None:
        """Plots the performance of a selected sector against the S&P 500 index.

        Args:
            sector_performance (pd.Series): Time series of sector performance.
            index_performance (pd.Series): Time series of S&P 500 index performance.
            selected_sector (str): The sector being analyzed.
            from_year (Union[int, str]): The start year of the analysis period.
            to_year (Union[int, str]): The end year of the analysis period.
        """

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(
            sector_performance.index,
            sector_performance,
            label=f"{selected_sector} Sector",
            color="blue",
        )
        ax2.plot(
            index_performance.index,
            index_performance,
            label="S&P 500 Index",
            color="red",
            alpha=0.7,
        )

        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"{selected_sector} Sector Price", color="blue")
        ax2.set_ylabel("S&P 500 Index Price", color="red")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        ax1.set_title(
            f"Sector Performance Analysis: {selected_sector} ({from_year}-{to_year})"
        )
        plt.show()


# Task2
class RevenueGrowthAnalysisGUI(QMainWindow):
    def __init__(
        self,
        stocks_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> None:
        """
        Initializes the AnalysisGUI with stock, company, and index data.

        Args:
            stocks_df (pd.DataFrame): Dataframe containing stock information.
            companies_df (pd.DataFrame): Dataframe containing company details.
            index_df (pd.DataFrame): Dataframe containing index data.
        """
        super().__init__()
        self.stocks_df = stocks_df
        self.companies_df = companies_df
        self.index_df = index_df
        self.init_ui()

    def init_ui(self) -> None:
        """
        Initializes the user interface for the analysis GUI.
        """
        self.setWindowTitle("Revenue Growth vs. Stock Performance Analysis")
        self.setGeometry(100, 100, 1000, 800)
        layout = QVBoxLayout()

        self.year_combo = self.create_year_combo()
        layout.addWidget(QLabel("Select Year:"))
        layout.addWidget(self.year_combo)

        analyze_button = QPushButton("Analyze", self)
        analyze_button.clicked.connect(self.perform_analysis)
        layout.addWidget(analyze_button)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

    def create_year_combo(self) -> QComboBox:
        """
        Creates and returns a combo box filled with years from 2014 to 2024.
        """
        year_combo = QComboBox(self)
        year_combo.addItems(
            [str(year) for year in range(2014, 2025)]
        )  # 2025 is exclusive
        return year_combo

    def perform_analysis(self) -> None:
        selected_year = int(self.year_combo.currentText())
        merged_df = self.merge_dataframes(selected_year)

        if not merged_df.empty:
            self.plot_revenue_growth_vs_stock_performance(merged_df, selected_year)
        else:
            QMessageBox.warning(
                self, "No Data", f"No data available for the year {selected_year}."
            )

    def merge_dataframes(self, selected_year: int) -> pd.DataFrame:
        """
        Merges stock and company dataframes filtered by the selected year.

        Args:
            selected_year (int): The year to filter data on.

        Returns:
            pd.DataFrame: The merged dataframe filtered by the selected year.
        """
        merged_df = pd.merge(
            self.stocks_df,
            self.companies_df[["Symbol", "Sector", "Shortname", "Revenuegrowth"]],
            on="Symbol",
        )
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        merged_df["Year"] = merged_df["Date"].dt.year
        return merged_df[merged_df["Year"] == selected_year]

    def plot_revenue_growth_vs_stock_performance(
        self, merged_df: pd.DataFrame, selected_year: int
    ) -> None:
        """
        Plots revenue growth against stock performance for the selected year.

        Args:
            merged_df (pd.DataFrame): Dataframe containing merged stock and company data.
            selected_year (int): The year of analysis.
        """

        fig = px.scatter(
            merged_df,
            x="Revenuegrowth",
            y="Close",
            color="Sector",
            hover_data=["Shortname"],
            title=f"Revenue Growth vs. Stock Performance ({selected_year})",
        )
        fig.update_yaxes(title_text="Stock Performance")
        fig.update_layout(legend_title_text="Sector")
        fig.show()


class PlottingWindow(QMainWindow):
    def __init__(self, figure: Any) -> None:
        """
        Initializes the PlottingWindow with a matplotlib figure.

        Args:
            figure (Any): A matplotlib figure object to be displayed.
        """
        super().__init__()
        self.setWindowTitle("Analysis Results")
        self.setGeometry(100, 100, 1800, 1000)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.canvas = FigureCanvas(figure)
        layout.addWidget(self.canvas)


# TASK 3
class MonthlyStockAnalysisGUI(QMainWindow):
    def __init__(self, stock_df: pd.DataFrame, companies_df: pd.DataFrame) -> None:
        """
        Initializes the GUI with data frames for stock prices and company information.

        Args:
            stock_df (pd.DataFrame): Historical stock prices.
            companies_df (pd.DataFrame): Company information.
        """
        super().__init__()
        self.stock_df = stock_df
        self.companies_df = companies_df
        self.stock_df["Date"] = pd.to_datetime(self.stock_df["Date"])
        self.init_ui()

    def init_ui(self) -> None:
        """
        Initializes the user interface components for the application.
        """
        self.setWindowTitle("Stock Analysis")
        self.setGeometry(100, 100, 1000, 800)
        layout = QVBoxLayout()

        self.setup_company_combo(layout)
        self.setup_month_combo(layout)
        self.setup_year_combo(layout)
        self.setup_analyze_button(layout)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

    def setup_company_combo(self, layout: QVBoxLayout) -> None:
        """
        Sets up the company selection combo box.

        Args:
            layout (QVBoxLayout): The layout to add the company combo box to.
        """
        layout.addWidget(QLabel("Select Company:"))
        self.company_combo = QComboBox()
        self.company_combo.addItems(sorted(self.companies_df["Shortname"].unique()))
        layout.addWidget(self.company_combo)

    def setup_month_combo(self, layout: QVBoxLayout) -> None:
        """
        Sets up the month selection combo box.

        Args:
            layout (QVBoxLayout): The layout to add the month combo box to.
        """

        layout.addWidget(QLabel("Enter Month (1-12):"))
        self.month_combo = QComboBox()
        self.month_combo.addItems([str(i) for i in range(1, 13)])
        layout.addWidget(self.month_combo)

    def setup_year_combo(self, layout: QVBoxLayout) -> None:
        """
        Sets up the year selection combo box.

        Args:
            layout (QVBoxLayout): The layout to add the year combo box to.
        """
        layout.addWidget(QLabel("Select Year:"))
        self.year_combo = QComboBox()
        self.year_combo.addItems(
            [str(year) for year in range(2014, 2025)]
        )  # 2025 is exclusive
        layout.addWidget(self.year_combo)

    def setup_analyze_button(self, layout: QVBoxLayout) -> None:
        """
        Sets up the analyze button and its click event handler.

        Args:
            layout (QVBoxLayout): The layout to add the analyze button to.
        """
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze_stock)
        layout.addWidget(analyze_button)

    def analyze_stock(self) -> None:
        """
        Analyzes the stock performance for the selected company, month, and year.

        Fetches the stock data based on the selection and displays the
        analysis results. Shows a warning message if no data is available.
        """
        company_shortname = self.company_combo.currentText()
        month = int(self.month_combo.currentText())
        year = int(self.year_combo.currentText())

        filtered_stock = filter_stock_data(
            self.stock_df, self.companies_df, company_shortname, month, year
        )

        if not filtered_stock.empty:
            plot_stock_prices(filtered_stock, company_shortname, month, year)
        else:
            QMessageBox.warning(self, "No Data", "Data available only until 02-2024")


def filter_stock_data(
    stock_df: pd.DataFrame,
    companies_df: pd.DataFrame,
    company_shortname: str,
    month: int,
    year: int,
) -> pd.DataFrame:
    """
    Filters stock data for a specific company and date range.

    Args:
        stock_df (pd.DataFrame): DataFrame containing historical stock data.
        companies_df (pd.DataFrame): DataFrame containing company information.
        company_shortname (str): The short name of the company.
        month (int): The month to filter by.
        year (int): The year to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing stock data for the specified company and date range.
    """
    company_symbol = companies_df[companies_df["Shortname"] == company_shortname][
        "Symbol"
    ].values[0]
    return stock_df[
        (stock_df["Symbol"] == company_symbol)
        & (stock_df["Date"].dt.month == month)
        & (stock_df["Date"].dt.year == year)
    ]


def plot_stock_prices(
    filtered_stock: pd.DataFrame, company_shortname: str, month: int, year: int
) -> None:
    """
    Plots high and low stock prices for a selected company, month, and year.

    Args:
        filtered_stock (pd.DataFrame): The filtered stock data to plot.
        company_shortname (str): The short name of the company.
        month (int): The month of the data.
        year (int): The year of the data.
    """
    high_price = filtered_stock["High"].max()
    low_price = filtered_stock["Low"].min()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_stock["Date"],
            y=filtered_stock["High"],
            mode="lines",
            name="High Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered_stock["Date"],
            y=filtered_stock["Low"],
            mode="lines",
            name="Low Price",
        )
    )

    add_price_annotations(fig, filtered_stock, high_price, low_price)

    fig.update_layout(
        title=f"Stock Prices for {company_shortname} - {month}/{year}",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    fig.show()


def add_price_annotations(
    fig: go.Figure, filtered_stock: pd.DataFrame, high_price: float, low_price: float
) -> None:
    """
    Adds annotations for the highest and lowest prices in the stock data plot.

    Args:
        fig (go.Figure): The figure object to add annotations to.
        filtered_stock (pd.DataFrame): The filtered stock data.
        high_price (float): The highest stock price in the filtered data.
        low_price (float): The lowest stock price in the filtered data.
    """
    if not filtered_stock["High"].isnull().all():
        fig.add_annotation(
            x=filtered_stock.loc[filtered_stock["High"].idxmax()]["Date"],
            y=high_price,
            text=f"Highest Price: {high_price}",
            showarrow=True,
            arrowhead=1,
        )
    if not filtered_stock["Low"].isnull().all():
        fig.add_annotation(
            x=filtered_stock.loc[filtered_stock["Low"].idxmin()]["Date"],
            y=low_price,
            text=f"Lowest Price: {low_price}",
            showarrow=True,
            arrowhead=1,
        )


class MainWindow(QMainWindow):
    def __init__(
        self,
        stocks_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> None:
        """
        Initializes the main window with data for analysis.
        """
        super().__init__()
        self.stocks_df = stocks_df
        self.companies_df = companies_df
        self.index_df = index_df
        self.init_ui()

    def init_ui(self) -> None:
        """
        Sets up the UI components.
        """
        self.setWindowTitle("Stock Analysis - S&P 500")
        self.setGeometry(100, 100, 1000, 800)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.setup_buttons(layout)

    def setup_buttons(self, layout: QVBoxLayout) -> None:
        """
        Configures buttons for analysis options.
        """
        buttons_info = [
            ("1. Sector Performance Analysis", self.open_sector_performance_analysis),
            (
                "2. Revenue Growth vs. Stock Performance Analysis",
                self.open_revenue_growth_analysis,
            ),
            ("3. Monthly Stock Analysis", self.open_monthly_analysis),
        ]

        for text, handler in buttons_info:
            button = QPushButton(text, self)
            button.setFont(QFont("Arial", 14, QFont.Bold))
            button.clicked.connect(handler)
            layout.addWidget(button)

    def open_sector_performance_analysis(self) -> None:
        self.sector_performance_window = SectorPerformanceGUI(
            self.stocks_df, self.companies_df, self.index_df
        )
        self.sector_performance_window.show()

    def open_revenue_growth_analysis(self) -> None:
        self.revenue_growth_window = RevenueGrowthAnalysisGUI(
            self.stocks_df, self.companies_df, self.index_df
        )
        self.revenue_growth_window.show()

    def open_monthly_analysis(self) -> None:
        self.monthly_analysis_window = MonthlyStockAnalysisGUI(
            self.stocks_df, self.companies_df
        )
        self.monthly_analysis_window.show()


if __name__ == "__main__":
    """
    Initializes and runs the stock analysis application.
    """
    app = QApplication(sys.argv)
    stocks_file_path = "datasets/sp500_stocks.csv"
    companies_file_path = "datasets/sp500_companies.csv"
    index_file_path = "datasets/sp500_index.csv"

    stocks_df = pd.read_csv(stocks_file_path)
    companies_df = pd.read_csv(companies_file_path)
    index_df = pd.read_csv(index_file_path)
    main_window = MainWindow(stocks_df, companies_df, index_df)
    main_window.show()
    sys.exit(app.exec_())
