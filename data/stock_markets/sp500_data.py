import datetime

import pandas as pd
import yfinance as yf
import os
import sqlite3
import numpy as np


def setup_db(database_file=None):
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    if database_file is None:
        database_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../repository/Data.db")
    create_table_query = """
        CREATE TABLE IF NOT EXISTS sp500_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR NOT NULL,
            date_time DATETIME NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            vol NUMERIC  NOT NULL,
            adj_close NUMERIC NOT NULL
        )
    """

    conn = sqlite3.connect(database_file)
    conn.row_factory = dict_factory
    c = conn.cursor()
    c.execute(create_table_query)

    return c, conn


def crawl_sp500_stocks_dataset(database_file):
    cursor, connection = setup_db(database_file)

    check_data_query = """SELECT COUNT(*) as count FROM sp500_stocks"""
    count = cursor.execute(check_data_query).fetchone()
    if "count" in count and count["count"] > 0:
        cursor.close()
        connection.close()
        return

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('../repository/STOCK/S&P500-Info.csv')
    df.to_csv("../repository/STOCK/S&P500-Symbols.csv", columns=['Symbol'])

    df = table[0]
    stockdata = df['Symbol'].to_list()

    ind = 1
    for d in stockdata:
        print(ind, d, datetime.datetime.now())
        ind += 1
        full_stock_data = yf.download(d, '2010-01-01', '2022-09-18')
        full_stock_data = full_stock_data.to_dict("index")
        columns_list = {
            "Adj Close": "adj_close",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume"
        }

        for date_time in full_stock_data.keys():
            for ticker in stockdata:
                record = {
                    "ticker": ticker,
                    "date_time": date_time.strftime("%Y-%m-%d %H:%M:%S")
                }
                is_valid = True
                for data_column in columns_list.keys():
                    if (data_column, ticker) not in full_stock_data:
                        is_valid = False
                        continue

                    if np.isnan(full_stock_data[date_time][(data_column, ticker)]):
                        full_stock_data[date_time][(data_column, ticker)] = 0

                    record[columns_list[data_column]] = full_stock_data[date_time][(data_column, ticker)]

                if not is_valid:
                    continue

                query = """
                    INSERT INTO sp500_stocks 
                    (ticker, date_time, open, high, low, close, adj_close, vol)
                    VALUES 
                    ('%s', '%s', %f, %f, %f, %f, %f, %f)
                """ % (record["ticker"], record["date_time"], record["open"], record["high"],
                       record["low"], record["close"], record["adj_close"], record["volume"])

                cursor.execute(query)
            connection.commit()

    cursor.close()
    connection.close()


def load_sp500_stocks_data(database_file, isin=None):
    cursor, connection = setup_db(database_file=database_file)
    crawl_sp500_stocks_dataset(database_file)

    query = """SELECT * FROM sp500_stocks"""
    if isin is not None:
        query += " where isin='%s'" % isin
    query += " order by date_numeric asc"

    results = cursor.execute(query).fetchall()

    connection.commit()
    connection.close()

    return results


def load_sp500_stocks_isin_list(database_file):
    cursor, connection = setup_db(database_file=database_file)
    crawl_sp500_stocks_dataset(database_file)

    query = """SELECT DISTINCT isin FROM sp500_stocks"""
    results = [r[0] for r in cursor.execute(query).fetchall()]

    connection.commit()
    connection.close()

    return results


def load_sp500_stocks_data_by_isin(database_file, isin):
    cursor, connection = setup_db(database_file=database_file)
    crawl_sp500_stocks_dataset(database_file)

    query = """SELECT * FROM sp500_stocks WHERE isin=%s""" % isin
    results = cursor.execute(query).fetchall()

    connection.commit()
    connection.close()

    return results


if __name__ == "__main__":
    setup_db()
    crawl_sp500_stocks_dataset(database_file=None)
