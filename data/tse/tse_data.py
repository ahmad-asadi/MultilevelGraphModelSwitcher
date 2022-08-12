from datetime import datetime, timedelta

import pandas as pd

import os
import sqlite3


def setup_db(database_file):
    if database_file is None:
        database_file = "/home/ahmad/MultilevelGraphModelSwitcher/data/repository/Data.db"
    create_table_query = """
        CREATE TABLE IF NOT EXISTS tse_indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isin VARCHAR NOT NULL,
            ticker VARCHAR NOT NULL,
            date_numeric INTEGER NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            open INTEGER NOT NULL,
            high INTEGER NOT NULL,
            low INTEGER NOT NULL,
            close INTEGER NOT NULL,
            vol INTEGER  NOT NULL,
            cap REAL NOT NULL,
            count INTEGER NOT NULL,
            eng_name VARCHAR NOT NULL,
            fa_name VARCHAR NOT NULL,
            last INTEGER NOT NULL
        )
    """

    conn = sqlite3.connect(database_file)
    c = conn.cursor()
    c.execute(create_table_query)

    return c, conn


def crawl_tse_indices_dataset(database_file):
    cursor, connection = setup_db(database_file)

    check_data_query = """SELECT COUNT(*) FROM tse_indices"""
    count = cursor.execute(check_data_query).fetchone()
    if count is not None and count[0] > 0:
        cursor.close()
        connection.close()
        return

    repository_location = "/home/ahmad/MultilevelGraphModelSwitcher/data/repository/TSE/Indices"

    isin_list = []
    for file in os.listdir(repository_location):
        isin_list.append(file.split(".")[0])
        df = pd.read_csv(os.path.join(repository_location, file), encoding="utf-16")
        instrument_data = df.values
        print(file)

        for data in instrument_data:
            record = {
                "isin": file.split(".")[0],
                "ticker": data[0],
                "date_numeric": data[1],
                "year": str(data[1])[:4],
                "month": str(data[1])[4:6],
                "day": str(data[1])[6:],
                "open": int(data[2]),
                "high": int(data[3]),
                "low": int(data[4]),
                "close": int(data[5]),
                "vol": int(data[6]),
                "cap": float(data[7]),
                "count": int(data[8]),
                "eng_name": data[10],
                "fa_name": data[11],
                "last": int(data[12]),
            }

            insert_query = """
                INSERT INTO tse_indices (isin, ticker, date_numeric, year, month, day, open, high, low, close, vol, 
                cap, count, eng_name, fa_name, last) 
                VALUES (:isin, :ticker, :date_numeric, :year, :month, :day, :open, :high, :low,
                :close, :vol, :cap, :count, :eng_name, :fa_name, :last)
            """

            cursor.execute(insert_query, record)
        connection.commit()

    min_date_query = "SELECT MIN(date_numeric), MAX(date_numeric) FROM tse_indices"
    min_date, max_date = cursor.execute(min_date_query).fetchone()

    date = min_date
    # date = 20180321
    while date < max_date:
        existing_isins_list_query = "SELECT isin FROM tse_indices WHERE date_numeric=%d" % date
        existing_isins_list = [r[0] for r in cursor.execute(existing_isins_list_query).fetchall()]
        non_existing_isins = set(isin_list) - set(existing_isins_list)

        if len(non_existing_isins) == len(existing_isins_list):
            continue

        print(date, len(non_existing_isins))

        year = date // 10000
        month = (date - year * 10000) // 100
        day = (date - year * 10000 - month * 100)

        for isin in non_existing_isins:
            last_isin_record_query = """
               SELECT * FROM tse_indices WHERE isin = '%s' and date_numeric < %d order by date_numeric desc
            """ % (isin, date)
            last_isin_record = cursor.execute(last_isin_record_query).fetchone()
            if last_isin_record is not None and len(last_isin_record) > 0:
                record = {
                    "isin": isin,
                    "ticker": last_isin_record[2],
                    "date_numeric": date,
                    "year": year,
                    "month": month,
                    "day": day,
                    "open": last_isin_record[10],
                    "high": last_isin_record[10],
                    "low": last_isin_record[10],
                    "close": last_isin_record[10],
                    "vol": 0,
                    "cap": last_isin_record[12],
                    "count": 0,
                    "eng_name": last_isin_record[14],
                    "fa_name": last_isin_record[15],
                    "last": last_isin_record[10],
                }
            else:
                last_isin_record_query = """
                SELECT * FROM tse_indices WHERE isin = '%s' and date_numeric > %d order by date_numeric asc
                """ % (isin, date)
                last_isin_record = cursor.execute(last_isin_record_query).fetchone()

                record = {
                    "isin": isin,
                    "ticker": last_isin_record[2],
                    "date_numeric": date,
                    "year": year,
                    "month": month,
                    "day": day,
                    "open": last_isin_record[7],
                    "high": last_isin_record[7],
                    "low": last_isin_record[7],
                    "close": last_isin_record[7],
                    "vol": 0,
                    "cap": last_isin_record[12],
                    "count": 0,
                    "eng_name": last_isin_record[14],
                    "fa_name": last_isin_record[15],
                    "last": last_isin_record[7],
                }

            insert_query = """
                            INSERT INTO tse_indices (isin, ticker, date_numeric, year, month, day, open, high, low, 
                            close, vol, cap, count, eng_name, fa_name, last) 
                            VALUES (:isin, :ticker, :date_numeric, :year, :month, :day, :open, :high, :low,
                            :close, :vol, :cap, :count, :eng_name, :fa_name, :last)
                        """

            cursor.execute(insert_query, record)

        connection.commit()

        date = datetime(year=year, month=month, day=day) + timedelta(days=1)
        date = date.year * 10000 + date.month * 100 + date.day

    cursor.close()
    connection.close()


def load_tse_indices_data(database_file):
    cursor, connection = setup_db(database_file=database_file)
    crawl_tse_indices_dataset(database_file)

    query = """SELECT * FROM tse_indices"""
    results = cursor.execute(query).fetchall()

    connection.commit()
    connection.close()

    return results


def load_tse_indices_isin_list(database_file):
    cursor, connection = setup_db(database_file=database_file)
    crawl_tse_indices_dataset(database_file)

    query = """SELECT DISTINCT isin FROM tse_indices"""
    results = [r[0] for r in cursor.execute(query).fetchall()]

    connection.commit()
    connection.close()

    return results


def load_tse_indices_data_by_isin(database_file, isin):
    cursor, connection = setup_db(database_file=database_file)
    crawl_tse_indices_dataset(database_file)

    query = """SELECT * FROM tse_indices WHERE isin=%s""" % isin
    results = cursor.execute(query).fetchall()

    connection.commit()
    connection.close()

    return results


if __name__ == "__main__":
    crawl_tse_indices_dataset(database_file=None)
