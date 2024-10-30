from logger import logging
import pathlib
import sys
import pandas as pd
from dotenv import load_dotenv
import pymysql
import os
import yaml

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
dbname = os.getenv("dbname")

def saveAlldata(dataSavePath, savedata):
    try:
        logging.info("Starting data ingestion...")
        pathlib.Path(dataSavePath).mkdir(parents=True, exist_ok=True)
        savedata.to_csv(dataSavePath + "/rawdata.csv", index=False, header=True)
        logging.info("Data saved successfully at %s", dataSavePath)
    except Exception as e:
        logging.error("Error saving data: %s", e)
        raise

def dataingestion_FromSql():
    try:
        logging.info("Attempting to connect to the database...")
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=dbname
        )
        logging.info("Connection established with database %s", dbname)
        
        try:
            data = pd.read_sql_query("SELECT * FROM saved_data", mydb)
            logging.info("Data fetched successfully from the database & shape is: %s", data.shape)
            data.columns = ['clean_comment', 'category'] 

            print(data.head(5))
            return data
        except Exception as e:
            logging.error("Error fetching data from database: %s", e)
            raise
        finally:
            mydb.close()
            logging.info("Database connection closed.")
    except pymysql.MySQLError as db_err:
        logging.error("Database connection failed: %s", db_err)
        raise
    except Exception as e:
        logging.error("Unexpected error during data ingestion: %s", e)
        raise

def main():
    try:
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent

        filePath = sys.argv[1]
        complete_fileSave_location = home_dir.as_posix() + filePath

        dataset = dataingestion_FromSql()
        saveAlldata(dataSavePath=complete_fileSave_location, savedata=dataset)

    except IndexError:
        logging.error("File path argument missing. Usage: python script.py <filePath>")
    except Exception as e:
        logging.error("An error occurred in the main function: %s", e)

if __name__ == "__main__":
    main()
