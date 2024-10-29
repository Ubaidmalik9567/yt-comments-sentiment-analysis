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


def saveAlldata(dataSavePath,savedata):
        
        logging.info("data ingestion started")
        pathlib.Path(dataSavePath).mkdir(parents=True,exist_ok=True)
        savedata.to_csv(dataSavePath + "/rawdata.csv", index=False, header=True)

    
def dataingestion_FromSql():
    
        logging.info("Reading start from database")

        mydb = pymysql.connect(
                host=host,
                user=user,
                password=password,
                db=dbname
        )

        logging.info("Connection Establish! with data base %s", mydb)
        data = pd.read_sql_query("Select * from saved_data",mydb)
        print(data.head())
            
        return data
    
def main():
    
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent 

        filePath = sys.argv[1]
        complete_fileSave_location = home_dir.as_posix() + filePath 

        dataset = dataingestion_FromSql()
        saveAlldata(dataSavePath= complete_fileSave_location,savedata= dataset)

if __name__ == "__main__":
    main()