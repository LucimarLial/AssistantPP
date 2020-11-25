import os

import streamlit as st
import sqlalchemy as db
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv())
# load environment variables

DB_USER = os.getenv("DB_USER")
DB_PASSWD = os.getenv("DB_PASSWD")
DB_IP = os.getenv("DB_IP")
DB_NAME = os.getenv("DB_NAME")


# @st.cache(allow_output_mutation=True)
def database(db_user=None, db_passwd=None, db_ip=None, db_name=None, is_table_log=False):
	if is_table_log:
		return db.create_engine(f'postgresql://{DB_USER}:{DB_PASSWD}@{DB_IP}/{DB_NAME}')
		
	return db.create_engine(f'postgresql://{db_user}:{db_passwd}@{db_ip}/{db_name}')

