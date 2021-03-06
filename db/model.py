import os

import sqlalchemy as db
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


from db import database # connection to the database

import streamlit as st

Base = declarative_base()


BASE_DIR = os.path.join( os.path.abspath('.') )
DB_DIR = os.path.join( BASE_DIR, 'db' )

# Modeling tb_log_operation table as Python class
# ORM (Object Relational Mapper) creates an object oriented wrapper around the database connection, so that it is possible to interact with tables like Python classes.

class LogOperation(Base):

	__tablename__ = 'tb_log_operation'

	id = db.Column(db.BigInteger, db.Sequence('id_log_operation'), primary_key=True)
	number_workflow = db.Column(db.Integer())
	name_dataset = db.Column(db.String(500))
	name_column = db.Column(db.String(500))
	function_operator = db.Column(db.String(500))
	name_operator = db.Column(db.String(500))
	type_operator = db.Column(db.String(500))
	timestamp = db.Column(db.DateTime(), nullable=True)

	def __repr__(self):
		return "<LogOperation(id='%s', number_workflow='%s', name_dataset='%s', name_column='%s', function_operator='%s', name_operator='%s', type_operator='%s', timestamp='%s')>" % (
					self.id,
					self.number_workflow,
					self.name_dataset,
					self.name_column,
					self.function_operator,
					self.name_operator,
					self.type_operator,
					self.timestamp
				)



# Save preprocessing operations in tb_log_operation

def save_to_database_ORM(conn, **kwargs):

	Session = sessionmaker(bind=conn)
	session = Session()

	log_operation = LogOperation(**kwargs)
	session.add(log_operation)

	session.commit()
	
# increment number_workflow 
	
def query_database_ORM_last_number_workflow(conn):

	Session = sessionmaker(bind=conn)
	session = Session()	
	
	query = session.query(func.max(LogOperation.number_workflow)).first()[0]

	if query is None:
		return 1

	return query + 1


# Creates tb_log_operation in the DB, if it does not exist

engine = database(is_table_log=True)
LogOperation.__table__.create(bind=engine, checkfirst=True)

