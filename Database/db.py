from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
import psycopg2

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'emp_info'
    
    empl_no = Column(Integer, primary_key=True)
    full_name = Column(String)
    surname = Column(String)
    cohort = Column(String)
    event_time = Column(TIMESTAMP)   