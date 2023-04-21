from sqlalchemy import Column, Integer, String, TIMESTAMP, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
import psycopg2

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'emp_info'
    
    empl_no = Column(Integer, primary_key=True)
    full_name = Column(String)
    cohort = Column(String)
    
class FaceMeta(Base):
    __tablename__ = 'face_metas'

    id = Column(Integer, primary_key=True)
    empl_no = Column(Integer, nullable=False)
    img_name = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)

class FaceEmbedding(Base):
    __tablename__ = 'face_embedding'

    id = Column(Integer, primary_key=True)
    face_id = Column(Integer, nullable=False)
    dimension = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)