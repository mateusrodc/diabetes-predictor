import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


engine = create_engine("sqlite:///banco.db", echo=True)
Base = declarative_base()

class Previsao(Base):
    __tablename__ = "previsoes"
    id = Column(Integer, primary_key=True)
    sexo = Column(String)
    idade = Column(Integer)
    imc = Column(Float)
    probabilidade = Column(Float)
    resultado = Column(String)

Base.metadata.create_all(engine)

def get_session():
    Session = sessionmaker(bind=engine)
    return Session()
