import os
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docu.db'))
ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE)
Base = declarative_base()


class Document(Base):
    __tablename__ = 'documents'
    doc_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    num_chunks = Column(Integer, default=0)


def init_db():
    Base.metadata.create_all(bind=ENGINE)


def get_session():
    return SessionLocal()
