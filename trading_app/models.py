from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10))
    side = Column(String(10))
    qty = Column(Integer)
    type = Column(String(20))
    status = Column(String(20))
    price = Column(Float)
    submitted_at = Column(DateTime, default=datetime.utcnow)

    def save(self):
        session = SessionLocal()
        session.add(self)
        session.commit()
        session.close()

# DB connection
if all(
    os.getenv(var)
    for var in [
        "MYSQL_USER",
        "MYSQL_PASSWORD",
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DB",
    ]
):
    DB_URL = (
        f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
        f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{os.getenv('MYSQL_DB')}"
    )
else:
    DB_URL = "sqlite:///:memory:"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)
