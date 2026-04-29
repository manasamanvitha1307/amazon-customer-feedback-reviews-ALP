from sqlalchemy import Column, String
from backend.app.database.databse import Base

class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)