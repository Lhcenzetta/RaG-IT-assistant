from .database import Base
from sqlalchemy import Column, Integer , Float , ForeignKey, String, Boolean
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, autoincrement=True , primary_key=True)
    email = Column(String)
    hashedpassword = Column(String)
    isactive = Column(Boolean)
    created_at = Column(String)

    queries = relationship("Query" ,back_populates="user")

class Query(Base):
    __tablename__ = "query"

    d = Column(Integer, autoincrement=True , primary_key=True)
    userid = Column(Integer , ForeignKey("users.id"), nullable=False)
    question = Column(String)
    answer = Column(String)
    cluster = Column(String)
    latency_ms = Column(String)
    created_at = Column(String)

    user = relationship("User" ,back_populates="queries")