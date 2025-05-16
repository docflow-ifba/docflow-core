import uuid
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, TEXT
from sqlalchemy.orm import relationship
from app.config.database import Base

class TableModel(Base):
    __tablename__ = "notice_tables"

    table_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    notice_id = Column(UUID(as_uuid=True), ForeignKey("notices.notice_id"), nullable=False)
    content = Column(TEXT, nullable=False)

    notice = relationship("NoticeModel", back_populates="tables")
