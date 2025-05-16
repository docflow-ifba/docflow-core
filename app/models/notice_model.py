import uuid
from sqlalchemy import Column, String, LargeBinary, Text, Date
from sqlalchemy.dialects.postgresql import UUID
from app.config.database import Base
from sqlalchemy.orm import relationship

class NoticeModel(Base):
    __tablename__ = "notices"

    notice_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    deadline = Column(Date, nullable=False)
    pdf_bytes = Column(LargeBinary, nullable=False)
    content_markdown = Column(Text, nullable=False)
    clean_markdown = Column(Text, nullable=False)

    tables = relationship("TableModel", back_populates="notice", cascade="all, delete-orphan")
