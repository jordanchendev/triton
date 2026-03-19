import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from triton.database import get_db
from triton.models import Document
from triton.schemas import DocumentCreate, DocumentResponse, DocumentListResponse

router = APIRouter()


@router.post("", response_model=DocumentResponse, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)):
    doc = Document(
        type=payload.type,
        source_url=payload.source_url,
        title=payload.title,
        content=payload.content,
        metadata_=payload.metadata,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    type: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(Document)
    if type:
        query = query.filter(Document.type == type)
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return DocumentListResponse(documents=documents, total=total, page=page, page_size=page_size)
