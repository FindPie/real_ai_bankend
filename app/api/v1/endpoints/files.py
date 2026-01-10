import base64
import io
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.schemas.file import FileParseResponse

router = APIRouter()

# 支持的文件类型
SUPPORTED_EXTENSIONS = {
    "pdf": "PDF",
    "docx": "Word",
    "txt": "文本",
    "md": "文本",
    "json": "文本",
    "js": "文本",
    "ts": "文本",
    "css": "文本",
    "html": "文本",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/parse",
    response_model=FileParseResponse,
    summary="解析上传的文件",
    description="解析 PDF、Word 或文本文件，提取文本内容",
)
async def parse_file(file: UploadFile = File(...)) -> FileParseResponse:
    """
    解析上传的文件

    支持的文件类型:
    - PDF (.pdf)
    - Word (.docx)
    - 文本文件 (.txt, .md, .json, .js, .ts, .css, .html)

    文件大小限制: 10MB
    """
    # 检查文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 获取文件扩展名
    extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {extension}。支持的格式: {', '.join(SUPPORTED_EXTENSIONS.keys())}",
        )

    # 读取文件内容
    content_bytes = await file.read()

    # 检查文件大小
    if len(content_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大，最大支持 {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    file_type = SUPPORTED_EXTENSIONS[extension]

    try:
        if extension == "pdf":
            text_content = await parse_pdf(content_bytes)
        elif extension == "docx":
            text_content = await parse_docx(content_bytes)
        else:
            # 文本文件
            text_content = content_bytes.decode("utf-8")
    except Exception as e:
        logger.error(f"解析文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"解析文件失败: {str(e)}")

    return FileParseResponse(
        content=text_content,
        file_type=file_type,
        filename=file.filename,
        size=len(content_bytes),
    )


async def parse_pdf(content: bytes) -> str:
    """解析 PDF 文件"""
    try:
        import pypdf

        pdf_file = io.BytesIO(content)
        reader = pypdf.PdfReader(pdf_file)

        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts).strip()
    except ImportError:
        raise Exception("PDF 解析库未安装，请安装 pypdf: pip install pypdf")
    except Exception as e:
        raise Exception(f"PDF 解析错误: {str(e)}")


async def parse_docx(content: bytes) -> str:
    """解析 Word 文件"""
    try:
        from docx import Document

        doc_file = io.BytesIO(content)
        doc = Document(doc_file)

        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        return "\n\n".join(text_parts).strip()
    except ImportError:
        raise Exception("Word 解析库未安装，请安装 python-docx: pip install python-docx")
    except Exception as e:
        raise Exception(f"Word 解析错误: {str(e)}")


@router.post(
    "/parse-batch",
    response_model=List[FileParseResponse],
    summary="批量解析文件",
    description="批量解析多个文件",
)
async def parse_files_batch(
    files: List[UploadFile] = File(...),
) -> List[FileParseResponse]:
    """批量解析多个文件"""
    results = []
    for file in files:
        try:
            result = await parse_file(file)
            results.append(result)
        except HTTPException as e:
            logger.warning(f"解析文件 {file.filename} 失败: {e.detail}")
            continue
    return results
