import os
from typing import List, Optional
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

# --- PROMPT TEMPLATE ---
CUSTOM_PROMPT_TEMPLATE = """
Bạn là một trợ lý AI chuyên nghiệp và thân thiện, đang thực hiện QA theo kiểu RAG.
Chỉ sử dụng thông tin trong NGỮ CẢNH để trả lời. 
- Nếu không tìm thấy trong ngữ cảnh, hãy nói rõ: "Không tìm thấy thông tin trong tài liệu."
- Không suy đoán hoặc bịa thêm. Luôn trả lời ngắn gọn, tiếng Việt, có trích nguồn (trang) nếu có.

NGỮ CẢNH:
{context}

LỊCH SỬ TRÒ CHUYỆN:
{chat_history}

CÂU HỎI: {question}

CÂU TRẢ LỜI (tiếng Việt):
"""

QA_PROMPT = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)


def get_pdf_docs(pdf_path: str) -> Optional[List[Document]]:
    """Trích văn bản theo TRANG -> Document, giữ metadata page để truy vết nguồn."""
    reader = PdfReader(pdf_path)

    docs: List[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"Lỗi khi đọc trang {i}: {e}")
            text = ""
        text = text.replace("\x00", "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
    if not docs:
        print("PDF không có văn bản (có thể là scan). Cần OCR để tiếp tục.")
        return None
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    """Chunk theo Document để giữ metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", "?", "!", "—", "-", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: List[Document], api_key: str) -> Optional[FAISS]:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",  # mới
            google_api_key=api_key,
        )
        return FAISS.from_documents(chunks, embedding=embeddings)
    except Exception as e:
        print(f"Lỗi khi tạo FAISS/Embeddings: {e}")
        print("Kiểm tra GOOGLE_API_KEY và phiên bản langchain_google_genai.")
        return None


def build_chain(vs: FAISS, api_key: str) -> Optional[ConversationalRetrievalChain]:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",           # hoặc "gemini-1.5-flash" cho rẻ/nhanh
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True,
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        retriever = vs.as_retriever(
            search_type="mmr",               # đa dạng ngữ cảnh
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,    # rất hữu ích khi debug/hiển thị nguồn
        )
        return chain
    except Exception as e:
        print(f"Lỗi khi khởi tạo chain: {e}")
        return None


def print_sources(source_docs: List[Document]):
    """In ra nguồn tham chiếu (trang) sau mỗi câu trả lời."""
    if not source_docs:
        print("Nguồn: (không có)")
        return
    seen = set()
    out = []
    for d in source_docs:
        page = d.metadata.get("page", "?")
        if page not in seen:
            out.append(f"trang {page}")
            seen.add(page)
    print("Nguồn:", ", ".join(out))


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Lỗi: Chưa thiết lập GOOGLE_API_KEY trong .env")
        return

    pdf_path = 'D:/DL_learning/LLM/RAG Working/RAG_in-memory/Text.pdf'

    docs = get_pdf_docs(pdf_path)
    chunks = split_docs(docs)
    vs = build_vectorstore(chunks, api_key)

    chain = build_chain(vs, api_key)
    if not chain:
        return

    print("Xử lý xong! Bắt đầu hỏi đáp (gõ 'exit' hoặc 'quit' để thoát).")
    # while True:
    #     q = input("\nCâu hỏi của bạn: ").strip()
    #     if q.lower() in {"exit", "quit"}:
    #         print("Cảm ơn bạn đã sử dụng. Tạm biệt!")
    #         break
    #     if not q:
    #         continue
    #
    #     try:
    #         print("Gemini đang suy nghĩ...")
    #         result = chain({"question": q})
    #         answer = result.get("answer") or "Không nhận được câu trả lời."
    #         print("\nCâu trả lời:", answer)
    #         # in nguồn để kiểm chứng RAG
    #         print_sources(result.get("source_documents", []))
    #     except Exception as e:
    #         print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    main()
