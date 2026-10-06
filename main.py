import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Chưa cấu hình GEMINI_API_KEY trong file .env")

app = FastAPI(title="Digital Heritage Multi-Target AI API", version="2.0")
client = genai.Client(api_key=API_KEY)

class HeritageRequest(BaseModel):
    text: str
    level: int  # 1: Tiểu học, 2: THCS, 3: THPT
    target_language: str = "Vietnamese"  # Hỗ trợ khách nước ngoài (English, Chinese, Korean,...)

def build_heritage_prompt(level: int, text: str, target_language: str) -> str:
    # 1. Định nghĩa hướng dẫn dịch thuật cho khách quốc tế
    lang_instruction = ""
    if target_language.lower() != "vietnamese":
        lang_instruction = (
            f"\n\n[QUY TẮC CHO KHÁCH QUỐC TẾ]: Hãy dịch toàn bộ nội dung sang ngôn ngữ '{target_language}'. "
            "Đối với tên di sản, nhân vật lịch sử, triều đại hoặc địa danh đặc thù của Việt Nam (ví dụ: Chùa Thiên Mụ, triều Nguyễn), "
            "bắt buộc phải giữ nguyên tên tiếng Việt kèm phiên âm hoặc giải nghĩa trong ngoặc, tuyệt đối không dịch nghĩa đen để đảm bảo tính xác thực văn hóa."
        )

    # 2. Định nghĩa phong cách theo cấp học (Cấp 1, 2, 3) hoặc phục vụ khách du lịch
    if level == 1:
        base_prompt = (
            "Bạn là một người kể chuyện thiếu nhi thân thiện. Hãy giải thích đoạn di sản sau "
            "cho học sinh **Cấp 1 (Tiểu học)** hoặc khách tham quan nhỏ tuổi bằng ngôn ngữ cực kỳ đơn giản, ngắn gọn, vui tươi, "
            "giống như một câu chuyện cổ tích:\n\n"
            f"Nội dung gốc: {text}"
        )
    elif level == 2:
        base_prompt = (
            "Bạn là một hướng dẫn viên kiêm giáo viên lịch sử tâm huyết. Hãy tóm tắt đoạn di sản sau "
            "với mức độ phù hợp cho học sinh **Cấp 2 (THCS)** hoặc khách tham quan đại chúng, nêu bật các sự kiện cốt lõi. "
            "Đồng thời, hãy tạo thêm 2 câu hỏi trắc nghiệm vui (kèm đáp án):\n\n"
            f"Nội dung gốc: {text}"
        )
    elif level == 3:
        base_prompt = (
            "Bạn là một chuyên gia nghiên cứu văn hóa - lịch sử. Hãy phân tích chuyên sâu "
            "về đoạn di sản sau theo chuẩn tư duy của học sinh **Cấp 3 (THPT)** hoặc nhà nghiên cứu/khách tham quan chuyên sâu, "
            "làm nổi bật bối cảnh, giá trị kiến trúc nghệ thuật và ý nghĩa lịch sử:\n\n"
            f"Nội dung gốc: {text}"
        )
    else:
        raise ValueError("Cấp độ (level) chỉ nhận giá trị 1, 2 hoặc 3.")

    return base_prompt + lang_instruction

@app.post("/api/heritage/process")
async def process_heritage(request: HeritageRequest):
    try:
        prompt = build_heritage_prompt(request.level, request.text, request.target_language)
        
        # Gọi Gemini API phiên bản ổn định
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return {
            "status": "success",
            "target_audience_level": request.level,
            "target_language": request.target_language,
            "ai_result": response.text
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống AI: {str(e)}")