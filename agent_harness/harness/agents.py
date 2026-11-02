"""Cấu hình các agent dùng chung harness."""

from dataclasses import dataclass


VOICES = {
    "concise": "Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng.",
    "teacher": "Giải thích bằng tiếng Việt cho người mới, có ví dụ.",
    "formal": "Trả lời bằng tiếng Việt trang trọng và chính xác.",
}


@dataclass(frozen=True)
class Agent:
    """Vai trò, model và các tool agent được phép dùng."""

    name: str
    model: str
    role: str
    tools: tuple[str, ...]


AGENTS = {
    "reviewer": Agent(
        name="reviewer",
        model="qwen3-vl:2b-instruct",
        role=(
            "Đọc ảnh trang PDF, tóm tắt và viết báo cáo "
            "bằng tiếng Việt theo yêu cầu. "
            "Mỗi nhận xét cần chỉ rõ trang và bằng chứng. "
            "Không tự bịa lỗi hoặc thông tin ngoài tài liệu. "
            "Chưa đủ bằng chứng thì nói rõ. "
            "Chỉ lưu báo cáo khi người dùng yêu cầu "
            "và thao tác ghi file được cho phép."
        ),
        tools=(
            "list_files",
            "read_text",
            "view_pdf_page",
            "recall",
            "save_report",
        ),
    )
}