# ... giữ nguyên imports & helpers & schemas/validators ở trên ...

def build_messages_classify(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Tạo messages cho model phân loại side/action DỰA TRÊN JSON (không dùng ảnh).
    """
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "trigger_1h": trigger_1h or {}
    }

    system = {
        "role": "system",
        "content": (
            "Bạn là trader kỹ thuật. Nhiệm vụ: PHÂN LOẠI nhanh để quyết định side và hành động."
            "\n- Dựa hoàn toàn trên JSON context (structs 4H/1D + trigger_1h)."
            "\n- Chỉ trả về JSON theo schema:"
            + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
            + "\n- Quy tắc: đồng pha 1D/4H + 1H xác nhận → ENTER; mâu thuẫn/chưa rõ → WAIT; ngược mạnh → AVOID."
            "\n- reasons: 3–6 ý ngắn gọn, nêu bằng chứng (RSI, BB, MA20/50, HH/HL/LH/LL, phân kỳ, SR...)."
        )
    }

    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON:"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)}
        ]
    }
    return [system, user]


def build_messages_plan(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    side: str,
    classify_reasoning: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Messages để model xuất JSON kế hoạch vào lệnh: entries, SL, TP, ETA (không HTML, không ảnh).
    """
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "decision": {"side": side, "from_classify": classify_reasoning or {}}
    }

    system = {
        "role":"system",
        "content": (
            "Bạn là nhà giao dịch. Hãy lập kế hoạch vào lệnh ngắn gọn theo JSON duy nhất."
            "\n- Dựa trên JSON context (4H/1D)."
            "\n- Sử dụng ATR/biên độ gần nhất để đặt SL; TP bám theo SR/BB/Fibo."
            "\n- Quy tắc: long → stop < entries, tps tăng dần; short → stop > entries, tps giảm dần."
            "\n- Tối đa 2 Entry, tối đa 5 TP. Làm tròn đến 6 chữ số thập phân."
            "\n- Trả JSON theo schema: " + json.dumps(PLAN_SCHEMA, ensure_ascii=False)
        )
    }

    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON:"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)}
        ]
    }
    return [system, user]
