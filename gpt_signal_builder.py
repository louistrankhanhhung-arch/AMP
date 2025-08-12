from typing import Dict, Any, Optional
import json

# Placeholder – integrate with OpenAI Responses API in production.
def build_prompt_classify(struct: Dict[str, Any]) -> str:
    return json.dumps({
        "role": "You are a technical classifier. Return concise JSON.",
        "struct": struct,
        "requirements": {
            "output_schema": {"symbol": struct['symbol'], "status":"ENTER|WAIT|AVOID", "reason":"", "trigger_hint":{}}
        }
    }, ensure_ascii=False)

def build_prompt_plan(struct: Dict[str, Any]) -> str:
    return json.dumps({
        "role": "You are a strategist. Produce Entry1-2, SL, TP1-TP5 and ETA using ATR.",
        "struct": struct,
        "format": "Return HTML with <b>Entry</b>/<b>SL</b>/<b>TP</b> and a JSON payload we can parse."
    }, ensure_ascii=False)

def parse_plan_output(text: str) -> Dict[str, Any]:
    # TODO: implement strict parser that extracts numbers & validates ordering
    return {"ok": True, "html": text, "proposal": {}}
