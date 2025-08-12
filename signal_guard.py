from typing import Dict, Any
from app.config import settings

def validate_proposal(proposal: Dict[str, Any], struct: Dict[str, Any]) -> (bool, str):
    # TODO: real checks – here is a skeleton
    atr = struct['snapshot']['atr14']
    # SL distance rule
    entry = proposal.get('entry', [struct['snapshot']['price']['close']])[0]
    sl = proposal.get('sl', entry - settings.ATR_SL_K*atr)
    if (entry - sl) > settings.ATR_SL_K * atr:
        return False, f"SL too far (> {settings.ATR_SL_K}*ATR)"
    # TP spacing rule
    tps = proposal.get('tp', [])
    for i in range(1, len(tps)):
        if (tps[i]-tps[i-1]) < settings.TP_MIN_SPACING_ATR * atr:
            return False, "TP spacing too tight"
    return True, "ok"
