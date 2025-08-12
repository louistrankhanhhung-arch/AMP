import json, os
from datetime import datetime, timezone

def run():
    report = {
        "asof": datetime.now(timezone.utc).isoformat(),
        "market": {"VNINDEX":{"trend":"up","breadth":"62% adv","vol":"-8% vs 20D"}},
        "summary": {"ENTER": 0, "WAIT": 0, "AVOID": 0},
        "highlights": {"top_enter": [], "near_trigger": [], "risk_notes": []},
        "watchlist": []
    }
    with open('/mnt/data/coverage_report.json','w',encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open('/mnt/data/newsletter.html','w',encoding='utf-8') as f:
        f.write('<h1>Investor Map Pro – Weekly</h1><p>Coming soon…</p>')
    return '/mnt/data/coverage_report.json', '/mnt/data/newsletter.html'

if __name__ == '__main__':
    run()
