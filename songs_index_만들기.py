"""songs 폴더의 .osz 파일 목록을 songs/index.json 으로 만듭니다.
곡을 추가/삭제한 뒤 이 파일을 실행(더블클릭 또는 python songs_index_만들기.py)하고, 폴더를 다시 업로드하세요.
Lv1 채보는 목록에 넣지 않습니다. 곡 이름·연주자 표시와 중복 숨기기는 songs/meta.json 에서 정합니다."""
import json
import re
from pathlib import Path

base = Path(__file__).resolve().parent / "songs"
base.mkdir(exist_ok=True)
names = sorted(p.name for p in base.glob("*.osz") if not re.search(r"-Lv1-\d+K\.osz$", p.name, re.I))
(base / "index.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(names)}개 채보 등록:")
for n in names:
    print(" -", n)
input("\n엔터를 누르면 닫힙니다.")
