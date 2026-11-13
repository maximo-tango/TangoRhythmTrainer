#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
탱고 리듬 트레이너 채보 만들기
==============================

회원들이 탱고의 리듬과 멜로디를 발로 표현하도록 연습하는 채보를 만든다.
노트는 왼발(0)·오른발(1) 두 칸이고, 걷듯이 왼발·오른발을 번갈아 딛는다.

모드
  악기 역할
    박자   피아노·반도네온 — 치고 빠지는 중음역 어택
    멜로디 바이올린·콘트라베이스(활)·보컬 비브라토 — 같은 음이 버티거나 커지는 소리

  beat     박자 easy: 탱고의 기본 걸음(tiempo)에 반박자(doble tiempo)와 싱커페이션(síncopa)을 더한다.
             - 기본 걸음: 마르카토 박 두 개마다 한 걸음 (보통 1분에 60~70걸음)
             - 반박자: 걸음 사이 박에 강한 소리가 있으면 한 걸음 더
             - 싱커페이션: 박을 당겨 치는 강세(16분 앞당김, 3-3-2 리듬)가 있으면 그 자리에서 딛고,
               뒤따르는 박이 약하면(붙임줄) 그 박은 쉰다
             - 소리가 거의 없는 박은 쉰다(pausa)
  beat_hard 박자 hard: 피아노·반도네온이 치는 8분음표·16분음표 리듬까지 딛는다.
             - 마디(16분 여덟 칸)마다 탱고 리듬형 중 어택에 가장 맞는 것을 고른다
               (4분, 8분, 16분 연타, 싱코파, 3-3-2, 하바네라, 아라스트레…). 칸마다 따로 정하지 않아
               말이 되는 리듬만 나온다
             - 같은 선율이 되풀이되는 마디는 같은 리듬이 되게 서로 맞춘다
  모든 모드: 노트는 계산한 격자 자리가 아니라 가까운 실제 피아노·반도네온 어택에 붙인다
  musical  종합: 박자 hard 리듬(8분·16분)과 멜로디를 골고루 (MUSICAL_BASE 로 easy 걸음도 가능). 길게 이어지는 멜로디는 '길게 눌렀다 떼는' 노트로
             긴 박자(2박·4박)를 표현한다. 길게 누르는 동안 다른 발은 딛지 않는다(두 발 동시 없음).
             박 = 마르카토 한 박(한 마디 4박). 2박 = 걸음 하나 길이, 4박 = 걸음 둘 길이

설치 (처음 한 번)
  pip install librosa soundfile numpy mutagen

사용
  python tango_chart_maker.py                    # 이 파일 옆 '음원' 폴더의 곡을 모두 → 'songs' 폴더
  python tango_chart_maker.py 음원폴더 --out songs
  python tango_chart_maker.py "Nido Gaucho.mp3"  # 한 곡만

결과 (songs 폴더를 그대로 GitHub 저장소에 올린다)
  songs/index.json               곡 목록 (게임이 읽는다). 곡마다 모드별 난이도 level 도 적는다
                                 (1분당 판정 수 ÷ 100, 0.7~1.5. 공홈 게임·순위와 같은 식)
  songs/<곡id>/audio.mp3         음원 (원본 그대로 복사)
  songs/<곡id>/beat.json         박자 easy 채보
  songs/<곡id>/beat_hard.json    박자 hard 채보 (8분·16분)
  songs/<곡id>/musical.json      종합 채보
  songs/meta.json                (선택) 곡 이름·악단 고치기. 키는 곡id
                                 예) {"nido-gaucho": {"orchestra": "Carlos Di Sarli", "singer": "Alberto Podestá"}}
                                 바리아시온 구간: {"곡id": {"variacion": [["2:24", "2:47"]]}}
                                   → 박자 hard·종합에서 그 구간을 16분 연타로 채운다 (빠른 음 줄기는 분석으로 잘 안 잡힌다)

곡 이름·악단·가수·연도는 음원 태그에서 읽는다. 틀리면 songs/meta.json 에 적고 다시 실행한다
(분석은 다시 하지 않으려면 --skip-existing).

채보 형식 (beat.json / beat_hard.json / musical.json)
  {"version": 1, "mode": "beat", "bpm": 64.2, "notes": [[시작ms, 발, 끝ms, 종류, 강박], ...]}
    발: 0 왼발 1 오른발 / 끝ms: 길게 누르는 노트의 끝(짧은 노트는 0)
    종류: "p" 기본 걸음(4분), "h" 반박자(8분), "q" 16분, "s" 싱커페이션, "m" 멜로디(길게) / 강박: 1이면 마디 첫 박

주의: 녹음에서 소리의 세기·높이를 분석해 만드는 것이라 완벽하지 않다. 이상한 곳은 json 을 직접 고쳐도 된다.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover
    print("librosa 가 없습니다. 먼저 설치하세요:  pip install librosa soundfile numpy mutagen")
    sys.exit(1)

try:
    from mutagen import File as MutagenFile
except ImportError:  # pragma: no cover
    MutagenFile = None

VERSION = 1
SR = 22050
HOP = 256  # 약 11.6ms
AUDIO_EXT = {".mp3", ".m4a", ".ogg", ".wav", ".flac"}

# ── 튜닝 값 (곡이 이상하게 나오면 여기서 조절) ───────────────────
FINE_BPM_RANGE = (95.0, 190.0)  # 마르카토 박(4로 치는 박) 빠르기 범위
PAUSE_STRENGTH = 0.10           # 기본 걸음: 이보다 약하고 조용하면 쉰다
HALF_MIN = 0.45                 # 반박자: 최소 세기
HALF_REL = 0.85                 # 반박자: 앞뒤 걸음 세기 평균에 대한 비율
SYNC_MIN = 0.55                 # 싱커페이션: 최소 세기
SYNC_REL = 1.10                 # 싱커페이션: 가까운 걸음보다 이만큼 세야 한다
SYNC_TIE = 0.55                 # 싱커페이션 뒤 걸음이 (싱커 세기 × 이 값)보다 약하면 붙임줄로 보고 쉰다
MIN_GAP_MS = 150                # 두 걸음 사이 최소 간격
SNAP_MS = 35                    # 노트를 이 안의 실제 피아노·반도네온 어택에 붙인다
SNAP_MIN = 0.30                 # 이보다 약한 어택에는 붙이지 않는다
MELODY_MIN_S = 0.45             # 멜로디 노트 최소 길이(초)
MELODY_MAX_SHARE = 0.40         # 종합: 곡 전체에서 멜로디(길게) 노트가 차지하는 최대 비율
MELODY_PER_PHRASE = 2           # 종합: 한 프레이즈(4마디)에 멜로디 노트 최대 개수
MUSICAL_BASE = "hard"            # 종합: 멜로디를 얹을 박자 채보 ("hard" 8분·16분 리듬 / "easy" 기본 걸음)


# ── 곡 정보 ──────────────────────────────────────────────────
def slugify(text):
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:60] or "song"


def _first(tags, key):
    v = tags.get(key) if tags else None
    if isinstance(v, list):
        v = v[0] if v else None
    return str(v).strip() if v else ""


def _clean(s):
    return re.sub(r"\s+", " ", s.replace("#", "")).strip()


INST_RE = re.compile(r"\s*(?:[-/]\s*)?(?:#?instrumental|inst\.?)\s*$", re.I)


def read_info(path):
    """음원 태그에서 제목·악단·가수·연도. 태그가 없으면 파일 이름"""
    tags = {}
    if MutagenFile is not None:
        try:
            f = MutagenFile(str(path), easy=True)
            tags = dict(f.tags or {}) if f is not None else {}
        except Exception:
            tags = {}
    title = _clean(_first(tags, "title"))
    artist = re.sub(r"^\d+_", "", _clean(_first(tags, "artist")))
    album_artist = re.sub(r"^\d+_", "", _clean(_first(tags, "albumartist")))
    date = _first(tags, "date")

    singer = ""
    instrumental = False
    if INST_RE.search(title):
        title = INST_RE.sub("", title).strip(" -")
        instrumental = True
    if album_artist:
        orchestra = album_artist
        if artist and artist.lower() != album_artist.lower():
            singer = artist
    else:
        orchestra = artist
    if INST_RE.search(orchestra):
        orchestra = INST_RE.sub("", orchestra).strip(" -/")
        instrumental = True
    if singer and (singer.lower() in ("instrumental", "inst", "inst.") or INST_RE.fullmatch(singer)):
        singer = ""
        instrumental = True
    if instrumental and not singer:
        singer = "연주곡"
    if orchestra.lower() in ("instrumental", "tango", "tango compilations"):
        orchestra = ""
    m = re.search(r"(19|20)\d\d", date)
    year = m.group(0) if m else ""
    if not title:
        title = re.sub(r"^\d+\s*[.\-]?\s*(?:-\s*)?", "", path.stem).replace("_", " ").strip()
    return {"title": title, "orchestra": orchestra, "singer": singer, "year": year}


# ── 분석 도구 ────────────────────────────────────────────────
def robust_norm(env):
    hi = float(np.percentile(env, 98)) if len(env) else 1.0
    return np.clip(env / (hi + 1e-9), 0.0, 1.5)


def strength_at(env, t, win=0.035):
    """시각 t(초) 앞뒤 win 안의 가장 큰 값"""
    a = max(0, int((t - win) * SR / HOP))
    b = min(len(env), int((t + win) * SR / HOP) + 1)
    return float(env[a:b].max()) if b > a else 0.0


def mean_at(env, t0, t1):
    a = max(0, int(t0 * SR / HOP))
    b = min(len(env), int(t1 * SR / HOP) + 1)
    return float(env[a:b].mean()) if b > a else 0.0


def track_fine_beats(onset_env, duration):
    """마르카토 박(보통 1분에 120~140)을 찾는다"""
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=SR, hop_length=HOP, start_bpm=125, tightness=100
    )
    beats = librosa.frames_to_time(frames, sr=SR, hop_length=HOP)
    if len(beats) < 8:
        raise RuntimeError("박자를 충분히 찾지 못했습니다")
    step = float(np.median(np.diff(beats)))
    bpm = 60.0 / step
    # 너무 느리면 사이를 채우고, 너무 빠르면 하나 걸러 쓴다
    while bpm < FINE_BPM_RANGE[0]:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        step /= 2
        bpm *= 2
    while bpm > FINE_BPM_RANGE[1]:
        beats = beats[::2]
        step *= 2
        bpm /= 2
    # 앞뒤로 빠진 박 채우기
    head = []
    t = beats[0] - step
    while t > 0.05:
        head.append(t)
        t -= step
    tail = []
    t = beats[-1] + step
    while t < duration - 0.3:
        tail.append(t)
        t += step
    return np.array(sorted(head) + list(beats) + tail), bpm


def snap_steps(steps, env):
    """노트를 ±SNAP_MS 안의 실제 어택 봉우리에 붙인다 (계산한 격자 자리 → 실제 소리)"""
    fps = SR / HOP
    w = int(SNAP_MS / 1000 * fps)
    out = []
    for st in steps:
        c = int(round(st[0] * fps))
        a, b = max(0, c - w), min(len(env), c + w + 1)
        if b - a >= 3:
            k = a + int(np.argmax(env[a:b]))
            if env[k] >= SNAP_MIN and a < k < b - 1:
                st = (k / fps,) + tuple(st[1:])
        out.append(st)
    return out


def pick_phase(times, score, period):
    """times 를 period 개씩 묶었을 때 score 평균이 가장 큰 자리"""
    vals = [float(np.mean([score(t) for t in times[p::period]])) for p in range(period)]
    return int(np.argmax(vals))


# ── 분석 ────────────────────────────────────────────────────
def analyze(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / SR
    y_h, y_p = librosa.effects.hpss(y)
    onset = robust_norm(librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP))
    perc = robust_norm(librosa.onset.onset_strength(y=y_p, sr=SR, hop_length=HOP))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms = rms / (np.percentile(rms, 95) + 1e-9)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bass = robust_norm(S[freqs < 200].sum(axis=0))
    rms_h = librosa.feature.rms(y=y_h, hop_length=HOP)[0]
    rms_p = librosa.feature.rms(y=y_p, hop_length=HOP)[0]

    n = min(len(onset), len(perc), len(rms), len(bass), len(rms_h), len(rms_p))
    onset, perc, rms, bass, rms_h, rms_p = (a[:n] for a in (onset, perc, rms, bass, rms_h, rms_p))
    # 걸음 판단에 쓰는 세기: 피아노·반도네온 음역(150~4000Hz) 어택 + 타악 성분
    mel = librosa.feature.melspectrogram(y=y_p, sr=SR, hop_length=HOP, n_mels=96, fmin=150, fmax=4000)
    mid = robust_norm(librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=SR, hop_length=HOP))
    n = min(n, len(mid))
    onset, perc, rms, bass, rms_h, rms_p, mid = (x[:n] for x in (onset, perc, rms, bass, rms_h, rms_p, mid))
    hit = 0.3 * onset + 0.3 * perc + 0.4 * mid
    # 박자 hard 의 8분·16분: 피아노·반도네온 어택만 (바이올린·베이스의 활 소리는 뺀 타악 성분)
    pb = robust_norm(0.6 * mid + 0.4 * perc)

    fine, fine_bpm = track_fine_beats(onset, duration)
    # 기본 걸음 = 마르카토 박 둘 중 센 쪽 (저음 + 타격)
    phase = pick_phase(fine, lambda t: strength_at(hit, t) + strength_at(bass, t, 0.05), 2)
    pulses = fine[phase::2]
    # 마디 첫 박(2/4 한 마디 = 걸음 둘): 저음이 센 쪽
    bar_phase = pick_phase(pulses, lambda t: strength_at(bass, t, 0.05), 2)

    return {
        "y_h": y_h,
        "melody_bands": melody_bands(y_h),
        "duration": duration,
        "hit": hit,
        "pb": pb,
        "rms": rms,
        "rms_h": rms_h,
        "rms_p": rms_p,
        "pulses": pulses,
        "fine": fine,
        "bar_phase": bar_phase,
        "bpm": 60.0 / float(np.median(np.diff(pulses))),
        "fine_bpm": fine_bpm,
    }


# ── 박자 채보 ────────────────────────────────────────────────
def beat_steps(a, extras=True):
    """박자 easy: [(초, 종류, 강박)] — 종류 p 기본 걸음, h 반박자, s 싱커페이션
    걸음 사이에는 노트를 하나만 넣는다"""
    hit, rms, pulses = a["hit"], a["rms"], a["pulses"]
    steps = []
    skip_next_pulse = False
    for i, p in enumerate(pulses):
        d = (pulses[i + 1] - p) if i + 1 < len(pulses) else (p - pulses[i - 1])
        accent = (i - a["bar_phase"]) % 2 == 0
        sp = strength_at(hit, p)
        quiet = mean_at(rms, p - d / 4, p + d / 4) < 0.12
        if skip_next_pulse:
            skip_next_pulse = False
        elif not (sp < PAUSE_STRENGTH and quiet):
            steps.append((p, "p", accent))
        if not extras or i + 1 >= len(pulses):
            continue
        sn = strength_at(hit, pulses[i + 1])
        s25 = strength_at(hit, p + d * 0.25, 0.025)
        s50 = strength_at(hit, p + d * 0.5)
        s75 = strength_at(hit, p + d * 0.75, 0.025)
        # 싱커페이션: 박을 앞당긴 강세 (0.75 = 다음 박 16분 앞, 0.25 = 3-3-2 의 둘째 강세 자리)
        if s75 >= SYNC_MIN and s75 >= SYNC_REL * max(sn, s50):
            steps.append((p + d * 0.75, "s", False))
            if sn < SYNC_TIE * s75:
                skip_next_pulse = True  # 붙임줄: 다음 박은 쉰다
            continue
        if s25 >= SYNC_MIN and s25 >= SYNC_REL * max(sp, s50):
            steps.append((p + d * 0.25, "s", False))
            continue
        # 반박자 (doble tiempo)
        if s50 >= HALF_MIN and s50 >= HALF_REL * (sp + sn) / 2:
            steps.append((p + d * 0.5, "h", False))
    return thin_steps(snap_steps(steps, a["pb"]))


def thin_steps(steps):
    """너무 가까운 걸음 정리 (기본 걸음을 남긴다)"""
    steps.sort()
    out = []
    for s in steps:
        if out and (s[0] - out[-1][0]) * 1000 < MIN_GAP_MS:
            if out[-1][1] != "p" and s[1] == "p":
                out[-1] = s
            continue
        out.append(s)
    return out


# ── 박자 hard: 탱고 리듬형 ────────────────────────────────────
# 한 마디(2/4, 걸음 둘) = 16분 여덟 칸. 반 마디(걸음 하나) 네 칸마다 아래 리듬형 중 하나를 고른다.
#   x... 4분(기본 걸음)        x.x. 8분 둘(doble tiempo)
#   xxx. 16분 둘 + 8분          x.xx 8분 + 16분 둘          xxxx 16분 넷(스타카토 연타)
#   x..x 점8분 + 16분(싱코파: 다음 박을 16분 앞당김. 뒤 반 마디가 비면 붙임줄,
#        두 반 마디로 3-3-2, 하바네라(x..xx.x.), 마디 끝이면 아라스트레(다음 강박으로 끌기))
#   ..x. 박은 쉬고 8분만        .... 쉼(붙임줄·pausa)
# 반 마디에 16분이 혼자 있는 꼴(xx.., .x.. 등)은 리듬형에 없어서 나오지 않는다.
HALF_PATTERNS = {  # 리듬형: 덜 흔한 꼴일수록 깎는 점수
    "x...": 0.0, "x.x.": 0.0, "xxx.": -0.15, "x.xx": -0.15, "xxxx": -0.25,
    "x..x": -0.10, "..x.": -0.30, "....": 0.0,
}
SLOT_COST = (0.20, 0.85, 0.70, 0.85)  # 칸(박, 16분, 8분, 16분)을 딛는 데 필요한 세기 (근처 걸음 세기 대비)
REPEAT_WEIGHT = 1.5              # 반복 맞춤: 같은 선율이 되풀이되는 마디들의 판단을 이만큼 더한다
REPEAT_SIM = 0.92                # 반복 맞춤: 화성(크로마)이 이만큼 닮은 마디를 '되풀이'로 본다
REPEAT_MAX = 6                   # 반복 맞춤: 마디마다 참고하는 되풀이 마디 수
GHOST_MIN = 0.20                 # 8분·16분·싱커 노트 자리의 어택이 이보다 약하면(소리가 없으면) 뺀다
REPEAT_RHYTHM_SIM = 0.75        # 반복 맞춤: 어택 짜임도 이만큼 닮아야 같은 대목 (화성만 같고 편곡이 다른 곳은 빼고)


def _bar_patterns():
    pats = []
    for h1, c1 in HALF_PATTERNS.items():
        for h2, c2 in HALF_PATTERNS.items():
            pats.append((h1 + h2, c1 + c2))
    return pats


BAR_PATTERNS = _bar_patterns()
_PAT_MASK = np.array([[ch == "x" for ch in p] for p, _ in BAR_PATTERNS], dtype=float)
_PAT_PRIOR = np.array([c for _, c in BAR_PATTERNS])


def hard_bars(a):
    """마디마다 칸 여덟 개의 시각·세기(근처 걸음 세기 대비)와 화성"""
    pb, rms, pulses, bp = a["pb"], a["rms"], a["pulses"], a["bar_phase"]
    n = len(pulses)
    sp_all = np.array([strength_at(pb, p) for p in pulses])
    d_med = float(np.median(np.diff(pulses)))
    chroma = a["melody_bands"]["high"][0]

    def pulse_t(j):
        if j < 0:
            return pulses[0] + j * d_med
        if j >= n:
            return pulses[-1] + (j - n + 1) * d_med
        return pulses[j]

    bars = []
    for j0 in range(bp - 2, n, 2):
        if j0 + 1 < 0:
            continue
        times, ev, valid, accent = [], [], [], []
        for h in range(2):
            j = j0 + h
            t0, t1 = pulse_t(j), pulse_t(j + 1)
            ok = 0 <= j < n - 1 or (j == n - 1)
            local = max(0.3, float(np.median(sp_all[max(0, j - 8):j + 9]))) if 0 <= j < n else 1.0
            quiet = 0 <= j < n and strength_at(a["hit"], t0) < PAUSE_STRENGTH and mean_at(rms, t0 - (t1 - t0) / 4, t0 + (t1 - t0) / 4) < 0.12
            for q in range(4):
                t = t0 + (t1 - t0) * q / 4
                times.append(t)
                inside = 0 <= j < n and (q == 0 or j < n - 1) and t < a["duration"] - 0.2
                valid.append(inside)
                e = strength_at(pb, t, 0.025) / local if inside else 0.0
                if q == 0 and quiet:
                    e = 0.0
                ev.append(e)
                accent.append(q == 0 and h == 0)
        # 화성: 걸음마다 크로마 평균 (되풀이 찾기)
        vs = []
        for h in range(2):
            f0, f1 = _frames(pulse_t(j0 + h), pulse_t(j0 + h + 1))
            vs.append(chroma[:, f0:f1].mean(axis=1) if f1 <= chroma.shape[1] and f1 > f0 else np.zeros(12))
        v = np.concatenate(vs)
        bars.append({"times": times, "ev": np.array(ev), "valid": np.array(valid), "accent": accent,
                     "chroma": v / (np.linalg.norm(v) + 1e-9)})
    return bars


def beat_steps_hard(a):
    """박자 hard: [(초, 종류, 강박)] — 종류 p 기본 걸음(4분), h 8분, q 16분, s 싱커페이션

    1) 마디마다 탱고 리듬형(HALF_PATTERNS 두 개) 중 피아노·반도네온 어택에 가장 맞는 것을 고른다.
       점수 = 딛는 칸마다 (어택 세기 − 칸 비용) + 리듬형 점수. 칸마다 따로 정하지 않으니
       말이 되는 리듬만 나온다.
    2) 같은 선율·화성이 되풀이되는 마디들은 서로의 점수를 더해서 고른다.
       그래서 되풀이되는 대목은 같은 리듬이 되고(외워서 딛기 좋다), 녹음 잡음에 덜 흔들린다.
    3) 노트를 실제 어택에 붙인다."""
    bars = hard_bars(a)
    if not bars:
        return []
    cost = np.array(SLOT_COST * 2)
    E = np.array([b["ev"] for b in bars])                      # 마디 × 8
    V = np.array([b["valid"] for b in bars])
    own = (E - cost) @ _PAT_MASK.T + _PAT_PRIOR                 # 마디 × 리듬형
    own -= 100.0 * ((~V).astype(float) @ _PAT_MASK.T)           # 곡 밖 칸은 딛지 않는다
    # 되풀이 맞춤
    C = np.array([b["chroma"] for b in bars])
    S = C @ C.T
    total = own.copy()
    for m in range(len(bars)):
        sim = S[m].copy()
        sim[max(0, m - 1):m + 2] = -1  # 바로 옆 마디는 빼고
        cand = [k for k in np.argsort(-sim)[:REPEAT_MAX] if sim[k] >= REPEAT_SIM]
        # 리듬 짜임도 비슷해야 같은 대목 (화성만 같고 편곡이 다른 곳은 빼고)
        cand = [k for k in cand if float(E[m] @ E[k]) / (np.linalg.norm(E[m]) * np.linalg.norm(E[k]) + 1e-9) >= REPEAT_RHYTHM_SIM]
        if cand:
            total[m] += REPEAT_WEIGHT * own[cand].mean(axis=0)
    choice = np.argmax(total, axis=1)

    var = a.get("variacion") or []

    def in_var(t):
        return any(v0 <= t < v1 for v0, v1 in var)

    steps = []
    for b, ci in zip(bars, choice):
        pat = BAR_PATTERNS[ci][0]
        for k, ch in enumerate(pat):
            if not b["valid"][k]:
                continue
            q, half = k % 4, pat[(k // 4) * 4:(k // 4) * 4 + 4]
            if in_var(b["times"][k]):
                # 바리아시온: 16분 칸을 모두 딛는다 (0.11초 안팎의 실제 음 줄기를 절반 빠르기로)
                steps.append((b["times"][k], "p" if q == 0 else "h" if q == 2 else "q", b["accent"][k]))
                continue
            if ch != "x":
                continue
            # 되풀이 맞춤으로 골랐어도 그 자리에 소리가 없으면 딛지 않는다
            if q and strength_at(a["pb"], b["times"][k], 0.025) < GHOST_MIN:
                continue
            if q == 0:
                kind = "p"
            elif q == 2:
                kind = "h"
            else:
                kind = "s" if half == "x..x" else "q"
            steps.append((b["times"][k], kind, b["accent"][k]))
    return thin_steps(snap_steps(steps, a["pb"]))


def to_notes(steps):
    """걸음 → [시작ms, 발, 끝ms, 종류, 강박]. 왼발부터 번갈아"""
    notes = []
    for i, st in enumerate(steps):
        t, kind, accent = st[0], st[1], st[2]
        end = st[3] if len(st) > 3 else 0
        notes.append([int(round(t * 1000)), i % 2, int(round(end * 1000)) if end else 0, kind, 1 if accent else 0])
    return notes


# ── 멜로디 ───────────────────────────────────────────────────
MEL_HOP = 512


def melody_bands(y_h):
    """멜로디 악기 음역별 크로마·세기·어택
    high: 바이올린·보컬 (G3~C7)   low: 콘트라베이스 (E1~G3)"""
    out = {}
    for name, fmin, n_oct in (("high", librosa.note_to_hz("G3"), 4), ("low", librosa.note_to_hz("E1"), 2)):
        C = np.abs(librosa.cqt(y_h, sr=SR, hop_length=MEL_HOP, fmin=fmin, n_bins=12 * n_oct, bins_per_octave=12))
        chroma = librosa.feature.chroma_cqt(C=C, sr=SR, hop_length=MEL_HOP, n_chroma=12)
        energy = C.sum(axis=0)
        energy = energy / (np.percentile(energy, 95) + 1e-9)
        attack = robust_norm(librosa.onset.onset_strength(S=librosa.amplitude_to_db(C), sr=SR, hop_length=MEL_HOP))
        out[name] = (chroma, energy, attack)
    return out


def _frames(t0, t1):
    a = max(0, int(t0 * SR / MEL_HOP))
    b = max(a + 1, int(t1 * SR / MEL_HOP))
    return a, b


def melody_runs(a):
    """길게 버티는 멜로디 음 [(점수, 시작초, 끝초, 박 수 2|4)]
    한 박씩 보면서, 같은 음(크로마)이 이어지고 · 세기가 줄지 않고(피아노처럼 꺼지지 않고)
    · 중간에 새 어택이 없으면 '버티는 음'으로 본다"""
    fine = a["fine"]
    cands = []
    for band, (chroma, energy, attack) in a["melody_bands"].items():
        nb = len(fine) - 1
        pc = np.zeros(nb, dtype=int)
        share = np.zeros(nb)
        en = np.zeros(nb)
        vec = np.zeros((nb, 12))
        for i in range(nb):
            f0, f1 = _frames(fine[i], fine[i + 1])
            v = chroma[:, f0:f1].mean(axis=1)
            vec[i] = v / (np.linalg.norm(v) + 1e-9)
            pc[i] = int(np.argmax(v))
            share[i] = float(v.max() / (v.sum() + 1e-9))
            en[i] = float(energy[f0:f1].mean())
        for i in range(nb):
            for k in (4, 2):
                if i + k > nb:
                    continue
                seg = range(i, i + k)
                if any(pc[j] != pc[i] for j in seg):
                    continue
                if min(float(vec[i] @ vec[j]) for j in seg) < 0.85:
                    continue
                if min(en[j] for j in seg) < 0.25 or min(share[j] for j in seg) < 0.16:
                    continue
                # 꺼지지 않는 소리: 마지막 박 세기가 첫 박의 60% 이상 (피아노는 금방 줄어든다)
                if en[i + k - 1] < 0.6 * en[i]:
                    continue
                # 중간 박에 새 어택이 크면 다른 음(반도네온·피아노 연타)
                f0, _ = _frames(fine[i], fine[i + 1])
                head = float(attack[max(0, f0 - 2):f0 + 3].max())
                inner = max(float(attack[max(0, _frames(fine[j], fine[j])[0] - 1):_frames(fine[j], fine[j])[0] + 2].max())
                            for j in range(i + 1, i + k))
                if inner > max(0.55, 1.2 * head):
                    continue
                swell = en[i + k - 1] / (en[i] + 1e-9)  # 활·비브라토는 커지기도 한다
                score = k * float(np.mean(share[i:i + k])) * float(np.mean(en[i:i + k])) * min(1.5, swell)
                if band == "low":
                    score *= 0.9
                cands.append((score, float(fine[i]), float(fine[i + k]), k))
                break  # 4박이 되면 2박은 따로 넣지 않는다
    return cands


def musical_steps(a):
    """종합: 박자 hard 리듬 + 길게 누르는 멜로디 노트(2박·4박)"""
    pulses, fine = a["pulses"], a["fine"]
    beat = float(np.median(np.diff(fine)))  # 마르카토 한 박

    def nearest_beat(t):
        return int(np.argmin(np.abs(fine - t)))

    # 바이올린·콘트라베이스·보컬처럼 버티는 음 → 박에 맞춘 2박 또는 4박 노트
    cands = melody_runs(a)
    cands.sort(reverse=True)

    # 골고루: 곡 전체 비율과 프레이즈(4마디 = 걸음 8개)마다 개수를 제한
    phrase_len = beat * 16
    budget = a["duration"] * MELODY_MAX_SHARE
    per_phrase = {}
    chosen = []
    var = a.get("variacion") or []
    for score, s, e, k in cands:
        if any(s < v1 and e > v0 for v0, v1 in var):
            continue  # 바리아시온은 16분 연타로 딛는다
        if budget <= 0:
            break
        ph = int((s - fine[0]) // phrase_len)
        if per_phrase.get(ph, 0) >= MELODY_PER_PHRASE:
            continue
        # 앞뒤 멜로디 노트와 최소 한 박은 띄운다 (그 사이 걸음이 들어간다)
        if any(s < ce + beat * 0.9 and e > cs - beat * 0.9 for cs, ce, _ in chosen):
            continue
        chosen.append((s, e, k))
        per_phrase[ph] = per_phrase.get(ph, 0) + 1
        budget -= e - s
    chosen.sort()

    out = []
    base = beat_steps_hard(a) if MUSICAL_BASE == "hard" else beat_steps(a, extras=True)
    for st in base:
        # 길게 누르는 동안(시작~끝)은 다른 발을 딛지 않는다. 끝나는 박에서 다음 걸음
        near_end = [ce for cs, ce, _ in chosen if abs(st[0] - ce) < 0.08]
        if near_end:  # 길게 누른 노트가 끝나는 박: 끝나는 순간에 맞춰 다음 발
            st = (near_end[0],) + tuple(st[1:])
        elif any(cs - MIN_GAP_MS / 1000 < st[0] < ce + MIN_GAP_MS / 1000 for cs, ce, _ in chosen):
            continue
        if out and st[0] - out[-1][0] < MIN_GAP_MS / 1000:
            continue
        out.append(st)
    for cs, ce, k in chosen:
        i = nearest_beat(cs)
        accent = any(abs(cs - p) < 0.03 and (j - a["bar_phase"]) % 2 == 0 for j, p in enumerate(pulses))
        out.append((cs, "m", accent, ce))
    out.sort(key=lambda x: x[0])
    return out, sum(ce - cs for cs, ce, _ in chosen), {2: sum(1 for c in chosen if c[2] == 2), 4: sum(1 for c in chosen if c[2] == 4)}


def check_no_overlap(notes):
    """두 발이 동시에 눌리는 곳이 없는지 (길게 누르는 노트가 끝나기 전에 다음 노트가 오면 안 된다)"""
    for a_, b_ in zip(notes, notes[1:]):
        busy_until = a_[2] or a_[0]
        if b_[0] < busy_until:
            raise RuntimeError(f"두 발이 겹침: {a_} / {b_}")
        if b_[1] == a_[1]:
            raise RuntimeError(f"같은 발 연속: {a_} / {b_}")


# ── 출력 ────────────────────────────────────────────────────
MODE_NAMES = ("beat", "beat_hard", "musical")


def chart_level(notes):
    """난이도 배수 = 1분당 판정 수 ÷ 100 (0.7~1.5, 소수 둘째 자리).
    공홈 게임(rhythm-engine.ts chartLevel)·순위 DB(_rhythm_level)와 같은 식:
    판정 수 = 노트 + 긴 노트 뗌, 길이 = 마지막 노트 끝 + 1.5초"""
    if not notes:
        return 0.7
    n = len(notes) + sum(1 for x in notes if x[2] > x[0])
    dur = max(max(x[0], x[2]) for x in notes) + 1500
    v = min(1.5, max(0.7, n * 600 / dur))
    return int(v * 100 + 0.5 + 1e-9) / 100  # 자바스크립트 Math.round 와 같게 (반올림)


def read_levels(song_dir):
    out = {}
    for m in MODE_NAMES:
        f = song_dir / f"{m}.json"
        if f.exists():
            out[m] = chart_level(json.loads(f.read_text(encoding="utf-8"))["notes"])
    return out


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ranges(items):
    """meta.json 의 구간 [["2:24", "2:47"], [30, 41.5]] → [(초, 초)]"""
    def sec(x):
        if isinstance(x, (int, float)):
            return float(x)
        m, _, s2 = str(x).strip().rpartition(":")
        return (int(m) * 60 if m else 0) + float(s2)
    out = []
    for it in items or []:
        try:
            a0, a1 = sec(it[0]), sec(it[1])
        except (TypeError, ValueError, IndexError):
            print(f"  ! 구간을 읽지 못함: {it}")
            continue
        if a1 > a0:
            out.append((a0, a1))
    return out


def process(path, out_dir, meta, catalog, seen, skip_existing):
    digest = file_md5(path)
    if digest in seen:
        print(f"  같은 음원이 이미 있어 건너뜀: {path.name} (= {seen[digest]})")
        return
    info = read_info(path)
    # 이미 목록에 있는 같은 음원은 그 id 를 그대로 쓴다 (id 를 손으로 정리해 둔 곡)
    sid = next((s["id"] for s in catalog.values() if s.get("md5") == digest), None)
    if sid is None:
        sid = slugify(f"{info['orchestra']} {info['title']}" if info["orchestra"] else info["title"])
        base = sid
        k = 2
        while sid in catalog and catalog[sid].get("md5") != digest:
            sid = f"{base}-{k}"
            k += 1
    seen[digest] = sid
    info.update({k2: v for k2, v in (meta.get(sid) or {}).items() if k2 in ("title", "orchestra", "singer", "year")})
    song_dir = out_dir / sid
    audio_name = "audio" + path.suffix.lower()
    if skip_existing and all((song_dir / f"{m}.json").exists() for m in ("beat", "beat_hard", "musical")):
        prev = catalog.get(sid, {})
        prev.update(info)
        prev["level"] = read_levels(song_dir)
        catalog[sid] = prev
        print(f"  이미 있음: {sid}")
        return
    t0 = time.time()
    a = analyze(path)
    a["variacion"] = parse_ranges((meta.get(sid) or {}).get("variacion"))
    beat = to_notes(beat_steps(a))
    beat_hard = to_notes(beat_steps_hard(a))
    mus_steps, melody_s, holds = musical_steps(a)
    musical = to_notes(mus_steps)
    check_no_overlap(beat)
    check_no_overlap(beat_hard)
    check_no_overlap(musical)
    song_dir.mkdir(parents=True, exist_ok=True)  # 분석이 된 곡만 폴더를 만든다
    shutil.copyfile(path, song_dir / audio_name)
    for mode, notes in (("beat", beat), ("beat_hard", beat_hard), ("musical", musical)):
        write_json(song_dir / f"{mode}.json", {"version": VERSION, "mode": mode, "bpm": round(a["bpm"], 1), "notes": notes})
    kinds = {k3: sum(1 for x in beat if x[3] == k3) for k3 in ("p", "h", "s")}
    hk = {k3: sum(1 for x in beat_hard if x[3] == k3) for k3 in ("p", "h", "q", "s")}
    catalog[sid] = {
        "id": sid,
        **info,
        "duration_ms": int(a["duration"] * 1000),
        "bpm": round(a["bpm"], 1),
        "audio": f"{sid}/{audio_name}",
        "modes": {"beat": f"{sid}/beat.json", "beat_hard": f"{sid}/beat_hard.json", "musical": f"{sid}/musical.json"},
        "notes": {"beat": len(beat), "beat_hard": len(beat_hard), "musical": len(musical)},
        "level": {"beat": chart_level(beat), "beat_hard": chart_level(beat_hard), "musical": chart_level(musical)},
        "md5": digest,
        "source": path.name,
    }
    print(
        f"  {sid}: 걸음 {a['bpm']:.0f}/분 · 박자 easy {len(beat)}개(기본 {kinds['p']}·반박자 {kinds['h']}·싱커 {kinds['s']})"
        f" · hard {len(beat_hard)}개(8분 {hk['h']}·16분 {hk['q']}·싱커 {hk['s']})"
        f" · 종합 {len(musical)}개(길게 2박 {holds[2]}·4박 {holds[4]}, {melody_s:.0f}초)"
        f" · {time.time() - t0:.0f}초"
    )


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="탱고 리듬 트레이너 채보 만들기 (박자·종합)")
    ap.add_argument("inputs", nargs="*", help="음원 파일이나 폴더 (비우면 이 파일 옆 '음원' 폴더)")
    ap.add_argument("--out", default=str(here / "songs"), help="결과 폴더 (기본: songs)")
    ap.add_argument("--skip-existing", action="store_true", help="이미 만든 곡은 다시 분석하지 않고 곡 정보만 고침")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] or [here / "음원"]
    files = []
    for p in inputs:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.suffix.lower() in AUDIO_EXT:
            files.append(p)
    if not files:
        print("음원을 찾지 못했습니다. '음원' 폴더에 mp3 를 넣거나 파일을 지정하세요.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    index_path = out_dir / "index.json"
    catalog = {}
    if index_path.exists():
        try:
            old = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                catalog = {s["id"]: s for s in old.get("songs", [])}
        except Exception:
            catalog = {}
    seen = {s["md5"]: s["id"] for s in catalog.values() if s.get("md5")}
    seen_this_run = {}

    print(f"{len(files)}곡 처리 → {out_dir}")
    for f in files:
        print(f"- {f.name}")
        try:
            # 이미 목록에 있는 같은 음원은 그 id 로 다시 만든다
            digest = file_md5(f)
            if digest in seen_this_run:
                print(f"  같은 음원이 이미 있어 건너뜀 (= {seen_this_run[digest]})")
                continue
            process(f, out_dir, meta, catalog, seen_this_run, args.skip_existing)
        except Exception as e:  # 한 곡이 실패해도 나머지는 계속
            print(f"  ! 실패: {e}")

    songs = sorted(catalog.values(), key=lambda s: ((s.get("orchestra") or "~").lower(), s.get("title", "").lower()))
    for s in songs:
        s.update({k: v for k, v in (meta.get(s["id"]) or {}).items() if k in ("title", "orchestra", "singer", "year")})
    write_json(index_path, {"version": VERSION, "songs": songs})
    print(f"\n완료: {len(songs)}곡 → {index_path}")
    print("songs 폴더를 GitHub 저장소에 올리면 게임에 나옵니다.")


if __name__ == "__main__":
    main()
