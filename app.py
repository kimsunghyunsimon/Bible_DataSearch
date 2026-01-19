import streamlit as st
import pandas as pd
import json
from collections import Counter
import re

# ---------------------------------------------------------
# 1. 설정: 성경 책 이름 연결 (Alias) 및 통합 규칙
# ---------------------------------------------------------
BOOK_ALIASES = {
    "눅": "누가복음",
    "마": "마태복음",
    "막": "마가복음",
    "요": "요한복음",
    "행": "사도행전",
    "롬": "로마서",
    "창": "창세기",
    "출": "출애굽기"
    # 필요시 계속 추가 가능
}

MERGE_RULES = {
    "이르시되": "이르되",
    "가라사대": "이르되",
    "사람들": "사람",
    "자들": "자"
}

STOPWORDS_EXACT = {
    "위하", "것이", "너희", "너희가", "너희는", "내가", "네가",
    "그", "이", "저", "내", "네", "나", "너", "우리",
    "있다", "있는", "있어", "하니", "하나", "하라", "이에"
}

SUFFIXES = [
    "하사", "하시니라", "하시매", "하더라", "하니라", "하리로다", 
    "께서", "에게", "으로", "에서", "하고", "이나", "까지", "부터", "이라", "니라",
    "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "께", "여"
]

IGNORE_STARTS = {'이', '그', '저', '내', '네', '나', '너', '우', '자', '누'}
IGNORE_ENDS = {'것', '들', '등', '중', '뿐', '쯤', '위', '가', '는', '도', '를', '은'}

def normalize_word(word):
    if len(word) < 2: return word
    for suffix in SUFFIXES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if len(stem) >= 1: return stem
    return word

def is_stop_pattern(word):
    if len(word) not in [2, 3]: return False
    if "너희" in word or "위하" in word: return True
    if word[0] in IGNORE_STARTS and word[-1] in IGNORE_ENDS: return True
    return False

# ---------------------------------------------------------
# 2. 데이터 로드 및 전처리
# ---------------------------------------------------------
OT_BOOKS = [
    "창세기", "출애굽기", "레위기", "민수기", "신명기", "여호수아", "사사기", "룻기",
    "사무엘상", "사무엘하", "열왕기상", "열왕기하", "역대상", "역대하", "에스라", "느헤미야",
    "에스더", "욥기", "시편", "잠언", "전도서", "아가", "이사야", "예레미야",
    "예레미야애가", "에스겔", "다니엘", "호세아", "요엘", "아모스", "오바댜", "요나",
    "미가", "나훔", "하박국", "스바냐", "학개", "스가랴", "말라기"
]
NT_BOOKS = [
    "마태복음", "마가복음", "누가복음", "요한복음", "사도행전", "로마서", "고린도전서", "고린도후서",
    "갈라디아서", "에베소서", "빌립보서", "골로새서", "데살로니가전서", "데살로니가후서", "디모데전서", "디모데후서",
    "디도서", "빌레몬서", "히브리서", "야고보서", "베드로전서", "베드로후서", "요한일서", "요한이서",
    "요한삼서", "유다서", "요한계시록"
]
ALL_BOOKS_ORDER = OT_BOOKS + NT_BOOKS

@st.cache_data
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError: return pd.DataFrame()

    rows = []
    for book, chapters in data.items():
        normalized_book_name = BOOK_ALIASES.get(book, book)
        for chapter, verses in chapters.items():
            for verse, content in verses.items():
                rows.append({
                    "book": normalized_book_name,
                    "chapter": int(chapter),
                    "verse": int(verse),
                    "text": content.get("text", ""),
                    "testament": "구약" if normalized_book_name in OT_BOOKS else ("신약" if normalized_book_name in NT_BOOKS else "기타")
                })
                
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['book'] = pd.Categorical(df['book'], categories=ALL_BOOKS_ORDER, ordered=True)
    
    if df['book'].isnull().any():
        df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
        
    return df

# ---------------------------------------------------------
# 3. 핵심 분석 함수
# ---------------------------------------------------------
def get_top_words_fast(df, n=10):
    full_text = " ".join(df['text'].tolist())
    raw_words = re.findall(r'\w+', full_text)
    raw_counter = Counter(raw_words)
    final_counter = Counter()
    
    for word, count in raw_counter.items():
        if word in MERGE_RULES:
            target_word = MERGE_RULES[word]
            final_counter[target_word] += count
            continue
        stem = normalize_word(word)
        if stem in MERGE_RULES:
            target_word = MERGE_RULES[stem]
            final_counter[target_word] += count
            continue
        if stem in STOPWORDS_EXACT or is_stop_pattern(stem) or is_stop_pattern(word):
            continue
        if len(stem) > 1:
            final_counter[stem] += count
            
    return final_counter.most_common(n)

def search_word_in_bible(df, keyword):
    keyword = keyword.strip()
    if not keyword: return 0, [], ""
    
    results = []
    # AND 검색
    if '+' in keyword:
        keywords = [k.strip() for k in keyword.split('+') if k.strip()]
        count = 0
        for _, row in df.iterrows():
            text = row['text']
            if all(k in text for k in keywords):
                count += 1
                book_name = row['book'] if pd.notna(row['book']) else "알수없음"
                results.append(f"[{book_name} {row['chapter']}:{row['verse']}] {text}")
        return count, results, "verse"
    # 단일 검색
    else:
        count = 0
        for _, row in df.iterrows():
            text = row['text']
            c = text.count(keyword)
            if c > 0:
                count += c
                book_name = row['book'] if pd.notna(row['book']) else "알수없음"
                results.append(f"[{book_name} {row['chapter']}:{row['verse']}] {text}")
        return count, results, "word"

# ---------------------------------------------------------
# 4. UI 구성 (모바일 친화적 상단 배치)
# ---------------------------------------------------------
st.set_page_config(page_title="성경 데이터 분석", layout="wide")
st.title("📖 성경 빅데이터 분석기")

df = load_data("bible_data.json")

if not df.empty:
    # [수정됨] 사이드바(st.sidebar) 대신 메인 화면 상단에 배치
    # 가로형(horizontal=True)으로 배치하여 모바일에서 공간 절약
    st.write("### 🔍 검색 범위 설정")
    
    scope = st.radio(
        "분석할 범위를 선택하세요:", 
        ["성경 전체", "구약만", "신약만", "책 별로 선택"], 
        horizontal=True
    )

    target_df = df.copy()
    if scope == "구약만": target_df = df[df['testament'] == "구약"]
    elif scope == "신약만": target_df = df[df['testament'] == "신약"]
    elif scope == "책 별로 선택":
        valid_books = df['book'].dropna().unique()
        available_books = [b for b in ALL_BOOKS_ORDER if b in valid_books]
        # 책 선택 메뉴도 바로 아래에 배치
        sel = st.selectbox("성경책을 선택하세요:", available_books)
        target_df = df[df['book'] == sel]

    st.markdown("---") # 구분선
    st.info(f"📊 현재 분석 대상: **{scope}** (총 {len(target_df):,} 구절)")

    # 탭 구성
    tab1, tab2 = st.tabs(["🏆 Top 10 단어", "🔎 단어 검색"])

    with tab1:
        st.subheader("가장 자주 등장하는 단어 Top 10")
        st.markdown("""
        <small>ℹ️ '사람들'→'사람' 통합 / '이에', '이/그/저' 등 불용어 제외</small>
        """, unsafe_allow_html=True)
        
        if st.button("분석 시작", key="btn_top", type="primary"):
            top_list = get_top_words_fast(target_df, 10)
            top_df = pd.DataFrame(top_list, columns=["단어", "빈도수"])
            top_df.index = top_df.index + 1
            st.table(top_df)

    with tab2:
        st.subheader("단어 빈도수 및 상세 검색")
        st.caption("팁: '예수+사랑' 처럼 입력하면 두 단어가 모두 있는 구절을 찾습니다.")
        
        kwd = st.text_input("검색어 입력 (엔터)")
        if kwd:
            cnt, vss, r_type = search_word_in_bible(target_df, kwd)
            
            if r_type == "verse":
                st.success(f"조건 만족 구절: **{cnt}절**")
            else:
                st.success(f"등장 횟수: **{cnt}번**")
            
            if vss:
                with st.expander("구절 보기 (클릭)", expanded=True):
                    for v in vss: st.text(v)
