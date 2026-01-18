import streamlit as st
import pandas as pd
import json
from collections import Counter
import re

# ---------------------------------------------------------
# 1. 설정: 성경 책 이름 연결 (Alias) 및 통합 규칙
# ---------------------------------------------------------

# [추가됨] 성경 책 이름 변환 (JSON의 약어 -> 표준 이름)
# 여기에 "창": "창세기" 등을 추가하면 다른 약어도 해결됩니다.
BOOK_ALIASES = {
    "눅": "누가복음",
    # 필요하다면 아래처럼 추가 가능
    # "마": "마태복음",
    # "행": "사도행전", 
}

# (1) 단어 통합 규칙 (이 단어들은 이걸로 합친다)
MERGE_RULES = {
    "이르시되": "이르되",
    "가라사대": "이르되",
    "사람들": "사람",
    "자들": "자"
}

# (2) 무조건 제외할 단어 (Exact Match)
STOPWORDS_EXACT = {
    "위하", "것이", "너희", "너희가", "너희는", "내가", "네가",
    "그", "이", "저", "내", "네", "나", "너", "우리",
    "있다", "있는", "있어", "하니", "하나", "하라", "이에"
}

# (3) 떼어낼 조사/어미 (긴 것부터)
SUFFIXES = [
    "하사", "하시니라", "하시매", "하더라", "하니라", "하리로다", 
    "께서", "에게", "으로", "에서", "하고", "이나", "까지", "부터", "이라", "니라",
    "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "께", "여"
]

# (4) 패턴으로 제외
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
        # [수정됨] 책 이름 변환 (눅 -> 누가복음)
        # BOOK_ALIASES에 있는 이름이면 바꿔서 저장, 없으면 원래 이름 사용
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
    
    # 정렬을 위해 카테고리화 (표준 순서 적용)
    df['book'] = pd.Categorical(df['book'], categories=ALL_BOOKS_ORDER, ordered=True)
    
    # 매칭 안 되는 책(nan) 방지 및 확인
    if df['book'].isnull().any():
        unknowns = df[df['book'].isnull()]['book'].unique() # 여기선 원본값을 잃어서 확인 어려울 수 있음
        # 안전하게 정렬 수행
        df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
        # nan을 문자열 '기타' 등으로 채우거나 제거 (여기선 제거하지 않고 유지하되 맨 뒤로 보냄)
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

    # AND 검색 (A+B)
    if '+' in keyword:
        keywords = [k.strip() for k in keyword.split('+') if k.strip()]
        count = 0
        for _, row in df.iterrows():
            text = row['text']
            if all(k in text for k in keywords):
                count += 1
                # 책 이름이 혹시 nan이면 처리
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
# 4. UI 구성
# ---------------------------------------------------------
st.set_page_config(page_title="성경 데이터 분석", layout="wide")
st.title("📖 성경 빅데이터 분석기")

df = load_data("bible_data.json")

if not df.empty:
    st.sidebar.header("🔍 검색 범위 설정")
    scope = st.sidebar.radio("범위 선택", ["성경 전체", "구약만", "신약만", "책 별로 선택"])

    target_df = df.copy()
    if scope == "구약만": target_df = df[df['testament'] == "구약"]
    elif scope == "신약만": target_df = df[df['testament'] == "신약"]
    elif scope == "책 별로 선택":
        # nan 제외하고 정렬된 리스트만 표시
        valid_books = df['book'].dropna().unique()
        available_books = [b for b in ALL_BOOKS_ORDER if b in valid_books]
        sel = st.sidebar.selectbox("성경책 선택", available_books)
        target_df = df[df['book'] == sel]

    st.info(f"분석 대상: **{scope}** ({len(target_df):,} 구절)")

    tab1, tab2 = st.tabs(["📊 많이 나오는 단어 (Top 10)", "🔎 단어 찾기"])

    with tab1:
        st.subheader("가장 자주 등장하는 단어 Top 10")
        st.markdown("""
        > **ℹ️ 분석 기준 안내**
        > * **합산 카운트:** '사람들' → '사람', '이르시되' → '이르되'
        > * **불용어 제외:** '이에', '이/그/저' 등 지시대명사, '것/들/위하' 등
        """)
        
        if st.button("분석 시작", key="btn_top"):
            top_list = get_top_words_fast(target_df, 10)
            top_df = pd.DataFrame(top_list, columns=["단어", "빈도수"])
            top_df.index = top_df.index + 1
            st.table(top_df)

    with tab2:
        st.subheader("단어 빈도수 및 상세 검색")
        st.caption("💡 팁: 두 단어가 모두 들어간 구절을 찾으려면 **+**를 쓰세요. (예: **예수+사랑**)")
        
        kwd = st.text_input("검색어 입력")
        if kwd:
            cnt, vss, r_type = search_word_in_bible(target_df, kwd)
            
            if r_type == "verse":
                st.success(f"조건을 모두 만족하는 구절은 총 **{cnt}절** 발견되었습니다.")
            else:
                st.success(f"'{kwd}' 단어는 총 **{cnt}번** 등장합니다.")
            
            if vss:
                with st.expander("구절 보기"):
                    for v in vss: st.text(v)
