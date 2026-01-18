import streamlit as st
import pandas as pd
import json
from collections import Counter
import re

# ---------------------------------------------------------
# 1. 불용어 (분석에서 제외할 단어들) 설정
# ---------------------------------------------------------
# 여기에 제외하고 싶은 단어를 계속 추가하시면 됩니다.
STOPWORDS = {
    "이", "그", "저", "것", "수", "등", "들", "및", "곧", "또",
    "내가", "그의", "그가", "그들이",  "나를",
    "내", "네", "나", "너", "우리", "저희", "너희", "당신",
    "가", "이", "은", "는", "을", "를", "의", "에게", "께", "와", "과", # 조사(완벽 분리는 안되지만 띄어쓰기 된 경우)
    "가라사대", "이르시되", "대답하여", "하더라", "하니라", "하시니라" # 성경 투의 접속/서술어
}

# ---------------------------------------------------------
# 2. 성경 책 순서 및 메타데이터 정의
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

def get_testament(book_name):
    if book_name in OT_BOOKS: return "구약"
    elif book_name in NT_BOOKS: return "신약"
    return "기타"

# ---------------------------------------------------------
# 3. 데이터 로드 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()

    rows = []
    for book, chapters in data.items():
        for chapter, verses in chapters.items():
            for verse, content in verses.items():
                text = content.get("text", "")
                rows.append({
                    "book": book,
                    "chapter": int(chapter),
                    "verse": int(verse),
                    "text": text,
                    "testament": get_testament(book)
                })
    
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['book'] = pd.Categorical(df['book'], categories=ALL_BOOKS_ORDER, ordered=True)
    df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 4. 핵심 분석 함수 (수정됨)
# ---------------------------------------------------------
def get_top_words(df, n=10):
    """불용어를 제외하고 가장 많이 나온 단어 추출"""
    full_text = " ".join(df['text'].tolist())
    words = re.findall(r'\w+', full_text)
    
    # [수정된 부분] 불용어 목록(STOPWORDS)에 없는 단어만 남깁니다.
    meaningful_words = [w for w in words if w not in STOPWORDS]
    
    return Counter(meaningful_words).most_common(n)

def search_word_in_bible(df, keyword):
    """특정 단어 포함 검색"""
    count = 0
    results = []
    keyword = keyword.strip()
    if not keyword: return 0, []

    for _, row in df.iterrows():
        text = row['text']
        c = text.count(keyword)
        if c > 0:
            count += c
            results.append(f"[{row['book']} {row['chapter']}:{row['verse']}] {text}")
    return count, results

# ---------------------------------------------------------
# 5. Streamlit 화면 구성
# ---------------------------------------------------------
st.set_page_config(page_title="성경 데이터 분석", layout="wide")
st.title("📖 성경 빅데이터 분석기")

df = load_data("bible_data.json")

if df.empty:
    st.error("데이터를 불러오지 못했습니다. bible_data.json 파일을 확인해주세요.")
else:
    # 사이드바
    st.sidebar.header("🔍 검색 범위 설정")
    scope = st.sidebar.radio("범위 선택", ["성경 전체", "구약만", "신약만", "책 별로 선택"])

    target_df = df.copy()

    if scope == "구약만":
        target_df = df[df['testament'] == "구약"]
    elif scope == "신약만":
        target_df = df[df['testament'] == "신약"]
    elif scope == "책 별로 선택":
        available_books = [b for b in ALL_BOOKS_ORDER if b in df['book'].unique()]
        selected_book = st.sidebar.selectbox("성경책 선택", available_books)
        target_df = df[df['book'] == selected_book]

    book_info = f" ({selected_book})" if scope == "책 별로 선택" else ""
    st.info(f"현재 분석 대상: **{scope}{book_info}** (총 {len(target_df):,}개의 구절)")

    # 탭 구성
    tab1, tab2 = st.tabs(["📊 많이 나오는 단어 (Top 10)", "🔎 특정 단어 찾기"])

    with tab1:
        st.subheader(f"가장 자주 등장하는 단어 Top 10 (불용어 제외)")
        
        # 사용자 편의를 위해 제거된 단어 목록을 살짝 보여줍니다.
        with st.expander("ℹ️ 현재 통계에서 제외된 단어들 확인"):
            st.write(", ".join(sorted(STOPWORDS)))

        if st.button("분석 시작", key="btn_top"):
            with st.spinner("단어를 세는 중입니다..."):
                top_list = get_top_words(target_df, 10)
                top_df = pd.DataFrame(top_list, columns=["단어", "빈도수"])
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.table(top_df)
                with col2:
                    st.bar_chart(top_df.set_index("단어"))

    with tab2:
        st.subheader("단어 빈도수 검색")
        search_keyword = st.text_input("검색할 단어를 입력하세요")
        if search_keyword:
            total_count, verses = search_word_in_bible(target_df, search_keyword)
            st.success(f"검색어 '{search_keyword}'(을/를) 포함하는 단어는 총 **{total_count}번** 등장합니다.")
            if verses:
                with st.expander("📖 발견된 구절 보기"):
                    for v in verses:
                        st.text(v)
