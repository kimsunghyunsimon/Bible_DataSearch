import streamlit as st
import pandas as pd
import json
from collections import Counter
import re

# ---------------------------------------------------------
# 1. 성경 책 순서 및 메타데이터 정의
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
# 2. 데이터 로드 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()

    rows = []
    # JSON 구조: { "창세기": { "1": { "1": {"text": "태초에...", ...} } } }
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
    # 성경 순서대로 정렬 (가나다순 X -> 창세기, 출애굽기... 순서 O)
    df['book'] = pd.Categorical(df['book'], categories=ALL_BOOKS_ORDER, ordered=True)
    df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 3. 핵심 분석 함수
# ---------------------------------------------------------
def get_top_words(df, n=10):
    """가장 많이 나온 어절 Top N 추출"""
    # 모든 텍스트를 하나로 합침
    full_text = " ".join(df['text'].tolist())
    # 특수문자 제거 및 단어 분리
    words = re.findall(r'\w+', full_text)
    # 빈도 계산
    return Counter(words).most_common(n)

def search_word_in_bible(df, keyword):
    """
    특정 단어가 포함된 횟수와 구절 찾기
    (예: '사랑' 검색 시 '사랑이', '사랑을' 모두 포함)
    """
    count = 0
    results = []
    
    keyword = keyword.strip()
    if not keyword: return 0, []

    for _, row in df.iterrows():
        text = row['text']
        # 해당 절에 키워드가 몇 번 나오는지 카운트
        c = text.count(keyword)
        if c > 0:
            count += c
            results.append(f"[{row['book']} {row['chapter']}:{row['verse']}] {text}")
            
    return count, results

# ---------------------------------------------------------
# 4. Streamlit 화면 구성 (UI)
# ---------------------------------------------------------
st.set_page_config(page_title="성경 데이터 분석", layout="wide")
st.title("📖 성경 빅데이터 분석기")

# 데이터 불러오기
df = load_data("bible_data.json")

if df.empty:
    st.error("데이터를 불러오지 못했습니다. 'bible_data.json' 파일이 같은 폴더에 있는지 확인해주세요.")
else:
    # [사이드바] 검색 범위 설정
    st.sidebar.header("🔍 검색 범위 설정")
    scope = st.sidebar.radio("범위 선택", ["성경 전체", "구약만", "신약만", "책 별로 선택"])

    target_df = df.copy()

    if scope == "구약만":
        target_df = df[df['testament'] == "구약"]
    elif scope == "신약만":
        target_df = df[df['testament'] == "신약"]
    elif scope == "책 별로 선택":
        # 현재 데이터에 있는 책 목록만 가져와서 선택 상자 표시
        available_books = [b for b in ALL_BOOKS_ORDER if b in df['book'].unique()]
        selected_book = st.sidebar.selectbox("성경책 선택", available_books)
        target_df = df[df['book'] == selected_book]

    # 현재 설정 상태 표시
    book_info = f" ({selected_book})" if scope == "책 별로 선택" else ""
    st.info(f"현재 분석 대상: **{scope}{book_info}** (총 {len(target_df):,}개의 구절)")

    # [메인] 탭 구성
    tab1, tab2 = st.tabs(["📊 많이 나오는 단어 (Top 10)", "🔎 특정 단어 찾기"])

    with tab1:
        st.subheader(f"가장 자주 등장하는 단어 Top 10")
        if st.button("분석 시작", key="btn_top"):
            with st.spinner("단어를 세는 중입니다..."):
                top_list = get_top_words(target_df, 10)
                
                # 시각화를 위한 데이터프레임 생성
                top_df = pd.DataFrame(top_list, columns=["단어", "빈도수"])
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.table(top_df)
                with col2:
                    st.bar_chart(top_df.set_index("단어"))

    with tab2:
        st.subheader("단어 빈도수 검색")
        st.caption("단어의 일부만 입력해도 포함된 모든 경우를 찾아냅니다. (예: '하나님' -> 하나님이, 하나님을, 하나님과...)")
        
        search_keyword = st.text_input("검색할 단어를 입력하세요")
        
        if search_keyword:
            total_count, verses = search_word_in_bible(target_df, search_keyword)
            
            st.success(f"검색어 '{search_keyword}'(을/를) 포함하는 단어는 총 **{total_count}번** 등장합니다.")
            
            if verses:
                with st.expander("📖 발견된 구절 보기 (클릭하세요)"):
                    for v in verses:
                        st.text(v)
