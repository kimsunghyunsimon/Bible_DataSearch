import streamlit as st
import pandas as pd
import json
from collections import Counter
import re

# ---------------------------------------------------------
# 1. 한국어 조사/어미 처리 로직 (핵심 수정 부분)
# ---------------------------------------------------------

# (1) 떼어낼 말 꼬리들 (길이 순서대로 정렬해야 긴 것부터 잘립니다)
# 여기에 계속 추가하면 '하나님께', '하나님으로' 등을 더 잘 합칠 수 있습니다.
SUFFIXES = [
    "하사", "하시니라", "하시매", "하더라", "하니라", "하리로다", # 서술격 어미
    "께서", "에게", "으로", "에서", "하고", "이나", "까지", "부터", "이라", "니라", # 긴 조사
    "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "로", "께", "여"  # 짧은 조사
]

# (2) 제외할 패턴 (앞글자 + 뒷글자 조합)
# 예: '이'로 시작하고 '것'으로 끝나는 2~3글자 -> 제거
IGNORE_STARTS = {'이', '그', '저', '내', '네', '나', '너', '우', '자', '누'}
IGNORE_ENDS = {'것', '들', '등', '중', '뿐', '쯤', '위', '가', '는', '도', '를', '은'}

def normalize_word(word):
    """
    단어의 꼬리(조사)를 자르고 기본형으로 만듭니다.
    예: 여호와께서 -> 여호와, 하나님이 -> 하나님
    """
    original_word = word
    # 길이가 2글자 이상일 때만 조사를 떼어냅니다 (한 글자 단어 보호)
    if len(word) < 2:
        return word
        
    for suffix in SUFFIXES:
        if word.endswith(suffix):
            # 조사를 뗐을 때 너무 짧아지면(1글자) 원래대로 둘지, 뗄지 결정
            # 여기서는 조사를 떼어냅니다. (예: '왕이' -> '왕')
            stem = word[:-len(suffix)]
            if len(stem) >= 1: 
                return stem
    return word

def is_stop_pattern(word):
    """
    사용자 요청 패턴 필터링:
    (이, 그, 저, 내...) + (는, 가, 것, 들...) 형태의 2~3음절 단어 제외
    """
    # 1. 길이 체크 (2~3글자)
    if len(word) in [2, 3]:
        # 2. 앞글자 체크
        if word[0] in IGNORE_STARTS:
            # 3. 뒷글자 체크 (혹은 꼬리를 뗀 상태에서도 체크)
            if word[-1] in IGNORE_ENDS:
                return True
            # 예: '그가', '이것', '저희' 등
            
    # 추가로 제외하고 싶은 특정 단어들
    if word in ["가라사대", "이르시되", "대답하여", "있느니라", "하였더라"]:
        return True
        
    return False

# ---------------------------------------------------------
# 2. 성경 메타데이터
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
# 3. 데이터 로드
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
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['book'] = pd.Categorical(df['book'], categories=ALL_BOOKS_ORDER, ordered=True)
    df = df.sort_values(by=['book', 'chapter', 'verse']).reset_index(drop=True)
    return df

# ---------------------------------------------------------
# 4. 분석 함수 (로직 적용됨)
# ---------------------------------------------------------
def get_top_words(df, n=10):
    full_text = " ".join(df['text'].tolist())
    words = re.findall(r'\w+', full_text)
    
    processed_words = []
    for w in words:
        # 1. 꼬리 자르기 (여호와께서 -> 여호와)
        stem = normalize_word(w)
        
        # 2. 패턴 필터링 (이것, 그가 -> 제외)
        if not is_stop_pattern(w) and not is_stop_pattern(stem):
            # 의미 있는 단어만 리스트에 추가
            if len(stem) > 1: # 한 글자 단어도 뺄까요? (필요시 삭제 가능)
                processed_words.append(stem)
    
    return Counter(processed_words).most_common(n)

def search_word_in_bible(df, keyword):
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
# 5. UI 구성
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
        available_books = [b for b in ALL_BOOKS_ORDER if b in df['book'].unique()]
        sel = st.sidebar.selectbox("성경책 선택", available_books)
        target_df = df[df['book'] == sel]

    st.info(f"분석 대상: **{scope}** ({len(target_df):,} 구절)")

    tab1, tab2 = st.tabs(["📊 많이 나오는 단어 (Top 10)", "🔎 단어 찾기"])

    with tab1:
        st.subheader("가장 자주 등장하는 단어 Top 10")
        st.caption("※ '여호와께서'는 '여호와'로 합치고, '이것/저것' 등은 제외했습니다.")
        
        if st.button("분석 시작", key="btn_top"):
            with st.spinner("단어 정제 및 분석 중..."):
                top_list = get_top_words(target_df, 10)
                top_df = pd.DataFrame(top_list, columns=["단어", "빈도수"])
                col1, col2 = st.columns([1, 2])
                with col1: st.table(top_df)
                with col2: st.bar_chart(top_df.set_index("단어"))

    with tab2:
        st.subheader("단어 빈도수 검색")
        kwd = st.text_input("검색어 입력")
        if kwd:
            cnt, vss = search_word_in_bible(target_df, kwd)
            st.success(f"'{kwd}' 포함 총 **{cnt}번** 등장")
            if vss:
                with st.expander("구절 보기"):
                    for v in vss: st.text(v)
