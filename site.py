import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
os.system('pip install scikit-learn')

from sklearn.feature_extraction.text import TfidfVectorizer

# 엑셀 파일 경로 설정 (실제 파일 경로로 변경하세요)
EXCEL_FILE_PATH = '수탐 엑셀.xlsx'

# 엑셀 파일 로드 및 데이터 전처리
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Q'] = df['Q'].fillna('')
    df['A'] = df['A'].fillna('')
    return df

df = load_data(EXCEL_FILE_PATH)

# TF-IDF 벡터화 도구 학습
@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(data['Q'].tolist())
    return vectorizer, vectors

question_vectorizer, question_vector = train_vectorizer(df)

# 유사 질문 찾기 함수
def get_most_similar_question(user_question, threshold):
    new_sen_vector = question_vectorizer.transform([user_question])
    simil_score = cosine_similarity(new_sen_vector, question_vector)
    if simil_score.max() < threshold:
        return None, "유사한 질문을 찾을 수 없습니다."
    else:
        max_index = simil_score.argmax()
        most_similar_question = df['Q'].tolist()[max_index]
        most_similar_answer = df['A'].tolist()[max_index]
        return most_similar_question, most_similar_answer

# 틱택토 게임 초기화
if 'board' not in st.session_state:
    st.session_state.board = [' ' for _ in range(9)]
    st.session_state.current_player = 'X'
    st.session_state.winner = None

# 틱택토 게임 로직
def check_winner(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 가로 승리
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 세로 승리
        (0, 4, 8), (2, 4, 6)  # 대각선 승리
    ]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != ' ':
            return board[combo[0]]
    return None

def make_move(index):
    if st.session_state.board[index] == ' ' and not st.session_state.winner:
        st.session_state.board[index] = st.session_state.current_player
        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.winner = winner
        else:
            st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
            if st.session_state.current_player == 'O':
                make_robot_move()

def make_robot_move():
    empty_cells = [i for i in range(9) if st.session_state.board[i] == ' ']
    if empty_cells:
        index = random.choice(empty_cells)
        st.session_state.board[index] = 'O'
        winner = check_winner(st.session_state.board)
        if winner:
            st.session_state.winner = winner
        st.session_state.current_player = 'X'

# Streamlit 앱 설정
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        body {{
            background-color: #f5f5dc;
            font-family: 'Nanum Gothic', sans-serif;
            color: #333333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2em;
            margin: 0 auto;
            width: 90%;
            max-width: 1200px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }}
        .content {{
            width: 100%;
            border-radius: 10px;
            padding: 2em;
            text-align: left;
            background-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .stButton button {{
            margin: 10px 0;
            width: 100%;
            border-radius: 10px;
            border: 1px solid #50c878;  /* 민트색으로 변경 */
            padding: 15px 30px;
            color: #50c878;  /* 민트색으로 변경 */
            background-color: #ffffff;
            font-size: 1.2em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .stButton button:hover {{
            background-color: #50c878;  /* 민트색으로 변경 */
            color: #ffffff;
            cursor: pointer;
        }}
        h1 {{
            color: #0ABAB5;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }}
        .header {{
            font-size: 2em;
            font-weight: bold;
            color: #0ABAB5;
            text-align: center;
            margin-bottom: 1em;
        }}
        .section {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2em;
            color: #333333;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .small-button {{
            margin: 10px 5px;
            width: auto;
            border-radius: 5px;
            border: 1px solid #50c878;  /* 민트색으로 변경 */
            padding: 8px 15px;
            color: #50c878;  /* 민트색으로 변경 */
            background-color: #ffffff;
            font-size: 0.8em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .small-button:hover {{
            background-color: #50c878;  /* 민트색으로 변경 */
            color: #ffffff;
            cursor: pointer;
        }}
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'active_button' not in st.session_state:
    st.session_state.active_button = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.title("저를 소개합니다!")

# 사이드바에 버튼 배열
st.sidebar.header("버튼을 클릭해보세요.")

large_buttons = {
    "나의 아바타": "자기소개.mp4",  # 이 경로를 실제 아바타 비디오 파일로 변경하세요.
    "나를 표현한 음악": "안녕하세요!.mp3"  # 이 경로를 실제 음악 파일로 변경하세요.
}

small_buttons = {
    "나의 장점": "누구보다 과제와 같은 활동에서 열정적인 참여도를 보여줍니다!.",
    "희망 진로": "소프트웨어 개발자가 되고 싶습니다.",
    "좋아하는 것": "게임, 친구들과 함께하는 놀이 등등 여러가 분야를 좋아합니다!.",
    "싫어하는 것": "약속을 참 소중히 생각하고 이러한 약속을 자주 깨는걸 싫어합니다.",
    "자기 소개": "안녕하세요! 반갑습니다, 저는 박지한입니다.",
    "진로 준비": "저의 진로를 위해 저는 열심히 코딩 준비를 하고있습니다.",
    "취미 활동": "저는 게임을 참 좋아하고 많이 합니다.",
    "성공 사례": "과거 열심히 노력해서 수학 1등급을 맞은 적이 있습니다."
}

for button, content in large_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None

if st.session_state.active_button == "나의 아바타":
    st.video("자기소개.mp4", format="video/mp4", start_time=0)

if st.session_state.active_button == "나를 표현한 음악":
    st.audio("안녕하세요!.mp3", format="audio/mp3")

for button, content in small_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        st.markdown(f"<div class='section'>{content}</div>", unsafe_allow_html=True)

# 유사도 임계값 슬라이더 추가
threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.43)

# 사용자 입력을 받는 입력 창
user_input = st.text_input("질문을 입력하세요:")

# 검색 버튼
if st.button("검색", key="search", help="small"):
    if user_input:
        # 유사 질문 찾기
        similar_question, answer = get_most_similar_question(user_input, threshold)
        
        if similar_question:
            st.session_state.conversation_history.append({"role": "assistant", "content": f"유사한 질문: {similar_question}"})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.write(f"**유사한 질문:** {similar_question}")
            st.write(f"**답변:** {answer}")
        else:
            st.write("유사한 질문을 찾을 수 없습니다.")

# 이전 대화 보기 버튼
if st.button("이전 대화 보기", key="view_history", help="small"):
    st.write("### 대화 기록")
    for msg in st.session_state.conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

# 새 검색 시작 버튼
if st.button("새 검색 시작", key="new_search", help="small"):
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.experimental_rerun()  # 페이지를 새로고침하여 대화 기록을 초기화

# 틱택토 게임 섹션
st.header("틱택토 게임")

# 게임 보드 UI
cols = st.columns(3)
for i in range(3):
    for j in range(3):
        index = i * 3 + j
        if cols[j].button(st.session_state.board[index], key=f'cell{index}'):
            make_move(index)

# 게임 상태 확인
if st.session_state.winner:
    st.write(f"게임 종료! 승자는 {st.session_state.winner}입니다!")
else:
    st.write(f"현재 플레이어: {st.session_state.current_player}")

# 게임 초기화 버튼
if st.button("게임 초기화"):
    st.session_state.board = [' ' for _ in range(9)]
    st.session_state.current_player = 'X'
    st.session_state.winner = None

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 묵찌빠 게임 함수
import streamlit as st
import random  # random 모듈 임포트

def play_game(user_choice):
    choices = ['가위', '바위', '보']
    computer_choice = random.choice(choices)  # 컴퓨터의 선택을 랜덤으로 정함
    
    if user_choice == computer_choice:
        result = "비겼습니다!"
    elif (user_choice == '가위' and computer_choice == '보') or \
         (user_choice == '바위' and computer_choice == '가위') or \
         (user_choice == '보' and computer_choice == '바위'):
        result = "사용자가 이겼습니다!"
    else:
        result = "컴퓨터가 이겼습니다!"
    
    return computer_choice, result

# Streamlit 앱 설정
st.title("묵찌빠 게임")

user_choice = st.selectbox("가위, 바위, 보 중 선택하세요:", ('가위', '바위', '보'))
if st.button("결과 보기"):
    computer_choice, result = play_game(user_choice)
    st.write(f"사용자 선택: {user_choice},컴퓨터 선택: {computer_choice}")
    st.write(result)