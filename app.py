import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import google.generativeai as genai
import datetime
import json
import io

# ==========================================
# 1. 페이지 및 세션 기본 설정
# ==========================================
st.set_page_config(
    page_title="스마트 인강실 통합 관리 시스템",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API 설정
def init_gemini():
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "gemini_key" in st.session_state and st.session_state.gemini_key:
        api_key = st.session_state.gemini_key
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Gemini API 설정 오류: {e}")
            return False
    return False

# Session State 초기화
if "user_role" not in st.session_state:
    st.session_state.user_role = None  # 'student', 'teacher', or None
if "teacher_password" not in st.session_state:
    st.session_state.teacher_password = "scs12345"
if "volume_threshold" not in st.session_state:
    st.session_state.volume_threshold = "50%"  # 기본 50%

# 더미 데이터 초기화
if "attendance_logs" not in st.session_state:
    st.session_state.attendance_logs = [
        {"학번": "20240101", "이름": "김철수", "구분": "입장", "시각": "09:00:15", "상태": "정상"},
        {"학번": "20240102", "이름": "이영희", "구분": "입장", "시각": "09:05:22", "상태": "정상"},
        {"학번": "20240101", "이름": "김철수", "구분": "퇴장", "시각": "10:30:40", "상태": "동일인 확인(정상)"},
        {"학번": "20240103", "이름": "박민수", "구분": "입장", "시각": "10:00:00", "상태": "정상"},
        {"학번": "20240103", "이름": "박영수", "구분": "퇴장", "시각": "11:15:10", "상태": "⚠️ 이름 불일치"},
    ]

if "applications" not in st.session_state:
    st.session_state.applications = [
        {"id": 1, "학번": "20240101", "이름": "김철수", "강좌명": "개념원리 수학 I", "수강기간": "2026.11.01 ~ 2026.12.31", "상태": "자동 승인", "요약": "정상 신청서 형식"},
        {"id": 2, "학번": "20240102", "이름": "이영희", "강좌명": "EBS 수능특강 영어", "수강기간": "2026.11.05 ~ 2026.12.20", "상태": "승인 대기", "요약": "기간 재확인 필요"},
    ]

if "notes" not in st.session_state:
    st.session_state.notes = [
        {
            "id": 1, "학번": "20240101", "이름": "김철수", "학년": "1학년", "과목": "수학 (공통수학1)",
            "제출시각": "10:32", "분량점수": 85, "개념연관성": 90, "판정": "자동 승인",
            "AI분석": "공통수학1 다항식 연산 개념 정리 양호. 분량 충족.", "이미지": None
        },
        {
            "id": 2, "학번": "20240102", "이름": "이영희", "학년": "2학년", "과목": "탐구(일반) - 생명과학",
            "제출시각": "11:05", "분량점수": 45, "개념연관성": 60, "판정": "애매함 (선생님 검토 필요)",
            "AI분석": "세포 호흡 단원 작성이나 작성 분량이 다소 부족함.", "이미지": None
        }
    ]

# ==========================================
# 2. 과목 데이터 구조
# ==========================================
SUBJECTS_DATA = {
    "1학년": {
        "국어": ["공통국어1", "공통국어2"],
        "수학": ["공통수학1", "공통수학2"],
        "영어": ["영어"],
        "한국사": ["한국사"],
        "통합사회": ["통합사회1", "통합사회2"],
        "통합과학": ["통합과학1", "통합과학2"]
    },
    "2학년": {
        "국어": ["화법과 언어", "독서와 작문", "문학"],
        "수학": ["대수", "미적분 1", "확률과 통계", "미적분 2", "기하"],
        "영어": ["영어"],
        "탐구(일반선택)": ["세계시민과 지리", "사회와 문화", "현대시민과 윤리", "세계사", "물리학", "화학", "생명과학", "지구과학"],
        "탐구(진로선택)": [
            "한국지리 탐구", "윤리와 사상", "정치", "경제", "법과 사회", "동아시아 역사 기행",
            "역학과 에너지", "전자기와 양자", "물질과 에너지", "화학 반응의 세계", "세포와 물질대사", "생물의 유전", "지구시스템과학", "행성우주과학"
        ]
    }
}

# ==========================================
# 3. Gemini API 연동 함수
# ==========================================
def analyze_application_image(image):
    """신청서 스캔 이미지 Gemini OCR 및 분석"""
    if not init_gemini():
        return {"강좌명": "Gemini API 미설정 (수동 입력 필요)", "수강기간": "미상", "요약": "API 키를 설정해주세요."}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        이 이미지는 인강 수강 신청서입니다.
        다음 정보를 JSON 형식으로 추출해 주세요:
        {
          "강좌명": "추출된 강좌명",
          "수강기간": "추출된 수강 기간",
          "요약": "신청서 상태 요약 (정상/식별불가 등)"
        }
        JSON 외의 다른 말은 하지 마세요.
        """
        response = model.generate_content([prompt, image])
        clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data
    except Exception as e:
        return {"강좌명": "자동 추출 실패", "수강기간": "미상", "요약": f"오류: {str(e)}"}

def analyze_note_image(image, grade, subject_full, volume_threshold_str):
    """인강 노트 Gemini AI 분량 및 개념 연관성 검수"""
    if not init_gemini():
        # Fallback Mock logic
        return {
            "분량점수": 70,
            "개념연관성": 80,
            "판정": "애매함 (선생님 검토 필요)",
            "상세이유": "Gemini API 키가 설정되지 않아 임시 테스트 결과가 생성되었습니다. (API 키 설정 필요)"
        }
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        threshold_num = int(volume_threshold_str.replace("%", ""))
        
        prompt = f"""
        당신은 학교 인강실 교사 보조 AI 검수 시스템입니다.
        학생이 제출한 필기 노트 이미지를 분석해주세요.
        
        [검수 기준]
        - 학생 학년/과목: {grade} - {subject_full}
        - 선생님이 설정한 최소 요구 분량 기준: {volume_threshold_str}
        
        [분석 요청]
        1. 분량점수 (0~100): 페이지 채움 정도, 작성 밀도, 글자 수 평가
        2. 개념연관성 (0~100): 해당 과목({subject_full})의 주요 개념/키워드가 제대로 포함되어 있는지 평가
        3. 판정결과: 
           - '자동 승인': 분량 점수가 기준({threshold_num}%) 이상이고 개념 연관성이 70점 이상인 경우
           - '애매함 (선생님 검토 필요)': 분량이 기준 근처(±15%)이거나, 필기 식별이 다소 모호한 경우
           - '반려': 분량이 기준보다 현저히 미달하거나 과목 내용과 전혀 무관한 경우
        4. 상세이유: 2~3줄로 구체적인 요약 설명 작성
        
        반드시 아래 JSON 형식으로만 답하세요:
        {{
          "분량점수": 80,
          "개념연관성": 85,
          "판정": "자동 승인",
          "상세이유": "개념 정리가 충실하며 선생님이 설정한 분량 기준을 상회합니다."
        }}
        """
        response = model.generate_content([prompt, image])
        clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return data
    except Exception as e:
        return {
            "분량점수": 50,
            "개념연관성": 50,
            "판정": "애매함 (선생님 검토 필요)",
            "상세이유": f"Gemini 분석 중 오류 발생: {str(e)}"
        }

# ==========================================
# 4. 사용자 인증 / 역할 선택 화면
# ==========================================
if st.session_state.user_role is None:
    st.title("🏫 스마트 인강실 통합 관리 시스템")
    st.subheader("접속할 서비스 역할을 선택해 주세요.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 👨‍🎓 학생용 서비스")
        st.write("- 출결 체크 (입/퇴실)")
        st.write("- 수강 신청서 카메라 스캔 제출")
        st.write("- 인강 필기 노트 사진 스캔 제출")
        if st.button("학생으로 시작하기", use_container_width=True, type="primary"):
            st.session_state.user_role = "student"
            st.rerun()

    with col2:
        st.warning("### 🧑‍🏫 선생님용 서비스")
        st.write("- 실시간 출결 및 동일인 확인")
        st.write("- AI 기반 인강 노트 자동/애매 검수")
        st.write("- 분량 검수 기준 설정 (25%~100%)")
        
        pwd_input = st.text_input("선생님 비밀번호 입력", type="password", key="login_pwd")
        if st.button("선생님으로 로그인", use_container_width=True):
            if pwd_input == st.session_state.teacher_password:
                st.session_state.user_role = "teacher"
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다. (초기 비밀번호: scs12345)")
    st.stop()

# 사이드바 역할 전환 및 기본 정보
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    role_label = "👨‍🎓 학생 모드" if st.session_state.user_role == "student" else "🧑‍🏫 선생님 모드"
    st.success(f"현재 접속: **{role_label}**")
    
    if st.button("역할 변경 (초기화)"):
        st.session_state.user_role = None
        st.rerun()
        
    st.divider()
    
    # Gemini API Key 입력 (Secrets 미설정 시 보완용)
    if "GEMINI_API_KEY" not in st.secrets:
        st.subheader("🔑 Gemini API 키 설정")
        user_key = st.text_input("API Key 입력", type="password", value=st.session_state.get("gemini_key", ""))
        if user_key:
            st.session_state.gemini_key = user_key
            st.caption("API Key가 세션에 저장되었습니다.")
    else:
        st.caption("✅ Streamlit Secrets에 API Key가 설정되어 있습니다.")

# ==========================================
# 5. 학생용 인터페이스 (Student Interface)
# ==========================================
if st.session_state.user_role == "student":
    st.title("👨‍🎓 학생용 인강실 관리 시스템")
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ 입·퇴실 출결 체크", "2️⃣ 수강 신청서 스캔", "3️⃣ 인강 노트 스캔 제출"])
    
    # [1단계] 출결 체크
    with tab1:
        st.header("📌 1단계: 입·퇴실 출결 체크")
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            check_type = st.radio("구분 선택", ["입장", "퇴장"], horizontal=True)
            student_id = st.text_input("8자리 학번 입력", max_chars=8, placeholder="예: 20261234")
            student_name = st.text_input("이름 입력", placeholder="예: 홍길동")
            
            if st.button("출결 입력 확인", type="primary"):
                if len(student_id) != 8 or not student_id.isdigit():
                    st.error("학번은 반드시 8자리 숫자로 입력해 주세요.")
                elif not student_name.strip():
                    st.error("이름을 입력해 주세요.")
                else:
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    status = "정상"
                    
                    if check_type == "퇴장":
                        # 이전 입장 기록 검증
                        entries = [log for log in st.session_state.attendance_logs if log["학번"] == student_id and log["구분"] == "입장"]
                        if entries:
                            last_entry = entries[-1]
                            if last_entry["이름"] == student_name.strip():
                                status = "동일인 확인(정상)"
                            else:
                                status = "⚠️ 이름 불일치"
                        else:
                            status = "⚠️ 입장 기록 없음"
                    
                    st.session_state.attendance_logs.append({
                        "학번": student_id,
                        "이름": student_name.strip(),
                        "구분": check_type,
                        "시각": now_str,
                        "상태": status
                    })
                    st.success(f"[{check_type}] 처리 완료: {student_name} ({student_id}) - {now_str}")
        
        with col_c2:
            st.subheader("📋 최근 출결 기록")
            df_att = pd.DataFrame(st.session_state.attendance_logs)
            st.dataframe(df_att.tail(5), use_container_width=True)

    # [2단계] 수강 신청서 스캔
    with tab2:
        st.header("📌 2단계: 인강 수강 신청서 스캔 등록")
        st.write("카메라로 촬영하거나 신청서 이미지를 업로드하여 자동 승인을 요청하세요.")
        
        app_img_file = st.file_uploader("신청서 이미지/사진 선택", type=["png", "jpg", "jpeg"], key="app_uploader")
        camera_app = st.camera_input("카메라로 신청서 촬영", key="app_camera")
        
        target_img = camera_app if camera_app is not None else app_img_file
        
        if target_img:
            img = Image.open(target_img)
            st.image(img, caption="스캔된 신청서 이미지", width=350)
            
            if st.button("Gemini AI 신청서 스캔 및 승인 요청", type="primary"):
                with st.spinner("Gemini API로 신청서 양식 분석 중..."):
                    res = analyze_application_image(img)
                    new_app = {
                        "id": len(st.session_state.applications) + 1,
                        "학번": st.session_state.attendance_logs[-1]["학번"] if st.session_state.attendance_logs else "20260000",
                        "이름": st.session_state.attendance_logs[-1]["이름"] if st.session_state.attendance_logs else "학생",
                        "강좌명": res.get("강좌명", "미상"),
                        "수강기간": res.get("수강기간", "미상"),
                        "상태": "승인 대기",
                        "요약": res.get("요약", "분석 완료")
                    }
                    st.session_state.applications.append(new_app)
                    st.success("신청서가 성공적으로 접수되었습니다. 선생님 승인을 대기 중입니다.")
                    st.json(res)

    # [3단계] 인강 노트 스캔 제출
    with tab3:
        st.header("📌 3단계: 인강 노트 스캔 제출")
        
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            s_grade = st.selectbox("학년 선택", ["1학년", "2학년"])
            
            # 학년별 과목 계층 선택
            sub_categories = list(SUBJECTS_DATA[s_grade].keys())
            s_cat = st.selectbox("과목 분류 선택", sub_categories)
            
            detail_subjects = SUBJECTS_DATA[s_grade][s_cat]
            s_detail = st.selectbox("세부 과목 선택", detail_subjects)
            
            full_subject_name = f"{s_cat} - {s_detail}"
            st.info(f"선택한 과목: **[{s_grade}] {full_subject_name}**")

        with col_n2:
            st.write("📸 **노트 스캔 촬영 / 이미지 업로드**")
            note_img_file = st.file_uploader("노트 이미지 선택", type=["png", "jpg", "jpeg"], key="note_uploader")
            note_camera = st.camera_input("노트 직접 촬영", key="note_camera")
            
            target_note_img = note_camera if note_camera is not None else note_img_file

        if target_note_img and st.button("🚀 노트 스캔 사진 시작 (AI 검수 제출)", type="primary", use_container_width=True):
            img = Image.open(target_note_img)
            
            with st.spinner(f"선생님 설정 기준({st.session_state.volume_threshold}) 및 과목 적합성을 Gemini AI가 검수 중..."):
                res = analyze_note_image(img, s_grade, full_subject_name, st.session_state.volume_threshold)
                
                std_id = st.session_state.attendance_logs[-1]["학번"] if st.session_state.attendance_logs else "20261001"
                std_name = st.session_state.attendance_logs[-1]["이름"] if st.session_state.attendance_logs else "김학생"
                
                new_note = {
                    "id": len(st.session_state.notes) + 1,
                    "학번": std_id,
                    "이름": std_name,
                    "학년": s_grade,
                    "과목": full_subject_name,
                    "제출시각": datetime.datetime.now().strftime("%H:%M"),
                    "분량점수": res["분량점수"],
                    "개념연관성": res["개념연관성"],
                    "판정": res["판정"],
                    "AI분석": res["상세이유"],
                    "이미지": img
                }
                st.session_state.notes.append(new_note)
                
                st.divider()
                if res["판정"] == "자동 승인":
                    st.balloons()
                    st.success(f"🎉 **자동 승인 완료!** (분량점수: {res['분량점수']}%, 연관성: {res['개념연관성']}%)")
                elif "애매함" in res["판정"]:
                    st.warning(f"⚠️ **검토 대기 (애매함)**: {res['상세이유']}")
                else:
                    st.error(f"❌ **반려됨**: {res['상세이유']}")

# ==========================================
# 6. 선생님용 인터페이스 (Teacher Interface)
# ==========================================
else:
    st.title("🧑‍🏫 선생님용 인강실 관리 대시보드")
    
    # 1. 선생님 전용 설정 (기준 선택 및 비밀번호 변경)
    with st.expander("🛠️ 선생님 관리 설정 (분량 기준 설정 & 비밀번호 변경)", expanded=False):
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.subheader("📏 분량 검수 기준 선택")
            selected_th = st.radio(
                "AI 노트 자동 검수 분량 기준",
                ["25%", "50%", "75%", "100%"],
                index=["25%", "50%", "75%", "100%"].index(st.session_state.volume_threshold),
                horizontal=True
            )
            if selected_th != st.session_state.volume_threshold:
                st.session_state.volume_threshold = selected_th
                st.success(f"분량 기준이 **{selected_th}**로 변경되었습니다.")
                
        with col_s2:
            st.subheader("🔒 선생님 비밀번호 변경")
            cur_pwd = st.text_input("현재 비밀번호", type="password", key="c_pwd")
            new_pwd = st.text_input("새 비밀번호", type="password", key="n_pwd")
            cfm_pwd = st.text_input("새 비밀번호 확인", type="password", key="cfm_pwd")
            
            if st.button("비밀번호 변경 완료"):
                if cur_pwd != st.session_state.teacher_password:
                    st.error("현재 비밀번호가 일치하지 않습니다.")
                elif not new_pwd:
                    st.error("새 비밀번호를 입력해 주세요.")
                elif new_pwd != cfm_pwd:
                    st.error("새 비밀번호와 확인 입력이 일치하지 않습니다.")
                else:
                    st.session_state.teacher_password = new_pwd
                    st.success("비밀번호가 성공적으로 변경되었습니다!")

    # 2. 통계 대시보드 (Plotly 시각화)
    st.subheader("📊 실시간 인강실 현황 데이터")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 출결 기록", len(st.session_state.attendance_logs))
    c2.metric("신청서 승인 대기", len([a for a in st.session_state.applications if a["상태"] == "승인 대기"]))
    c3.metric("애매한 노트 (수동 검토)", len([n for n in st.session_state.notes if "애매함" in n["판정"]]))
    c4.metric("현재 설정 분량 기준", st.session_state.volume_threshold)

    # Plotly 시각화 그래프
    if st.session_state.notes:
        df_notes = pd.DataFrame(st.session_state.notes)
        fig_notes = px.pie(df_notes, names="판정", title="인강 노트 AI 검수 결과 분포", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_notes.update_layout(height=300)
        st.plotly_chart(fig_notes, use_container_width=True)

    t_tab1, t_tab2, t_tab3 = st.tabs(["📋 출결 관리 모니터링", "📝 수강 신청서 승인", "🔍 애매한 노트 정밀 검수"])
    
    # [Tab 1] 출결 관리
    with t_tab1:
        st.header("📋 학생 출결 및 동일인 검증 현황")
        df_att = pd.DataFrame(st.session_state.attendance_logs)
        st.dataframe(df_att, use_container_width=True)
        
        # 이름 불일치 경고
        alerts = df_att[df_att["상태"].str.contains("⚠️")]
        if not alerts.empty:
            st.error("⚠️ 출결 이상 감지 내역:")
            st.dataframe(alerts, use_container_width=True)

    # [Tab 2] 수강 신청서 승인
    with t_tab2:
        st.header("📝 수강 신청서 승인 처리")
        for idx, app in enumerate(st.session_state.applications):
            with st.container():
                st.write(f"**[{app['상태']}] {app['이름']}({app['학번']}) - {app['강좌명']}** ({app['수강기간']})")
                st.caption(f"요약: {app['요약']}")
                
                btn_col1, btn_col2, _ = st.columns([1, 1, 4])
                if app["상태"] == "승인 대기":
                    if btn_col1.button(f"승인", key=f"app_approve_{idx}"):
                        st.session_state.applications[idx]["상태"] = "승인 완료"
                        st.rerun()
                    if btn_col2.button(f"반려", key=f"app_reject_{idx}"):
                        st.session_state.applications[idx]["상태"] = "반려됨"
                        st.rerun()
            st.divider()

    # [Tab 3] 애매한 노트 검수
    with t_tab3:
        st.header("🔍 애매한 노트 정밀 검수 (선생님 직접 확인)")
        st.info("AI 검수 결과 판정이 보류되었거나 '애매함' 상태인 노트를 직접 확인하여 승인/반려합니다.")
        
        ambiguous_notes = [n for n in st.session_state.notes if "애매함" in n["판정"]]
        
        if not ambiguous_notes:
            st.success("현재 수동 검토가 필요한 애매한 노트가 없습니다.")
        else:
            for note in ambiguous_notes:
                with st.expander(f"⚠️ [{note['학번']}] {note['이름']} - {note['과목']} (제출: {note['제출시각']})", expanded=True):
                    col_img, col_detail = st.columns([1, 1])
                    
                    with col_img:
                        if note["이미지"]:
                            st.image(note["이미지"], caption="스캔된 노트 이미지", use_container_width=True)
                        else:
                            st.info("스캔 이미지 샘플이 표시됩니다.")
                            
                    with col_detail:
                        st.write(f"**분량 점수:** {note['분량점수']}% (현재 선생님 기준: {st.session_state.volume_threshold})")
                        st.progress(note['분량점수'] / 100)
                        
                        st.write(f"**과목 개념 연관성:** {note['개념연관성']}%")
                        st.progress(note['개념연관성'] / 100)
                        
                        st.write(f"**Gemini AI 분석 의견:**")
                        st.warning(note["AI분석"])
                        
                        ac_col1, ac_col2 = st.columns(2)
                        if ac_col1.button("✅ 최종 승인 처리", key=f"pass_{note['id']}", type="primary"):
                            note["판정"] = "선생님 수동 승인"
                            st.success("승인 처리되었습니다.")
                            st.rerun()
                        if ac_col2.button("❌ 최종 반려 처리", key=f"fail_{note['id']}"):
                            note["판정"] = "선생님 수동 반려"
                            st.error("반려 처리되었습니다.")
                            st.rerun()