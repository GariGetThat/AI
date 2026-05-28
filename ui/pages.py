# ui/pages.py

from __future__ import annotations

import time

import streamlit as st

from ui.components import (
    header,
    step_badge,
    page_title,
    section_label,
    hint_box,
    status_success,
    status_running,
    chip_row,
    feature_card,
    pipeline_viz,
    json_panel,
    target_item,
    person_card,
    preview_placeholder,
    detection_preview,
    timeline_row,
    next_button,
)


DUMMY_TARGETS = {
    "targets": [
        {"type": "face", "id": "person_A", "bbox": [100, 120, 220, 360]},
        {"type": "face", "id": "person_B", "bbox": [260, 110, 380, 350]},
        {"type": "object", "id": "sign_001", "label": "간판", "bbox": [40, 400, 180, 460]},
        {"type": "object", "id": "building_001", "label": "건물명", "bbox": [230, 380, 390, 430]},
        {"type": "object", "id": "parcel_001", "label": "택배 정보", "bbox": [450, 410, 620, 470]},
    ]
}


def landing_page() -> None:
    header()

    st.markdown(
        """
        <div style="
            background:#F5F5F5;
            border-radius:0;
            padding:46px 40px 54px 40px;
            margin-bottom:34px;
            text-align:center;
        ">
            <div style="
                display:inline-block;
                background:#8C1F2E;
                color:white;
                border-radius:16px;
                padding:8px 18px;
                font-size:13px;
                font-weight:700;
                margin-bottom:22px;
            ">
                자연어 기반 · 병렬 탐지 · 통합 블러 — 맥락 이해 프라이버시 보호 AI
            </div>
            <div style="
                font-size:40px;
                font-weight:700;
                line-height:1.25;
                color:#212121;
                margin-bottom:18px;
            ">
                자연어로 말하면, AI가 알아서 처리합니다
            </div>
            <div style="
                font-size:15px;
                color:#737373;
                margin-bottom:28px;
            ">
                얼굴과 간판·건물명·택배 정보까지 — 자연어 명령 하나로 병렬 탐지하고 단일 패스 마스크 생성 후 Gaussian Blur를 적용합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("지금 시작하기 →"):
        st.session_state.page = "upload"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:#737373;font-size:12px;margin-bottom:18px;">AI 처리 6단계 흐름</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("①", "자연어 입력", "프롬프트 기반 명령"),
        ("②", "병렬 탐지", "얼굴 + 객체 동시"),
        ("③", "통합 결과", "단일 JSON 병합"),
        ("④", "인물 선택", "블러 제외 설정"),
        ("⑤", "통합 블러", "SAM2 단일 패스"),
        ("⑥", "내보내기", "타임라인 편집"),
    ]

    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="height:96px;">
                    <div style="font-size:11px;color:#808085;margin-bottom:10px;">{num}</div>
                    <div style="font-size:15px;font-weight:700;color:#26262B;margin-bottom:6px;">{title}</div>
                    <div style="font-size:11px;color:#808085;margin-bottom:18px;">{desc}</div>
                    <div style="height:2px;background:#D1D1D6;border-radius:1px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:#737373;font-size:13px;margin-bottom:18px;">주요 처리 기술 — 자연어 입력부터 타임라인 편집까지</div>',
        unsafe_allow_html=True,
    )

    chips = [
        "💬 프롬프트 입력",
        "👤 얼굴 탐지",
        "🔲 객체 탐지",
        "🔀 JSON 통합",
        "🚫 인물 제외",
        "🌀 Gaussian Blur",
        "✂ 타임라인 편집",
    ]
    chip_row(chips)

    st.markdown(
        """
        <div style="
            margin-top:80px;
            background:#E6E6E6;
            padding:28px;
            text-align:center;
            color:#737373;
            font-size:14px;
        ">
            ✓ 서버 업로드 없음 &nbsp;&nbsp; ✓ 로컬 처리 흐름 기반 &nbsp;&nbsp; ✓ 원본 파일 보호
            <br>
            <span style="font-size:12px;color:#B3B3B3;">Gari-Get-That • 맥락 이해 기반 프라이버시 보호 AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def upload_page() -> None:
    header()
    step_badge(1, 6, "영상 업로드 + 자연어 입력", "분석 전 영상과 명령을 함께 입력합니다")
    page_title(
        "영상과 명령을 함께 입력하고 분석을 시작하세요",
        "영상 파일과 자연어 프롬프트를 동시에 입력하면 AI가 병렬 탐지를 바로 시작합니다.",
    )

    left, right = st.columns([1, 1.08], gap="large")

    with left:
        section_label("① 영상 업로드")
        uploaded = st.file_uploader(
            "영상 파일을 드래그하거나 클릭하여 선택",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed",
        )

        if uploaded:
            st.session_state.uploaded_video_name = uploaded.name
            st.success(f"선택된 파일: {uploaded.name}")
        else:
            st.markdown('<div class="small">MP4 · MOV · AVI</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="card">
                <div style="font-size:12px;font-weight:700;margin-bottom:10px;">최근 파일</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div class="small">📄 회의_녹화_2024-03-15.mp4 · 1.2 GB · 45:32</div>
                <div class="small" style="margin-top:8px;">📄 인터뷰_영상.mov · 680 MB · 22:10</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        preview_placeholder("파일을 선택하면 썸네일이 표시됩니다")

    with right:
        section_label("② 자연어 프롬프트 입력")
        prompt = st.text_area(
            "자연어로 무엇을 보호할지 자유롭게 입력하세요",
            value=st.session_state.get("prompt", "집 위치 유추될 만한 건 다 가려줘"),
            height=116,
            label_visibility="collapsed",
        )
        st.session_state.prompt = prompt

        section_label("💡 명령 예시")
        example_cols = st.columns(3)
        examples = [
            "간판만 가려줘",
            "건물명과 택배 정보 블러 처리해줘",
            "사람 제외하고 전부 가려줘",
        ]
        for col, ex in zip(example_cols, examples):
            with col:
                if st.button(ex):
                    st.session_state.prompt = ex
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        video_ready = bool(st.session_state.get("uploaded_video_name"))
        prompt_ready = bool(prompt.strip())

        st.markdown(
            f"""
            <div class="white-card">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">분석 준비 상태</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div style="font-size:12px;color:#808085;">{'●' if video_ready else '○'} 영상 파일 <span style="float:right;">{'완료' if video_ready else '대기 중'}</span></div>
                <div style="font-size:12px;color:#808085;margin-top:8px;">{'●' if prompt_ready else '○'} 자연어 명령 <span style="float:right;">{'완료' if prompt_ready else '대기 중'}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    hint_box("영상 업로드 + 프롬프트 입력 후 분석이 시작됩니다")

    if st.button("분석 시작 →"):
        st.session_state.page = "detect"
        st.rerun()


def detection_progress_page() -> None:
    header()
    step_badge(2, 6, "병렬 탐지 진행", "인물 + 텍스트 + 장소 동시 탐지")
    page_title(
        "원본 영상에서 얼굴과 개인정보 객체를 동시에 탐지합니다",
        "두 탐지 루프가 동일한 원본 영상에 병렬로 실행되어 탐지 시간을 단축합니다.",
    )

    status_running("Face Loop + Object Loop 병렬 실행 중 — 원본 영상 단일 패스")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Face Loop**")
        st.markdown('<div class="small">SCRFD + ByteTrack</div>', unsafe_allow_html=True)
        detection_preview("face")
        st.markdown(
            """
            <div class="white-card" style="margin-top:10px;">
                <div style="font-size:13px;font-weight:500;">👤 탐지된 얼굴</div>
                <div class="small">3명 · face_bbox_list.json</div>
                <div class="small">SCRFD + ByteTrack</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Object Loop**")
        st.markdown('<div class="small">Grounding DINO / OCR / Qwen Reasoning</div>', unsafe_allow_html=True)
        detection_preview("object")
        st.markdown(
            """
            <div class="white-card" style="margin-top:10px;">
                <div style="font-size:13px;font-weight:500;">🔲 탐지된 객체</div>
                <div class="small">3개 · object_bbox_list.json</div>
                <div class="small">Grounding DINO</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="white-card">
            <div style="font-size:14px;font-weight:700;margin-bottom:8px;">🔀 JSON 통합 진행 중</div>
            <div class="page-desc" style="margin-bottom:10px;">face_bbox_list + object_bbox_list → privacy_targets.json 생성 중...</div>
            <div style="height:6px;background:#D1D1D6;border-radius:3px;">
                <div style="width:72%;height:6px;background:#4D80E6;border-radius:3px;"></div>
            </div>
            <div style="text-align:right;font-size:11px;color:#4D80E6;margin-top:4px;">처리 중… 72%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 통합 결과 확인 →", "merged")


def merged_result_page() -> None:
    header()
    step_badge(3, 6, "통합 탐지 결과", "보호 대상 목록 확인")
    page_title(
        "Face/Object 탐지 결과",
        "얼굴 탐지와 객체 탐지 결과를 단일 JSON으로 병합합니다.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        json_panel("privacy_targets.json", DUMMY_TARGETS)
        st.markdown('<div class="small" style="margin-top:10px;">총 5개 항목 통합 완료</div>', unsafe_allow_html=True)

    with right:
        target_item("얼굴 A", "인물", "blue")
        target_item("얼굴 B", "인물", "blue")
        target_item("📝 간판", "텍스트/간판", "red")
        target_item("건물명", "텍스트/간판", "red")
        target_item("택배 정보", "텍스트/간판", "red")

        st.markdown(
            """
            <div class="card" style="margin-top:10px;">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">탐지 결과 요약</div>
                <div style="font-size:13px;color:#26262B;">👤 인물 2개 | 📝 텍스트/간판 2개 | 🔲 기타 객체 1개</div>
                <div class="small" style="margin-top:8px;">다음 단계: 블러 제외 인물을 선택한 후 SAM2로 단일 패스 마스크를 생성합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    next_button("다음: 인물 선택 →", "select")


def person_select_page() -> None:
    header()
    step_badge(4, 6, "인물 선택", "블러 제외 설정")
    page_title(
        "중요 인물은 선명하게, 나머지는 자동 블러",
        "선택한 인물은 블러 처리에서 제외됩니다. 나머지 인물과 탐지된 개인정보 객체는 모두 자동 처리됩니다.",
    )

    st.markdown(
        """
        <div class="hint-box">
            <span style="color:#2E994D;font-weight:500;">📌 선택 인물 = 블러 제외 (선명 유지)</span>
            <span style="color:#D1D1D6;margin:0 24px;">|</span>
            <span style="color:#B34040;font-weight:500;">나머지 인물 + 탐지 객체 = 자동 블러 처리</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    people = [
        ("인물 A", "12분 34초", True),
        ("인물 B", "7분 22초", True),
        ("인물 C", "4분 10초", False),
    ]

    selected = []
    for col, (name, duration, default) in zip(cols, people):
        with col:
            is_selected = person_card(name, duration, selected=default, key=f"select_{name}")
            if is_selected:
                selected.append(name)

    st.session_state.selected_people = selected

    st.markdown("<br>", unsafe_allow_html=True)

    selected_text = ", ".join(selected) if selected else "없음"
    blur_people = [name for name, _, _ in people if name not in selected]
    blur_text = ", ".join(blur_people) if blur_people else "없음"

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:12px;font-weight:700;margin-bottom:8px;">선택 요약</div>
            <div style="font-size:13px;color:#2E994D;">✓ {selected_text} → 블러 제외 (선명 유지)</div>
            <div style="font-size:13px;color:#B34040;margin-top:8px;">✕ {blur_text} + 간판·건물명·택배 정보 → 자동 블러 처리</div>
            <div class="small" style="margin-top:8px;">privacy_targets.json → SAM2에 단일 패스로 전달</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 자동 블러 처리 →", "blur")


def blur_result_page() -> None:
    header()
    step_badge(5, 6, "자동 블러 처리", "SAM2 + Gaussian Blur 단일 패스")
    page_title(
        "선택 인물 제외 — 모든 대상에 통합 블러 자동 적용",
        "SAM2가 통합 마스크를 단일 패스로 생성하고, Gaussian Blur를 한 번에 적용합니다.",
    )

    left, right = st.columns([2.2, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div style="
                background:#F5F5F7;
                border:1.5px solid #80CC8C;
                border-radius:12px;
                height:316px;
                position:relative;
                padding:16px;
            ">
                <div style="font-size:12px;font-weight:500;color:#2E994D;">처리 후 (블러 적용)</div>
                <div style="position:absolute;left:34px;top:52px;width:100px;height:136px;background:#D6E6FF;border-radius:6px;"></div>
                <div style="position:absolute;left:34px;top:34px;font-size:12px;color:#4D80E6;">인물 A ✓</div>
                <div style="position:absolute;left:184px;top:44px;width:96px;height:130px;background:#D6E6FF;border-radius:6px;"></div>
                <div style="position:absolute;left:184px;top:28px;font-size:12px;color:#4D80E6;">인물 B ✓</div>
                <div style="position:absolute;left:322px;top:58px;width:94px;height:128px;background:#94949E;border-radius:6px;"></div>
                <div style="position:absolute;left:340px;top:112px;font-size:13px;font-weight:700;color:#E0E0E6;">BLUR</div>
                <div style="position:absolute;left:322px;top:40px;font-size:12px;color:#B34040;">인물 C</div>
                <div style="position:absolute;left:24px;top:214px;width:118px;height:54px;background:#94949E;border-radius:4px;"></div>
                <div style="position:absolute;left:166px;top:222px;width:128px;height:48px;background:#94949E;border-radius:4px;"></div>
                <div style="position:absolute;left:306px;top:204px;width:140px;height:52px;background:#94949E;border-radius:4px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="card">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">처리 상태</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div class="small">👤 얼굴 BBox <b style="float:right;color:#26262B;">2개</b></div>
                <div class="small" style="margin-top:14px;">🔲 객체 BBox <b style="float:right;color:#26262B;">3개</b></div>
                <div class="small" style="margin-top:14px;">🚫 제외 인물 <b style="float:right;color:#26262B;">2명</b></div>
                <div style="height:1px;background:#D1D1D6;margin:20px 0 10px;"></div>
                <div class="small" style="color:#2E994D;font-weight:500;">✓ SAM2 마스크 생성 완료</div>
                <div class="small" style="color:#2E994D;font-weight:500;margin-top:12px;">✓ Gaussian Blur 단일 패스 완료</div>
                <div class="small" style="color:#2E994D;font-weight:500;margin-top:12px;">처리 시간: 6.2초</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
            background:#EBF2FF;
            border:1px solid #B3D1FF;
            border-radius:8px;
            padding:14px 20px;
            color:#4D80E6;
            font-size:12px;
            font-weight:500;
        ">
            🧠 SAM2 통합 마스크 생성 → Gaussian Blur 단일 패스 | 선택 인물은 마스크에서 제외됩니다
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 타임라인 편집 및 내보내기 →", "export")


def export_page() -> None:
    header()
    step_badge(6, 6, "타임라인 편집 & 결과 내보내기", "블러 구간 조정 후 최종 저장")
    page_title(
        "블러 처리 결과를 검토하고 영상을 내보냅니다",
        "타임라인에서 블러 구간을 확인·조정하고 원하는 포맷으로 내보냅니다.",
    )

    left, right = st.columns([3.2, 0.9], gap="large")

    with left:
        preview_placeholder("📹 최종 블러 영상 미리보기")

    with right:
        st.markdown(
            """
            <div class="card">
                <div style="font-size:13px;font-weight:700;margin-bottom:10px;">블러 적용 항목</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
                <div class="white-card" style="margin-bottom:8px;">👤 인물 C</div>
                <div class="white-card" style="margin-bottom:8px;">📝 간판</div>
                <div class="white-card" style="margin-bottom:8px;">건물명</div>
                <div class="white-card" style="margin-bottom:8px;">택배 정보</div>
                <div class="small">총 4개 블러 항목</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="timeline-panel">', unsafe_allow_html=True)
    st.markdown("**⏱ 타임라인 편집기**")
    st.markdown("<hr>", unsafe_allow_html=True)
    timeline_row("인물 C", [(10, 25), (55, 30)], object_type=False)
    timeline_row("📝 간판", [(20, 15), (65, 20)], object_type=True)
    timeline_row("건물명", [(15, 20)], object_type=True)
    timeline_row("택배 정보", [(50, 25)], object_type=True)
    st.markdown('<div class="small">구간을 드래그하여 블러 범위를 조정할 수 있습니다.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1.4])
    with col1:
        st.button("수정 적용")
    with col2:
        st.button("임시저장")
    with col3:
        if st.button("영상 내보내기 →"):
            st.session_state.page = "final"
            st.rerun()


def final_page() -> None:
    header()
    step_badge(6, 6, "최종 결과", "블러 처리 완료 및 다운로드")
    status_success("처리 완료 — 영상 다운로드 준비가 됐습니다")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([2.2, 1], gap="large")

    with left:
        preview_placeholder("최종 블러 처리 결과 영상")

    with right:
        st.markdown(
            """
            <div class="white-card">
                <div style="font-size:14px;font-weight:700;margin-bottom:10px;">처리 결과</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
                <div class="small">블러 적용 <b style="float:right;color:#26262B;">얼굴 1명 + 객체 3개</b></div>
                <div class="small" style="margin-top:14px;">제외 인물 <b style="float:right;color:#26262B;">2명</b></div>
                <div style="height:1px;background:#D1D1D6;margin:18px 0 12px;"></div>
                <div style="background:#EBF2FF;border:1px solid #B3D1FF;border-radius:6px;padding:9px 12px;color:#4D80E6;font-size:12px;font-weight:500;">
                    ✓ 인물 A, 인물 B — 선명 유지
                </div>
                <div style="height:1px;background:#D1D1D6;margin:18px 0 8px;"></div>
                <div class="small">처리 시간 <b style="float:right;color:#26262B;">14.6초</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("⬇ 영상 다운로드")
    with col2:
        if st.button("← 다시 편집하기"):
            st.session_state.page = "export"
            st.rerun()
    with col3:
        if st.button("새 영상 업로드"):
            st.session_state.page = "upload"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div style="display:flex;gap:120px;">
                <div>
                    <div class="small">처리 시간</div>
                    <div style="font-size:14px;font-weight:700;">14.6초</div>
                </div>
                <div>
                    <div class="small">적용 대상</div>
                    <div style="font-size:14px;font-weight:700;">얼굴 1명 + 객체 3개</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_page() -> None:
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    page = st.session_state.page

    if page == "landing":
        landing_page()
    elif page == "upload":
        upload_page()
    elif page == "detect":
        detection_progress_page()
    elif page == "merged":
        merged_result_page()
    elif page == "select":
        person_select_page()
    elif page == "blur":
        blur_result_page()
    elif page == "export":
        export_page()
    elif page == "final":
        final_page()
    else:
        st.session_state.page = "landing"
        landing_page()