# ui/pages.py

from __future__ import annotations

from pathlib import Path

import config
import streamlit as st

from ui.pipeline_adapter import (
    get_file_hash,
    save_uploaded_video,
    run_detection_stage,
    load_person_db,
    load_object_db,
    run_face_export_and_merge,
    load_sam2_targets,
    run_blur_stage,
)
from ui.components import (
    header,
    step_badge,
    page_title,
    section_label,
    hint_box,
    status_success,
    status_running,
    chip_row,
    json_panel,
    target_item,
    person_card,
    preview_placeholder,
    detection_preview,
    timeline_row,
    next_button,
)


# ---------------------------------------------------------------------------
# 상태 관리
# ---------------------------------------------------------------------------

def reset_pipeline_state() -> None:
    """파이프라인 처리 상태를 초기화합니다.
    uploaded_file_hash / video_path / uploaded_video_name은
    upload_page에서 명시적으로 덮어쓰므로 여기서는 건드리지 않습니다.
    """
    keys = [
        "detection_done",
        "merge_done",
        "blur_done",
        "merged_targets",
        "output_video_path",
        "exclude_person_ids",
        "selected_people",
    ]
    for key in keys:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# 페이지: 랜딩
# ---------------------------------------------------------------------------

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
        ("③", "인물 선택", "블러 제외 설정"),
        ("④", "통합 결과", "단일 JSON 병합"),
        ("⑤", "통합 블러", "SAM2 단일 패스"),
        ("⑥", "내보내기", "최종 결과 저장"),
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




# ---------------------------------------------------------------------------
# 페이지: 업로드
# ---------------------------------------------------------------------------

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
            type=["mp4"],
            label_visibility="collapsed",
        )

        if uploaded:
            file_hash = get_file_hash(uploaded)

            # 파일이 바뀌었으면 파이프라인 상태 초기화
            if st.session_state.get("uploaded_file_hash") != file_hash:
                reset_pipeline_state()
                st.session_state.uploaded_file_hash = file_hash

                video_path = save_uploaded_video(uploaded)
                st.session_state.video_path = str(video_path)
                st.session_state.uploaded_video_name = uploaded.name

            st.success(f"선택된 파일: {uploaded.name}")
        else:
            st.markdown('<div class="small">MP4</div>', unsafe_allow_html=True)

    with right:
        section_label("② 자연어 프롬프트 입력")
        prompt = st.text_area(
            "자연어로 무엇을 보호할지 자유롭게 입력하세요",
            value=st.session_state.get("prompt", ""),
            height=116,
            label_visibility="collapsed",
            placeholder="예: 집 위치 유추될 만한 건 다 가려줘",
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
        if not st.session_state.get("video_path"):
            st.warning("먼저 영상 파일을 선택해주세요.")
            return
        if not st.session_state.get("prompt", "").strip():
            st.warning("자연어 명령을 입력해주세요.")
            return
        st.session_state.page = "detect"
        st.rerun()


# ---------------------------------------------------------------------------
# 페이지: 탐지 진행
# ---------------------------------------------------------------------------

def detection_progress_page() -> None:
    # Guard
    if not st.session_state.get("video_path"):
        st.warning("먼저 영상을 업로드해주세요.")
        if st.button("업로드 화면으로 이동"):
            st.session_state.page = "upload"
            st.rerun()
        return

    header()
    step_badge(2, 6, "병렬 탐지 진행", "얼굴 + 객체 동시 탐지")
    page_title(
        "원본 영상에서 얼굴과 개인정보 객체를 동시에 탐지합니다",
        "얼굴 탐지와 개인정보 객체 탐지를 실행하고, 결과를 다음 단계에서 확인합니다.",
    )

    if not st.session_state.get("detection_done", False):
        status_running("Face Loop + Object Loop 실행 중 — 얼굴 및 개인정보 객체 탐지")

        progress_bar = st.progress(0, text="탐지를 시작합니다...")

        def update_progress(pct: int, msg: str) -> None:
            progress_bar.progress(pct, text=msg)

        try:
            run_detection_stage(
                video_path=st.session_state.video_path,
                prompt=st.session_state.prompt,
                progress_callback=update_progress,
            )
        except FileNotFoundError as e:
            st.error(f"파일 오류: {e}")
            return
        except Exception as e:
            st.error(f"탐지 중 오류가 발생했습니다: {e}")
            return

        st.session_state.detection_done = True
        st.rerun()

    status_success("탐지 완료 — 인물 선택 단계로 이동할 수 있습니다")

    # 실제 탐지 결과 요약 표시
    person_db = load_person_db()
    face_count = len(person_db)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Face Loop**")
        st.markdown('<div class="small">SCRFD + ByteTrack</div>', unsafe_allow_html=True)
        if config.FACE_PREVIEW_PATH.exists():
            st.image(str(config.FACE_PREVIEW_PATH), caption="Face detection preview")
        else:
            detection_preview("face")
        st.markdown(
            f"""
            <div class="white-card" style="margin-top:10px;">
                <div style="font-size:13px;font-weight:500;">👤 탐지된 얼굴</div>
                <div class="small">{face_count}명 · face_bbox_list.json</div>
                <div class="small">SCRFD + ByteTrack</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        object_db = load_object_db()
        object_count = len(object_db) if isinstance(object_db, list) else 0

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Object Loop**")
        st.markdown(
            '<div class="small">PaddleOCR + Qwen2-VL Reasoning</div>',
            unsafe_allow_html=True,
        )

        if config.OBJECT_PREVIEW_PATH.exists():
            st.image(
                str(config.OBJECT_PREVIEW_PATH),
                caption="Object/Text detection preview",
            )
        else:
            detection_preview("object")

        st.markdown(
            f"""
            <div class="white-card" style="margin-top:10px;">
                <div style="font-size:13px;font-weight:500;">🔲 탐지된 객체/텍스트</div>
                <div class="small">{object_count}개 · object_db.json</div>
                <div class="small">PaddleOCR + Qwen2-VL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="white-card">
            <div style="font-size:14px;font-weight:700;margin-bottom:8px;">탐지 결과 준비 완료</div>
            <div class="page-desc" style="margin-bottom:10px;">
                얼굴 후보와 개인정보 객체 후보가 생성되었습니다. 다음 단계에서 블러 제외 인물을 선택합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 인물 선택 →", "select")


# ---------------------------------------------------------------------------
# 페이지: 인물 선택
# ---------------------------------------------------------------------------

def person_select_page() -> None:
    # Guard
    if not st.session_state.get("detection_done"):
        st.warning("먼저 탐지 단계를 완료해주세요.")
        if st.button("탐지 화면으로 이동"):
            st.session_state.page = "detect"
            st.rerun()
        return

    header()
    step_badge(3, 6, "인물 선택", "블러 제외 설정")
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

    person_db = load_person_db()

    if not person_db:
        st.warning("탐지된 인물 정보가 없습니다. 탐지 단계를 다시 실행해주세요.")
        if st.button("탐지 화면으로 이동"):
            st.session_state.page = "detect"
            st.rerun()
        return

    people = []
    for person_id, person in person_db.items():
        total_frames = person.get("total_frames", 0)
        people.append(
            {
                "person_id": person_id,
                "name": person_id,
                "duration": f"{total_frames} frames",
                "repr_image": person.get("repr_image"),
                "default_selected": person.get("is_main", False),
            }
        )

    exclude_person_ids = []
    cols = st.columns(min(len(people), 3))

    for idx, person in enumerate(people):
        with cols[idx % len(cols)]:
            is_selected = person_card(
                name=person["name"],
                duration=person["duration"],
                selected=person["default_selected"],
                key=f"select_{person['person_id']}",
            )
            if is_selected:
                exclude_person_ids.append(person["person_id"])

    st.session_state.exclude_person_ids = exclude_person_ids

    st.markdown("<br>", unsafe_allow_html=True)

    selected_text = ", ".join(exclude_person_ids) if exclude_person_ids else "없음"
    blur_people = [p["person_id"] for p in people if p["person_id"] not in exclude_person_ids]
    blur_text = ", ".join(blur_people) if blur_people else "없음"

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:12px;font-weight:700;margin-bottom:8px;">선택 요약</div>
            <div style="font-size:13px;color:#2E994D;">✓ {selected_text} → 블러 제외 (선명 유지)</div>
            <div style="font-size:13px;color:#B34040;margin-top:8px;">✕ {blur_text} + 탐지 객체 → 자동 블러 처리</div>
            <div class="small" style="margin-top:8px;">선택 결과는 다음 단계에서 SAM2 입력 JSON에 반영됩니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 통합 결과 확인 →", "merged")


# ---------------------------------------------------------------------------
# 페이지: 통합 결과
# ---------------------------------------------------------------------------

def merged_result_page() -> None:
    # Guard
    if not st.session_state.get("detection_done"):
        st.warning("먼저 탐지 단계를 완료해주세요.")
        if st.button("탐지 화면으로 이동"):
            st.session_state.page = "detect"
            st.rerun()
        return

    if "exclude_person_ids" not in st.session_state:
        st.warning("먼저 블러 제외 인물을 선택해주세요.")
        if st.button("인물 선택 화면으로 이동"):
            st.session_state.page = "select"
            st.rerun()
        return

    header()
    step_badge(4, 6, "통합 결과", "선택 인물 제외 기준으로 SAM2 입력 JSON 생성")
    page_title(
        "Face/Object 탐지 결과를 SAM2 입력 JSON으로 통합합니다",
        "사용자가 선택한 블러 제외 인물은 제외하고, 나머지 얼굴과 개인정보 객체를 하나의 target 목록으로 병합합니다.",
    )

    if not st.session_state.get("merge_done", False):
        exclude_ids = set(st.session_state.get("exclude_person_ids", []))

        with st.spinner("선택 인물 제외 기준으로 SAM2 입력 JSON을 생성하는 중입니다..."):
            try:
                targets = run_face_export_and_merge(exclude_ids)
            except FileNotFoundError as e:
                st.error(f"파일 오류: {e}")
                return
            except Exception as e:
                st.error(f"통합 결과 생성 중 오류가 발생했습니다: {e}")
                return

        st.session_state.merge_done = True
        st.session_state.merged_targets = targets
        st.rerun()
    else:
        targets = st.session_state.get("merged_targets") or load_sam2_targets()

    left, right = st.columns([1, 1], gap="large")

    with left:
        json_panel("sam2_targets.json", targets)
        st.markdown(
            f'<div class="small" style="margin-top:10px;">총 {len(targets)}개 항목 통합 완료</div>',
            unsafe_allow_html=True,
        )

    with right:
        face_count = sum(1 for t in targets if t.get("type") == "face")
        object_count = len(targets) - face_count

        for target in targets:
            target_type = target.get("type", "object")
            target_id = target.get("id", "unknown")

            if target_type == "face":
                target_item(target_id, "인물", "blue", meta="box 포함")
            else:
                label = target.get("label", target_type)
                target_item(label, "객체/텍스트", "red", meta="box 포함")

        st.markdown(
            f"""
            <div class="card" style="margin-top:10px;">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">통합 결과 요약</div>
                <div style="font-size:13px;color:#26262B;">👤 얼굴 {face_count}개 | 🔲 객체/텍스트 {object_count}개</div>
                <div class="small" style="margin-top:8px;">다음 단계에서 SAM2 마스크 생성 및 Gaussian Blur가 적용됩니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    next_button("다음: 자동 블러 처리 →", "blur")


# ---------------------------------------------------------------------------
# 페이지: 블러 처리
# ---------------------------------------------------------------------------

def blur_result_page() -> None:
    # Guard
    if not st.session_state.get("merge_done"):
        st.warning("먼저 통합 결과를 생성해주세요.")
        if st.button("통합 결과 화면으로 이동"):
            st.session_state.page = "merged"
            st.rerun()
        return

    header()
    step_badge(5, 6, "자동 블러 처리", "SAM2 + Gaussian Blur 단일 패스")
    page_title(
        "선택 인물 제외 — 모든 대상에 통합 블러 자동 적용",
        "SAM2가 통합 마스크를 생성하고, Gaussian Blur를 적용합니다.",
    )

    if not st.session_state.get("blur_done", False):
        with st.spinner("SAM2 마스크 생성 및 Gaussian Blur를 적용하는 중입니다..."):
            try:
                output_video_path = run_blur_stage(st.session_state.video_path)
            except (FileNotFoundError, ValueError) as e:
                st.error(f"설정 오류: {e}")
                st.info("이전 단계부터 다시 실행해주세요.")
                return
            except Exception as e:
                st.error(f"블러 처리 중 예상치 못한 오류가 발생했습니다: {e}")
                return

        st.session_state.output_video_path = str(output_video_path)
        st.session_state.blur_done = True
        st.rerun()

    status_success("블러 처리 완료 — 결과를 확인할 수 있습니다")

    # 실제 처리 결과 요약
    targets = st.session_state.get("merged_targets") or load_sam2_targets()
    face_count = sum(1 for t in targets if t.get("type") == "face")
    object_count = len(targets) - face_count
    exclude_count = len(st.session_state.get("exclude_person_ids", []))

    left, right = st.columns([2.2, 0.85], gap="large")

    with left:
        output_path = st.session_state.get("output_video_path")
        if output_path and Path(output_path).exists():
            st.video(output_path)
        else:
            preview_placeholder("블러 처리된 영상 미리보기")

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">처리 상태</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div class="small">👤 얼굴 BBox <b style="float:right;color:#26262B;">{face_count}개</b></div>
                <div class="small" style="margin-top:14px;">🔲 객체 BBox <b style="float:right;color:#26262B;">{object_count}개</b></div>
                <div class="small" style="margin-top:14px;">🚫 제외 인물 <b style="float:right;color:#26262B;">{exclude_count}명</b></div>
                <div style="height:1px;background:#D1D1D6;margin:20px 0 10px;"></div>
                <div class="small" style="color:#2E994D;font-weight:500;">✓ SAM2 마스크 생성 완료</div>
                <div class="small" style="color:#2E994D;font-weight:500;margin-top:12px;">✓ Gaussian Blur 단일 패스 완료</div>
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


# ---------------------------------------------------------------------------
# 페이지: 내보내기
# ---------------------------------------------------------------------------

def export_page() -> None:
    # Guard
    if not st.session_state.get("blur_done"):
        st.warning("먼저 블러 처리를 완료해주세요.")
        if st.button("블러 처리 화면으로 이동"):
            st.session_state.page = "blur"
            st.rerun()
        return

    header()
    step_badge(6, 6, "타임라인 편집 & 결과 내보내기", "블러 구간 조정 후 최종 저장")
    page_title(
        "블러 처리 결과를 검토하고 영상을 내보냅니다",
        "타임라인에서 블러 구간을 확인·조정하고 원하는 포맷으로 내보냅니다.",
    )

    targets = st.session_state.get("merged_targets") or load_sam2_targets()

    blur_targets = [t for t in targets]  # 이미 exclude 반영된 목록

    left, right = st.columns([3.2, 0.9], gap="large")

    with left:
        output_path = st.session_state.get("output_video_path")
        if output_path and Path(output_path).exists():
            st.video(output_path)
        else:
            preview_placeholder("📹 최종 블러 영상 미리보기")

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:13px;font-weight:700;margin-bottom:10px;">블러 적용 항목</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>', unsafe_allow_html=True)

        for target in blur_targets:
            label = target.get("label") or target.get("id", "unknown")
            t_type = "👤" if target.get("type") == "face" else "🔲"
            st.markdown(
                f'<div class="white-card" style="margin-bottom:8px;">{t_type} {label}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="small">총 {len(blur_targets)}개 블러 항목</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="timeline-panel">', unsafe_allow_html=True)
    st.markdown("**⏱ 타임라인**")
    st.markdown("<hr>", unsafe_allow_html=True)

    for target in blur_targets:
        label = target.get("label") or target.get("id", "unknown")
        is_object = target.get("type") != "face"
        # start_frame / end_frame 기반으로 타임라인 구간 계산
        # 전체 프레임 수는 아직 없으므로 상대 비율로만 표시, 추후 video_meta.total_frames 연동 예정
        start = int(target.get("start_frame", 0))
        end = int(target.get("end_frame", start + 1))
        if end > start:
            # 전체 길이를 end 기준으로 정규화 (단일 target 기준 임시 처리)
            total = end
            seg_start = int(start / total * 100)
            seg_width = int((end - start) / total * 100)
            segments = [(seg_start, max(seg_width, 5))]
        else:
            segments = [(0, 100)]
        timeline_row(label, segments, object_type=is_object)

    st.markdown('<div class="small">현재 버전에서는 블러 적용 구간을 확인할 수 있습니다.</div>', unsafe_allow_html=True)
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


# ---------------------------------------------------------------------------
# 페이지: 최종 결과
# ---------------------------------------------------------------------------

def final_page() -> None:
    # Guard
    if not st.session_state.get("blur_done"):
        st.warning("먼저 블러 처리를 완료해주세요.")
        if st.button("블러 처리 화면으로 이동"):
            st.session_state.page = "blur"
            st.rerun()
        return

    header()
    step_badge(6, 6, "최종 결과", "블러 처리 완료 및 다운로드")
    status_success("처리 완료 — 영상 다운로드 준비가 됐습니다")

    st.markdown("<br>", unsafe_allow_html=True)

    targets = st.session_state.get("merged_targets") or load_sam2_targets()
    face_count = sum(1 for t in targets if t.get("type") == "face")
    object_count = len(targets) - face_count
    exclude_ids = st.session_state.get("exclude_person_ids", [])
    exclude_count = len(exclude_ids)
    exclude_text = ", ".join(exclude_ids) if exclude_ids else "없음"

    left, right = st.columns([2.2, 1], gap="large")

    with left:
        output_path = st.session_state.get("output_video_path")

        if output_path and Path(output_path).exists():
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇ 영상 다운로드",
                    data=f,
                    file_name="privacy_guard_output.mp4",
                    mime="video/mp4",
                )
        else:
            preview_placeholder("최종 블러 처리 결과 영상")

    with right:
        st.markdown(
            f"""
            <div class="white-card">
                <div style="font-size:14px;font-weight:700;margin-bottom:10px;">처리 결과</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
                <div class="small">블러 적용 <b style="float:right;color:#26262B;">얼굴 {face_count}명 + 객체 {object_count}개</b></div>
                <div class="small" style="margin-top:14px;">제외 인물 <b style="float:right;color:#26262B;">{exclude_count}명</b></div>
                <div style="height:1px;background:#D1D1D6;margin:18px 0 12px;"></div>
                <div style="background:#EBF2FF;border:1px solid #B3D1FF;border-radius:6px;padding:9px 12px;color:#4D80E6;font-size:12px;font-weight:500;">
                    ✓ 선명 유지: {exclude_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("← 다시 편집하기"):
            st.session_state.page = "export"
            st.rerun()

    with col3:
        if st.button("새 영상 업로드"):
            reset_pipeline_state()
            st.session_state.page = "upload"
            st.rerun()


# ---------------------------------------------------------------------------
# 라우터
# ---------------------------------------------------------------------------

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
    elif page == "select":
        person_select_page()
    elif page == "merged":
        merged_result_page()
    elif page == "blur":
        blur_result_page()
    elif page == "export":
        export_page()
    elif page == "final":
        final_page()
    else:
        st.session_state.page = "landing"
        landing_page()