# ui/pages.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

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
    target_item,
    preview_placeholder,
    detection_preview,
    next_button,
)


# ---------------------------------------------------------------------------
# 상태 관리
# ---------------------------------------------------------------------------

def reset_pipeline_state() -> None:
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


def reset_all_state() -> None:
    reset_pipeline_state()

    keys = [
        "uploaded_file_hash",
        "video_path",
        "uploaded_video_name",
        "prompt",
    ]

    for key in keys:
        st.session_state.pop(key, None)


def invalidate_after_person_selection_change() -> None:
    st.session_state.merge_done = False
    st.session_state.blur_done = False
    st.session_state.pop("merged_targets", None)
    st.session_state.pop("output_video_path", None)


def resolve_person_image_path(person: dict) -> Path | None:
    image_value = (
        person.get("repr_image")
        or person.get("repr_crop_path")
        or person.get("crop_path")
        or person.get("image_path")
        or person.get("repr_crop")
    )

    if not image_value:
        return None

    path = Path(image_value)

    if path.exists():
        return path

    alt_path = config.BASE_DIR / path if hasattr(config, "BASE_DIR") else path
    if alt_path.exists():
        return alt_path

    return None


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
                얼굴 및 개인정보 자동 탐지 · 원하는 인물 유지 · 영상 프라이버시 보호
            </div>
            <div style="
                font-size:40px;
                font-weight:700;
                line-height:1.25;
                color:#212121;
                margin-bottom:18px;
            ">
                무엇을 가릴지 말해주면<br>
                AI가 알아서 찾아 처리합니다
            </div>
            <div style="
                font-size:15px;
                color:#737373;
                margin-bottom:28px;
            ">
                얼굴, 차량번호, 간판, 건물명 등<br>
                개인정보가 포함된 영역을 자동으로 찾아 안전하게 가려줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin-top:4px; margin-bottom:36px;">
        """,
        unsafe_allow_html=True,
    )

    button_left, button_center, button_right = st.columns([2.2, 1, 2.2])
    with button_center:
        next_button(
            "지금 시작하기",
            "upload",
            button_type="primary",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;color:#737373;font-size:12px;margin-bottom:18px;">AI 처리 5단계 흐름</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("①", "영상 업로드", "영상과 요청 입력"),
        ("②", "AI 분석", "얼굴·개인정보 탐지"),
        ("③", "인물 선택", "선명하게 유지할 인물 선택"),
        ("④", "적용 항목 확인", "가려질 대상 확인"),
        ("⑤", "결과 저장", "최종 영상 다운로드"),
    ]

    cols = st.columns(5)

    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="height:96px;text-align:left;">
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
        '<div style="text-align:center;color:#737373;font-size:13px;margin-bottom:18px;">주요 처리 기술</div>',
        unsafe_allow_html=True,
    )

    chips = [
        "💬 요청 입력",
        "👤 얼굴 찾기",
        "🔎 개인정보 영역 찾기",
        "🚫 인물 제외",
        "🌀 자연스러운 블러",
    ]

    chips_html = "".join(
        f"""
        <span style="
            display:inline-flex;
            align-items:center;
            justify-content:center;
            border:1px solid #D1D1D6;
            border-radius:18px;
            padding:8px 14px;
            margin:4px 5px;
            font-size:13px;
            color:#26262B;
            background:white;
            white-space:nowrap;
        ">
            {chip}
        </span>
        """
        for chip in chips
    )

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:center;
            align-items:center;
            flex-wrap:wrap;
            gap:4px;
            width:100%;
            margin-top:4px;
        ">
            {chips_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 페이지: 업로드
# ---------------------------------------------------------------------------

def upload_page() -> None:
    header()
    step_badge(1, 5, "영상 업로드", "영상과 가리고 싶은 내용을 입력합니다")
    page_title(
        "영상을 업로드하고, 가리고 싶은 내용을 입력하세요",
        "AI가 영상을 분석해 얼굴과 개인정보가 포함된 영역을 찾아줍니다.",
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

            if st.session_state.get("uploaded_file_hash") != file_hash:
                reset_pipeline_state()
                st.session_state.uploaded_file_hash = file_hash

                video_path = save_uploaded_video(uploaded)
                st.session_state.video_path = str(video_path)
                st.session_state.uploaded_video_name = uploaded.name

            st.success(f"선택된 파일: {uploaded.name}")
        else:
            st.markdown('<div class="small">지원 형식: MP4</div>', unsafe_allow_html=True)

    with right:
        section_label("② 가릴 내용 입력")
        prompt = st.text_area(
            "영상에서 어떤 부분을 가릴지 입력하세요",
            value=st.session_state.get("prompt", ""),
            height=116,
            label_visibility="collapsed",
            placeholder="예: 집 위치를 알 수 있는 정보는 모두 가려줘",
        )
        st.session_state.prompt = prompt

        st.markdown("""
            <style>
            div.stButton > button {
                font-size: 13px !important;
                padding: 8px 12px !important;
                min-height: 60px !important;
                white-space: normal !important;
                line-height: 1.2 !important;
            }
            </style>
            """, unsafe_allow_html=True)
        section_label("💡 요청 예시")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "간판만 가려줘",
                use_container_width=True,
                key="ex_sign"
            ):
                st.session_state.prompt = "간판만 가려줘"
                st.rerun()

        with col2:
            if st.button(
                "건물명·택배정보\n가려줘",
                use_container_width=True,
                key="ex_delivery"
            ):
                st.session_state.prompt = "건물명·택배정보 가려줘"
                st.rerun()

        with col3:
            if st.button(
                "사람 빼고 전부\n가려줘",
                use_container_width=True,
                key="ex_all"
            ):
                st.session_state.prompt = "사람 빼고 전부 가려줘"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        video_ready = bool(st.session_state.get("uploaded_video_name"))
        prompt_ready = bool(prompt.strip())

        st.markdown(
            f"""
            <div class="white-card">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">준비 상태</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div style="font-size:12px;color:#808085;">{'●' if video_ready else '○'} 영상 파일 <span style="float:right;">{'선택 완료' if video_ready else '대기 중'}</span></div>
                <div style="font-size:12px;color:#808085;margin-top:8px;">{'●' if prompt_ready else '○'} 가릴 내용 <span style="float:right;">{'입력 완료' if prompt_ready else '대기 중'}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    hint_box("영상과 요청 내용을 입력하면 AI 분석을 시작할 수 있습니다.")

    if st.button("AI 분석 시작 →", type="primary"):
        if not st.session_state.get("video_path"):
            st.warning("먼저 영상 파일을 선택해주세요.")
            return

        if not st.session_state.get("prompt", "").strip():
            st.warning("가리고 싶은 내용을 입력해주세요.")
            return

        st.session_state.page = "detect"
        st.rerun()


# ---------------------------------------------------------------------------
# 페이지: 탐지 진행
# ---------------------------------------------------------------------------

def detection_progress_page() -> None:
    if not st.session_state.get("video_path"):
        st.warning("먼저 영상을 업로드해주세요.")
        if st.button("업로드 화면으로 이동"):
            st.session_state.page = "upload"
            st.rerun()
        return

    header()
    step_badge(2, 5, "AI 분석", "영상 속 얼굴과 개인정보 영역을 찾습니다")
    page_title(
        "영상 속 얼굴과 개인정보를 찾고 있습니다",
        "얼굴, 간판, 건물명, 차량번호 등 가려야 할 수 있는 영역을 분석합니다.",
    )

    if not st.session_state.get("detection_done", False):
        status_running("AI가 영상 속 얼굴과 개인정보 영역을 분석하고 있습니다")

        progress_bar = st.progress(0, text="분석을 시작합니다...")

        def update_progress(pct: int, msg: str) -> None:
            user_msg = msg

            if "얼굴" in msg:
                user_msg = "얼굴이 등장하는 구간을 찾고 있습니다..."
            elif "객체" in msg or "텍스트" in msg:
                user_msg = "간판, 건물명, 차량번호 등 개인정보 영역을 찾고 있습니다..."
            elif "완료" in msg:
                user_msg = "분석이 완료되었습니다."

            progress_bar.progress(pct, text=user_msg)

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
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            return

        st.session_state.detection_done = True
        st.rerun()

    status_success("분석 완료 — 이제 선명하게 유지할 인물을 선택할 수 있습니다")

    person_db = load_person_db()
    face_count = len(person_db)

    object_db = load_object_db()
    object_count = len(object_db) if isinstance(object_db, list) else 0

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size:14px;font-weight:700;margin-bottom:6px;">
                분석 결과가 준비되었습니다
            </div>
            <div class="page-desc" style="margin-bottom:18px;">
                영상에서 얼굴과 개인정보가 포함될 수 있는 영역을 찾았습니다.
                다음 단계에서 선명하게 유지할 인물을 선택할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**얼굴 분석 결과**")
            st.markdown(
                '<div class="small">영상에 등장한 인물 후보를 찾았습니다</div>',
                unsafe_allow_html=True,
            )

            if config.FACE_PREVIEW_PATH.exists():
                st.image(
                    str(config.FACE_PREVIEW_PATH),
                    caption="영상에 등장한 인물 후보를 찾았습니다",
                    use_container_width=True,
                )
            else:
                detection_preview("face")

            st.markdown(
                f"""
                <div class="white-card" style="margin-top:10px;">
                    <div style="font-size:13px;font-weight:500;">👤 찾은 인물</div>
                    <div class="small">{face_count}명 · person_db.json</div>
                    <div class="small">다음 단계에서 블러 제외 대상을 선택할 수 있습니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("**Object Loop**")
            st.markdown(
                '<div class="small">PaddleOCR + Qwen2-VL Reasoning</div>',
                unsafe_allow_html=True,
            )

            if config.OBJECT_PREVIEW_PATH.exists():
                st.image(
                    str(config.OBJECT_PREVIEW_PATH),
                    caption="Object/Text detection preview",
                    use_container_width=True,
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

    st.markdown("<br>", unsafe_allow_html=True)

    next_button(
        "다음: 인물 선택 →",
        "select",
        button_type="primary",
    )

# ---------------------------------------------------------------------------
# 페이지: 인물 선택
# ---------------------------------------------------------------------------

def person_select_page() -> None:
    if not st.session_state.get("detection_done"):
        st.warning("먼저 탐지 단계를 완료해주세요.")
        if st.button("탐지 화면으로 이동"):
            st.session_state.page = "detect"
            st.rerun()
        return

    header()
    step_badge(3, 5, "인물 선택", "선명 유지 대상 설정")
    page_title(
        "선명하게 유지할 인물을 선택해주세요",
        "선택한 인물은 원본 그대로 유지되고, 나머지 인물과 개인정보는 자동으로 가려집니다.",
    )

    st.markdown(
        """
        <div class="hint-box">
            <span style="color:#2E994D;font-weight:500;">📌 선택한 인물은 선명하게 유지됩니다</span>
            <span style="color:#D1D1D6;margin:0 24px;">|</span>
            <span style="color:#B34040;font-weight:500;">선택하지 않은 인물과 개인정보는 자동으로 가려집니다</span>
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

    if "exclude_person_ids" not in st.session_state:
        st.session_state.exclude_person_ids = [
            person_id
            for person_id, person in person_db.items()
            if person.get("is_main", False)
        ]

    people = []

    for person_id, person in person_db.items():
        total_frames = (
            person.get("total_frames")
            or person.get("frame_count")
            or person.get("duration")
            or len(person.get("frames", []))
            or 0
        )

        people.append(
            {
                "person_id": person_id,
                "name": person_id,
                "duration": f"{total_frames} frames",
                "image_path": resolve_person_image_path(person),
            }
        )

    columns_per_row = 3 if len(people) <= 3 else 4

    for row_start in range(0, len(people), columns_per_row):
        row_people = people[row_start:row_start + columns_per_row]
        cols = st.columns(columns_per_row, gap="medium")

        for col, person in zip(cols, row_people):
            with col:
                selected = person["person_id"] in st.session_state.exclude_person_ids

                with st.container(border=True):
                    image_path = person["image_path"]

                    if image_path and image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="
                                width:180px;
                                height:180px;
                                border-radius:10px;
                                background:#F2F2F2;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:42px;
                                margin-bottom:12px;
                            ">👤</div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**{person['name']}**")
                    st.caption(person["duration"])

                    if selected:
                        if st.button(
                            "선명 유지 해제",
                            key=f"unselect_{person['person_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.exclude_person_ids.remove(person["person_id"])
                            invalidate_after_person_selection_change()
                            st.rerun()

                        st.caption("✓ 선명하게 유지됨")
                    else:
                        if st.button(
                            "선명하게 유지",
                            key=f"select_{person['person_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.exclude_person_ids.append(person["person_id"])
                            invalidate_after_person_selection_change()
                            st.rerun()

                        st.caption("가려질 예정")

    st.markdown("<br>", unsafe_allow_html=True)

    exclude_person_ids = st.session_state.get("exclude_person_ids", [])
    selected_text = ", ".join(exclude_person_ids) if exclude_person_ids else "없음"
    blur_people = [
        person["person_id"]
        for person in people
        if person["person_id"] not in exclude_person_ids
    ]
    blur_text = ", ".join(blur_people) if blur_people else "없음"

    with st.container(border=True):
        st.markdown(
            f"""
    <div style="padding:8px 0 2px 0;">
    <div style="font-size:14px;font-weight:700;color:#26262B;margin-bottom:26px;">선택 결과</div>
    <div style="font-size:13px;color:#2E994D;margin-bottom:6px;">✓ 선명하게 유지</div>
    <div class="small" style="margin-bottom:18px;">{selected_text}</div>
    <div style="font-size:13px;color:#B34040;margin-bottom:6px;">✓ 자동으로 가려짐</div>
    <div class="small" style="margin-bottom:18px;">{blur_text} 및 개인정보 항목</div>
    <div class="small">이 설정은 다음 단계에 적용됩니다.</div>
    </div>
            """,
            unsafe_allow_html=True,
        )

    next_button(
        "다음: 적용 내용 확인 →",
        "merged",
        button_type="primary",
    )

# ---------------------------------------------------------------------------
# 페이지: 적용 내용 확인
# ---------------------------------------------------------------------------

def merged_result_page() -> None:
    if not st.session_state.get("detection_done"):
        st.warning("먼저 영상 분석을 완료해주세요.")
        if st.button("분석 화면으로 이동"):
            st.session_state.page = "detect"
            st.rerun()
        return

    if "exclude_person_ids" not in st.session_state:
        st.warning("먼저 선명하게 유지할 인물을 선택해주세요.")
        if st.button("인물 선택 화면으로 이동"):
            st.session_state.page = "select"
            st.rerun()
        return

    header()
    step_badge(4, 5, "적용 내용 확인", "가려질 항목을 미리 확인합니다")
    page_title(
        "가려질 항목을 확인해주세요",
        "선명하게 유지할 인물을 제외하고, 자동으로 가려질 얼굴과 개인정보 항목을 정리했습니다.",
    )

    if not st.session_state.get("merge_done", False):
        exclude_ids = set(st.session_state.get("exclude_person_ids", []))

        with st.spinner("가려질 항목을 정리하는 중입니다..."):
            try:
                targets = run_face_export_and_merge(exclude_ids)
            except FileNotFoundError as e:
                st.error(f"파일 오류: {e}")
                return
            except Exception as e:
                st.error(f"적용 내용 생성 중 오류가 발생했습니다: {e}")
                return

        st.session_state.merge_done = True
        st.session_state.merged_targets = targets
        st.rerun()
    else:
        targets = st.session_state.get("merged_targets") or load_sam2_targets()

    face_targets = [t for t in targets if t.get("type") == "face"]
    object_targets = [t for t in targets if t.get("type") != "face"]
    keep_ids = st.session_state.get("exclude_person_ids", [])

    st.markdown(
        f"""
        <div class="white-card">
            <div style="font-size:15px;font-weight:700;margin-bottom:10px;">🧾 적용 내용 요약</div>
            <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
            <div class="small">선명하게 유지할 인물 <b style="float:right;color:#26262B;">{len(keep_ids)}명</b></div>
            <div class="small" style="margin-top:12px;">가려질 얼굴 <b style="float:right;color:#26262B;">{len(face_targets)}개</b></div>
            <div class="small" style="margin-top:12px;">가려질 개인정보 항목 <b style="float:right;color:#26262B;">{len(object_targets)}개</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 👤 가려질 얼굴")
        if face_targets:
            for target in face_targets:
                target_item(
                    target.get("id", "unknown"),
                    "얼굴",
                    "blue",
                    meta="선명 유지 대상에서 제외된 얼굴",
                )
        else:
            st.info("가려질 얼굴이 없습니다.")

    with right:
        st.markdown("### 🔲 가려질 개인정보")
        if object_targets:
            for target in object_targets:
                label = target.get("label") or target.get("id", "unknown")
                visible_text = target.get("visible_text", "")

                meta = "자동 감지된 개인정보 항목"
                if visible_text:
                    meta = f"인식된 텍스트: {visible_text}"

                target_item(
                    label,
                    "개인정보",
                    "red",
                    meta=meta,
                )
        else:
            st.info("가려질 개인정보 항목이 없습니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hint-box">
            위 항목들은 다음 단계에서 자동으로 가려집니다. 선명하게 유지할 인물을 바꾸고 싶다면 이전 단계로 돌아갈 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            "← 인물 다시 선택하기",
            use_container_width=True,
        ):
            st.session_state.page = "select"
            st.rerun()

    with col2:
        if st.button(
            "다음: 자동으로 가리기 →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "blur"
            st.rerun()

# ---------------------------------------------------------------------------
# 페이지: 최종 결과 확인 및 내보내기
# ---------------------------------------------------------------------------

def blur_result_page() -> None:
    if not st.session_state.get("merge_done"):
        st.warning("먼저 적용 내용을 확인해주세요.")
        if st.button("적용 내용 확인 화면으로 이동"):
            st.session_state.page = "merged"
            st.rerun()
        return

    header()

    if not st.session_state.get("blur_done", False):
        step_badge(5, 5, "개인정보 보호 처리 중", "선택 항목 자동 가림")
        page_title(
            "영상을 안전하게 처리하고 있습니다",
            "선택한 인물은 선명하게 유지하고, 나머지 얼굴과 개인정보 항목은 자동으로 가립니다.",
        )
    else:
        step_badge(5, 5, "최종 결과", "처리 완료 및 영상 내보내기")
        page_title(
            "처리된 영상을 확인하고 내보내세요",
            "개인정보 보호 처리가 완료된 영상을 확인한 뒤 다운로드하거나 새 영상을 편집할 수 있습니다.",
        )

    if not st.session_state.get("blur_done", False):
        with st.spinner("개인정보 보호 처리를 적용하는 중입니다..."):
            try:
                output_video_path = run_blur_stage(st.session_state.video_path)
            except (FileNotFoundError, ValueError) as e:
                st.error(f"설정 오류: {e}")
                st.info("이전 단계부터 다시 진행해주세요.")
                return
            except Exception as e:
                st.error(f"영상 처리 중 예상치 못한 오류가 발생했습니다: {e}")
                return

        st.session_state.output_video_path = str(output_video_path)
        st.session_state.blur_done = True
        st.rerun()

    status_success("처리 완료 — 결과 영상을 확인할 수 있습니다")

    targets = st.session_state.get("merged_targets") or load_sam2_targets()
    face_count = sum(1 for target in targets if target.get("type") == "face")
    object_count = len(targets) - face_count
    keep_count = len(st.session_state.get("exclude_person_ids", []))

    left, right = st.columns([2.2, 0.85], gap="large")

    with left:
        output_path = st.session_state.get("output_video_path")

        if output_path and Path(output_path).exists():
            st.video(output_path)
        else:
            preview_placeholder("처리된 영상 미리보기")

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div style="font-size:12px;font-weight:700;margin-bottom:8px;">처리 결과 요약</div>
                <div style="height:1px;background:#D1D1D6;margin-bottom:10px;"></div>
                <div class="small">가려진 얼굴 <b style="float:right;color:#26262B;">{face_count}개</b></div>
                <div class="small" style="margin-top:14px;">가려진 개인정보 항목 <b style="float:right;color:#26262B;">{object_count}개</b></div>
                <div class="small" style="margin-top:14px;">선명하게 유지한 인물 <b style="float:right;color:#26262B;">{keep_count}명</b></div>
                <div style="height:1px;background:#D1D1D6;margin:20px 0 10px;"></div>
                <div class="small" style="color:#2E994D;font-weight:500;">✓ 보호 영역 자동 감지 완료</div>
                <div class="small" style="color:#2E994D;font-weight:500;margin-top:12px;">✓ 영상 처리 완료</div>
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
            선택한 인물은 선명하게 유지되고, 가려야 할 얼굴과 개인정보 항목만 자동으로 처리되었습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    output_path = st.session_state.get("output_video_path")

    with col1:
        if output_path and Path(output_path).exists():
            with open(output_path, "rb") as f:
                st.download_button(
                    label="영상 내보내기",
                    data=f,
                    file_name="privacy_guard_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

    with col2:
        if st.button(
            "새 영상 편집하기",
            type="primary",
            use_container_width=True,
        ):
            reset_all_state()
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
    else:
        st.session_state.page = "landing"
        landing_page()