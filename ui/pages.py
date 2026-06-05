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

    if st.button("지금 시작하기 →", type="primary"):
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
        ("④", "적용 항목 확인", "블러 대상 검토"),
        ("⑤", "결과 내보내기", "최종 영상 저장"),
    ]

    cols = st.columns(5)

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
        '<div style="text-align:center;color:#737373;font-size:13px;margin-bottom:18px;">주요 처리 기술</div>',
        unsafe_allow_html=True,
    )

    chips = [
        "💬 프롬프트 입력",
        "👤 얼굴 탐지",
        "🔲 객체 탐지",
        "🚫 인물 제외",
        "🔀 JSON 통합",
        "🌀 Gaussian Blur",
    ]
    chip_row(chips)


# ---------------------------------------------------------------------------
# 페이지: 업로드
# ---------------------------------------------------------------------------

def upload_page() -> None:
    header()
    step_badge(1, 5, "영상 업로드 + 자연어 입력", "분석 전 영상과 명령을 함께 입력합니다")
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
        section_label("💡 명령 예시")
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
                "건물명·택배정보 \n가려줘",
                use_container_width=True,
                key="ex_delivery"
            ):
                st.session_state.prompt = "건물명·택배정보 가려줘"
                st.rerun()

        with col3:
            if st.button(
                "사람 빼고 전부 \n가려줘",
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

    if st.button("분석 시작 →", type="primary"):
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
    if not st.session_state.get("video_path"):
        st.warning("먼저 영상을 업로드해주세요.")
        if st.button("업로드 화면으로 이동"):
            st.session_state.page = "upload"
            st.rerun()
        return

    header()
    step_badge(2, 5, "병렬 탐지 진행", "얼굴 + 객체 동시 탐지")
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

    person_db = load_person_db()
    face_count = len(person_db)

    object_db = load_object_db()
    object_count = len(object_db) if isinstance(object_db, list) else 0

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size:14px;font-weight:700;margin-bottom:6px;">
                탐지 결과 준비 완료
            </div>
            <div class="page-desc" style="margin-bottom:18px;">
                얼굴 후보와 개인정보 객체 후보가 생성되었습니다. 다음 단계에서 블러 제외 인물을 선택합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Face Loop**")
            st.markdown(
                '<div class="small">SCRFD + ByteTrack + DBSCAN</div>',
                unsafe_allow_html=True,
            )

            if config.FACE_PREVIEW_PATH.exists():
                st.image(
                    str(config.FACE_PREVIEW_PATH),
                    caption="Face detection preview",
                    use_container_width=True,
                )
            else:
                detection_preview("face")

            st.markdown(
                f"""
                <div class="white-card" style="margin-top:10px;">
                    <div style="font-size:13px;font-weight:500;">👤 탐지된 인물</div>
                    <div class="small">{face_count}명 · person_db.json</div>
                    <div class="small">SCRFD + ByteTrack + DBSCAN</div>
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
    step_badge(3, 5, "인물 선택", "블러 제외 설정")
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

    # 3명 이하: 3열 / 4명 이상: 4열 기준으로 자동 줄바꿈
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
                            "블러 제외 해제",
                            key=f"unselect_{person['person_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.exclude_person_ids.remove(person["person_id"])
                            invalidate_after_person_selection_change()
                            st.rerun()

                        st.caption(f"✓ {person['person_id']} 블러 제외")
                    else:
                        if st.button(
                            "블러 제외",
                            key=f"select_{person['person_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.exclude_person_ids.append(person["person_id"])
                            invalidate_after_person_selection_change()
                            st.rerun()

                        st.caption("블러 적용 예정")

    st.markdown("<br>", unsafe_allow_html=True)

    exclude_person_ids = st.session_state.get("exclude_person_ids", [])
    selected_text = ", ".join(exclude_person_ids) if exclude_person_ids else "없음"
    blur_people = [
        person["person_id"]
        for person in people
        if person["person_id"] not in exclude_person_ids
    ]
    blur_text = ", ".join(blur_people) if blur_people else "없음"

    st.markdown(
        f"""
        <div class="card">
            <div style="font-size:12px;font-weight:700;margin-bottom:8px;">선택 요약</div>
            <div style="font-size:13px;color:#2E994D;">✓ {selected_text} → 블러 제외 (선명 유지)</div>
            <div style="font-size:13px;color:#B34040;margin-top:8px;">✕ {blur_text} + 탐지 객체 → 자동 블러 처리</div>
            <div class="small" style="margin-top:8px;">선택 결과는 다음 단계의 블러 적용 항목에 반영됩니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button(
        "다음: 자동 블러 처리 →",
        "merged",
        button_type="primary",
    )


# ---------------------------------------------------------------------------
# 페이지: 통합 결과
# ---------------------------------------------------------------------------

def merged_result_page() -> None:
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
    step_badge(4, 5, "블러 적용 항목 확인", "선택 인물 제외 기준으로 처리 대상을 정리합니다")
    page_title(
        "블러 처리될 항목을 확인하세요",
        "선택한 인물은 제외하고, 나머지 얼굴과 개인정보 객체가 블러 대상에 포함됩니다.",
    )

    if not st.session_state.get("merge_done", False):
        exclude_ids = set(st.session_state.get("exclude_person_ids", []))

        with st.spinner("블러 적용 항목을 정리하는 중입니다..."):
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

    face_targets = [t for t in targets if t.get("type") == "face"]
    object_targets = [t for t in targets if t.get("type") != "face"]
    exclude_ids = st.session_state.get("exclude_person_ids", [])

    st.markdown(
        f"""
        <div class="white-card">
            <div style="font-size:15px;font-weight:700;margin-bottom:10px;">🧾 블러 적용 영수증</div>
            <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
            <div class="small">블러 제외 인물 <b style="float:right;color:#26262B;">{len(exclude_ids)}명</b></div>
            <div class="small" style="margin-top:12px;">블러 적용 얼굴 <b style="float:right;color:#26262B;">{len(face_targets)}개</b></div>
            <div class="small" style="margin-top:12px;">블러 적용 객체/텍스트 <b style="float:right;color:#26262B;">{len(object_targets)}개</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### 👤 블러 적용 얼굴")
        if face_targets:
            for target in face_targets:
                target_item(
                    target.get("id", "unknown"),
                    "얼굴",
                    "blue",
                    meta="선택 인물 제외 후 블러 대상",
                )
        else:
            st.info("블러 처리할 얼굴 대상이 없습니다.")

    with right:
        st.markdown("### 🔲 블러 적용 객체/텍스트")
        if object_targets:
            for target in object_targets:
                label = target.get("label") or target.get("id", "unknown")
                visible_text = target.get("visible_text", "")

                meta = "box 포함"
                if visible_text:
                    meta = f"인식 텍스트: {visible_text}"

                target_item(
                    label,
                    "객체/텍스트",
                    "red",
                    meta=meta,
                )
        else:
            st.info("블러 처리할 객체/텍스트 대상이 없습니다.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hint-box">
            선택한 인물은 선명하게 유지되고, 위 항목만 SAM2 마스크 생성 및 Gaussian Blur 처리에 사용됩니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_button("다음: 자동 블러 처리 →", "blur", button_type="primary")


# ---------------------------------------------------------------------------
# 페이지: 최종 결과 확인 및 내보내기
# ---------------------------------------------------------------------------

def blur_result_page() -> None:
    if not st.session_state.get("merge_done"):
        st.warning("먼저 통합 결과를 생성해주세요.")
        if st.button("통합 결과 화면으로 이동"):
            st.session_state.page = "merged"
            st.rerun()
        return

    header()
    step_badge(5, 5, "최종 결과", "블러 처리 완료 및 내보내기")
    page_title(
        "블러 처리 결과를 확인하고 영상을 내보냅니다",
        "최종 결과 영상을 확인한 뒤 다운로드하거나 새 영상을 편집할 수 있습니다.",
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

    targets = st.session_state.get("merged_targets") or load_sam2_targets()
    face_count = sum(1 for target in targets if target.get("type") == "face")
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