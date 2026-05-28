# ui/components.py

from __future__ import annotations

import json
from typing import Any, Iterable

import streamlit as st


def header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <span class="brand">Gari-Get-That</span>
                <span class="brand-subtitle">맥락 이해 기반 프라이버시 보호 AI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_badge(step: int, total: int, title: str, desc: str = "") -> None:
    text = f"STEP {step} / {total} {title}"
    if desc:
        text += f" — {desc}"

    st.markdown(
        f"""
        <div class="step-badge">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, desc: str | None = None) -> None:
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="page-desc">{desc}</div>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def hint_box(text: str) -> None:
    st.markdown(f'<div class="hint-box">{text}</div>', unsafe_allow_html=True)


def status_success(text: str) -> None:
    st.markdown(f'<div class="status-success">✓ {text}</div>', unsafe_allow_html=True)


def status_running(text: str) -> None:
    st.markdown(f'<div class="status-running">⚡ {text}</div>', unsafe_allow_html=True)


def chip(text: str) -> None:
    st.markdown(f'<span class="chip">{text}</span>', unsafe_allow_html=True)


def chip_row(items: Iterable[str]) -> None:
    html = "".join([f'<span class="chip">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="icon-badge">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_card(icon: str, title: str) -> None:
    st.markdown(
        f"""
        <div class="pipeline-step">
            <div class="pipeline-icon">{icon}</div>
            <div class="pipeline-name">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_viz() -> None:
    st.markdown(
        """
        <div class="white-card">
            <div style="font-size:12px;font-weight:700;margin-bottom:12px;">
                AI 처리 6단계 흐름
            </div>
            <div style="height:1px;background:#D1D1D6;margin-bottom:12px;"></div>
            <div class="pipeline-grid">
                <div class="pipeline-step">
                    <div class="pipeline-icon">💬</div>
                    <div class="pipeline-name">자연어 입력</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">⚡</div>
                    <div class="pipeline-name">병렬 탐지</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">🔀</div>
                    <div class="pipeline-name">JSON 통합</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">🎯</div>
                    <div class="pipeline-name">인물 선택</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">🌀</div>
                    <div class="pipeline-name">통합 블러</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-icon">✂</div>
                    <div class="pipeline-name">내보내기</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def json_panel(title: str, data: Any) -> None:
    if isinstance(data, str):
        content = data
    else:
        content = json.dumps(data, ensure_ascii=False, indent=2)

    escaped = (
        content.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    st.markdown(
        f"""
        <div class="dark-panel">
            <div style="color:#99CCFF;font-weight:500;margin-bottom:12px;">
                {title}
            </div>
            <div style="height:1px;background:#40424D;margin:0 -16px 12px -16px;"></div>
            <pre style="white-space:pre-wrap;margin:0;color:#CCEBCC;">{escaped}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


def target_item(name: str, target_type: str, color: str = "blue", meta: str = "bbox 포함") -> None:
    if color == "red":
        bar_color = "#8C1F2E"
        tag_bg = "rgba(140, 31, 46, 0.18)"
        tag_color = "#8C1F2E"
    else:
        bar_color = "#4D80E6"
        tag_bg = "#EBF2FF"
        tag_color = "#4D80E6"

    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #D1D1D6;
            border-radius:8px;
            min-height:64px;
            margin-bottom:10px;
            position:relative;
            overflow:hidden;
        ">
            <div style="
                position:absolute;
                left:0;
                top:0;
                width:4px;
                height:100%;
                background:{bar_color};
            "></div>
            <div style="padding:12px 16px 10px 20px;">
                <div style="font-size:14px;font-weight:700;color:#26262B;margin-bottom:6px;">
                    {name}
                </div>
                <span style="
                    display:inline-block;
                    background:{tag_bg};
                    color:{tag_color};
                    border-radius:4px;
                    padding:3px 7px;
                    font-size:10px;
                    font-weight:500;
                ">
                    {target_type}
                </span>
                <span style="
                    float:right;
                    color:#808085;
                    font-size:11px;
                    margin-top:3px;
                ">
                    {meta}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def person_card(
    name: str,
    duration: str,
    selected: bool = False,
    key: str | None = None,
) -> bool:
    css_class = "person-card selected" if selected else "person-card"
    badge_text = "블러 제외" if selected else "블러 적용 예정"
    badge_bg = "#4D80E6" if selected else "#F5F5F7"
    badge_color = "#FFFFFF" if selected else "#808085"
    badge_border = "none" if selected else "1px solid #D1D1D6"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="person-thumb">👤</div>
            <div class="person-name">{name}</div>
            <div class="person-meta">{duration}</div>
            <div style="
                margin-top:14px;
                background:{badge_bg};
                color:{badge_color};
                border:{badge_border};
                border-radius:6px;
                padding:6px 0;
                text-align:center;
                font-size:12px;
                font-weight:500;
            ">
                {badge_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return st.checkbox(
        f"{name} 블러 제외",
        value=selected,
        key=key or f"person_{name}",
    )


def preview_placeholder(label: str = "영상 미리보기") -> None:
    st.markdown(
        f"""
        <div style="
            height:260px;
            background:#EBEBF0;
            border-radius:8px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#808085;
            font-size:13px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def detection_preview(kind: str = "face") -> None:
    if kind == "face":
        color = "#4D80E6"
        bg = "#EBF2FF"
        labels = ["얼굴 A", "얼굴 B", "얼굴 C"]
    else:
        color = "#8C1F2E"
        bg = "rgba(140, 31, 46, 0.25)"
        labels = ["간판", "건물명", "택배 정보"]

    st.markdown(
        f"""
        <div style="
            height:240px;
            background:#EBEBF0;
            border-radius:8px;
            position:relative;
            overflow:hidden;
        ">
            <div style="
                position:absolute;
                left:12px;
                top:10px;
                font-size:10px;
                color:#808085;
            ">
                영상 미리보기
            </div>

            <div style="
                position:absolute;
                left:30px;
                top:50px;
                width:86px;
                height:100px;
                background:{bg};
                border:1.5px solid {color};
                border-radius:4px;
            "></div>
            <div style="
                position:absolute;
                left:32px;
                top:34px;
                font-size:10px;
                font-weight:500;
                color:{color};
            ">{labels[0]}</div>

            <div style="
                position:absolute;
                left:170px;
                top:44px;
                width:90px;
                height:106px;
                background:{bg};
                border:1.5px solid {color};
                border-radius:4px;
            "></div>
            <div style="
                position:absolute;
                left:172px;
                top:28px;
                font-size:10px;
                font-weight:500;
                color:{color};
            ">{labels[1]}</div>

            <div style="
                position:absolute;
                left:294px;
                top:70px;
                width:82px;
                height:86px;
                background:{bg};
                border:1.5px solid {color};
                border-radius:4px;
            "></div>
            <div style="
                position:absolute;
                left:296px;
                top:54px;
                font-size:10px;
                font-weight:500;
                color:{color};
            ">{labels[2]}</div>

            <div style="
                position:absolute;
                left:0;
                bottom:0;
                width:100%;
                height:30px;
                background:#EBEBF0;
                border-top:1px solid #D1D1D6;
                padding:8px 10px;
                font-size:11px;
                color:#26262B;
            ">
                ▶ 00:00 / 01:32
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def timeline_row(label: str, segments: list[tuple[int, int]], object_type: bool = False) -> None:
    segment_html = ""

    for start, width in segments:
        cls = "timeline-segment object" if object_type else "timeline-segment"
        segment_html += f"""
        <div class="{cls}" style="left:{start}%;width:{width}%;"></div>
        """

    st.markdown(
        f"""
        <div class="timeline-row">
            <div class="timeline-label">{label}</div>
            <div class="timeline-track">
                {segment_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_button_placeholder() -> None:
    st.button("⬇ 영상 다운로드")


def next_button(label: str, next_page: str) -> None:
    if st.button(label):
        st.session_state.page = next_page
        st.rerun()