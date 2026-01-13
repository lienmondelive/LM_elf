import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import csv
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import base64
from io import BytesIO

def go_next(selected, q_data):
    page = st.session_state.page

    # 撤銷上一題分數
    prev_answer = st.session_state.answers[page]
    if prev_answer is not None:
        for attr, val in q_data["options"][prev_answer].items():
            st.session_state.scores[attr] -= val

    # 加上新分數
    st.session_state.answers[page] = selected
    for attr, val in q_data["options"][selected].items():
        st.session_state.scores[attr] += val

    # 換頁
    st.session_state.page += 1

@st.cache_data
def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def render_story_text(text):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def score_to_group(score_dict, group_config):
    group_scores = {}

    for group, cfg in group_config.items():
        attrs = cfg["attrs"]
        group_scores[group] = sum(
            score_dict.get(attr, 0) for attr in attrs
        )

    return group_scores

def max_gap_penalty(my_scores, elf_scores, features, max_score=10):
    gaps = [
        abs(float(my_scores[f]) - float(elf_scores[f]))
        for f in features
    ]
    max_gap = max(gaps) / max_score  # 0 ~ 1
    return max_gap

def clamp(x, min_v, max_v):
    return max(min_v, min(x, max_v))

@st.cache_data
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fig_to_base64(fig):
    svg_bytes = fig.to_image(format="svg")
    return base64.b64encode(svg_bytes).decode("utf-8")

def normalize_group_scores(group_scores, group_config):
    normalized = {}

    for group, score in group_scores.items():
        min_v = group_config[group]["min"]
        max_v = group_config[group]["max"]

        s = clamp(score, min_v, max_v)

        norm = round(1 + 9 * (score - min_v) / (max_v - min_v))
        if norm>10: norm=10
        normalized[group] = norm

    return normalized

def shared_strength(my_scores, elf_scores, features, max_score=10):
    mins = [
        min(float(my_scores[f]), float(elf_scores[f]))
        for f in features
    ]
    return np.mean(mins) / max_score   # 0 ~ 1

def cosine_similarity(my_scores, elf_scores, features):
    """
    形狀相似度（只看比例分布，不看強度）
    回傳範圍：0 ~ 1
    """
    v1 = np.array([float(my_scores[f]) for f in features])
    v2 = np.array([float(elf_scores[f]) for f in features])

    # 避免除以 0
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def absolute_similarity(my_scores, elf_scores, features, max_score=10):
    """
    絕對值接近度（看強不強、差多不多）
    回傳範圍：0 ~ 1
    """
    diffs = []

    for f in features:
        diff = abs(float(my_scores[f]) - float(elf_scores[f]))
        diffs.append(diff)

    avg_diff = np.mean(diffs)

    # 差距越小，分數越高
    score = 1 - (avg_diff / max_score)

    # 保底
    return max(score, 0.0)

def new_similarity(my_scores, elf_scores, features):
    if is_balanced_profile(my_scores, features):
        return match_for_balanced_user(my_scores, elf_scores, features)
    else:
        return match_for_peaked_user(my_scores, elf_scores, features)

def match_for_balanced_user(my_scores, elf_scores, features):
    shape = cosine_similarity(my_scores, elf_scores, features)
    abs_sim = absolute_similarity(my_scores, elf_scores, features, max_score=10)
    strength = shared_strength(my_scores, elf_scores, features)

    base = 0.6 * shape + 0.4 * abs_sim
    score = base * (0.85 + 0.15 * strength)

    # ===== 校正開始（就在這）=====
    max_gap = max(
        abs(float(my_scores[f]) - float(elf_scores[f]))
        for f in features
    ) / 10  # 0~1

    penalty = max(0, max_gap - 0.25)     # 超過 25% 才扣
    score = score * (1 - 0.5 * penalty)  # 最多扣約 20%
    # ===== 校正結束 =====

    return max(score, 0.0)


def match_for_peaked_user(my_scores, elf_scores, features, top_k=2):
    top_feats = sorted(
        features,
        key=lambda f: my_scores[f],
        reverse=True
    )[:top_k]

    shape = cosine_similarity(my_scores, elf_scores, top_feats)
    strength = shared_strength(my_scores, elf_scores, top_feats)

    score = shape * (0.8 + 0.2 * strength)

    # ===== 校正（同樣放在回傳前）=====
    max_gap = max(
        abs(float(my_scores[f]) - float(elf_scores[f]))
        for f in features
    ) / 10

    penalty = max(0, max_gap - 0.25)
    score = score * (1 - 0.5 * penalty)
    # ===== 校正結束 =====

    return max(score, 0.0)


def is_balanced_profile(scores, features, std_threshold=1.2):
    vals = np.array([scores[f] for f in features])
    return np.std(vals) <= std_threshold

# === 功能：根據答案重新計算分數 ===

def recalc_scores():
    st.session_state.scores = default_scores
    for ans in st.session_state.answers.values():
        for attr, val in ans.items():
            st.session_state.scores[attr] += val

def find_best_elf(my_scores, elfs, features):
    best_score = -1
    best_elf = None
    best_row = None
    recommand_sort = []

    for elf in elfs:
    # for row in candidate_elves:
        elf_name = elf['elf_name']
        suit_score = new_similarity(my_scores=my_scores,elf_scores=elf,features=features)
        recommand_sort.append({"elf_name":elf_name,"suit_score":suit_score})

        # if diff_score < best_score:
        if suit_score > best_score:
            best_score = suit_score
            best_elf = elf_name
            best_row = elf

    recommand_sort = sorted(recommand_sort, key=lambda x: x['suit_score'], reverse=True)
    elf_name = best_row.pop('elf_name', None)
    return best_elf, best_row, best_score, recommand_sort

@st.cache_data
def load_elves(elf_csv_path, feature):
    elves = []
    with open(elf_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = row.copy()           # 避免 DictReader cache 問題
            # row[feature] = float(row[feature])
            elves.append(row)
    return elves



base_folder = os.path.dirname(os.path.abspath(__file__))
config_folder = os.path.join(base_folder, "config")

questions = load_json(os.path.join(config_folder, "questions.json"))
stories = load_json(os.path.join(config_folder, "stories.json"))
default_scores = load_json(os.path.join(config_folder, "attributes.json"))
group_config = load_json(os.path.join(config_folder, "group_config.json"))
elves_info = load_json(os.path.join(config_folder, "elves_info.json"))
# elves_info = load_json(os.path.join(config_folder, "elves_info.json"))
features = list(group_config.keys())

elves_data_path = os.path.join(config_folder, "elves.csv")
elves_data = load_elves(elves_data_path, feature="score")


# === 初始化+題目設定 ===

total_pages = len(questions) + 2  # 包含開頭(0) + 結果頁(最後一頁)

if "page" not in st.session_state:
    st.session_state.page = 0

if "scores" not in st.session_state:
    st.session_state.scores = default_scores.copy()

if "answers" not in st.session_state:
    st.session_state.answers = [None] * total_pages

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "radar_base64" not in st.session_state:
    st.session_state.radar_base64 = None

if "result_base64" not in st.session_state:
    st.session_state.result_base64 = None

if "elf_base64" not in st.session_state:
    st.session_state.elf_base64 = None

if "best_elf" not in st.session_state:
    st.session_state.best_elf = None

if "is_calculating" not in st.session_state:
    st.session_state.is_calculating = False

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False

if "icon_yt_base64" not in st.session_state:
    st.session_state.icon_yt_base64 = img_to_base64(os.path.join("pictures", "youtube.png"))
if "icon_twitch_base64" not in st.session_state:
    img_to_base64(os.path.join("pictures", "twitch.png"))
    st.session_state.icon_twitch_base64 = img_to_base64(os.path.join("pictures", "twitch.png"))
if "icon_x_base64" not in st.session_state:
    st.session_state.icon_x_base64 = img_to_base64(os.path.join("pictures", "twitter.png"))

is_mobile = False

replacements = {
    "\n\n": "<br>",
    "\n": "<br>",
    "XXX": st.session_state.user_name
}

# === 頁面 0：開始畫面 ===

if st.session_state.page == 0:
    st.title("💫 L.M. Live 守護精靈測驗")
    # st.subheader("✨ 誰會成為你的連結者？")
    st.write("")
    st.write("L.M. Live 是法文 **Lien Monde Live** 的縮寫，譯「連結世界的直播」。")
    st.write("快來測驗看看 L.M. Live 中，誰最適合你吧！")
    st.markdown("<br>", unsafe_allow_html=True)

    st.session_state.user_name = st.text_input(
        "該怎麼稱呼你呢？",
        placeholder="我是誰"
    )
        
    st.session_state.view_mode = st.radio(
        "請選擇顯示方式",
        ["電腦版","手機版"],
        index=0,
        horizontal=True
    )

    st.caption("👇點擊下方按鈕開始你的測驗旅程")

    if st.button("開始測驗"):
        if st.session_state.user_name.strip() == "":
            st.warning("請先輸入名字再開始測驗")
        else:
            st.session_state.user_name = st.session_state.user_name.strip()
            st.session_state.page += 1
            st.rerun()


# === 問題頁面 ===
elif 1 <= st.session_state.page <= len(questions):
# elif 1 <= st.session_state.page <= 2:
    q_index = st.session_state.page - 1
    q_data = questions[q_index]
    a_story = stories[q_index]

    for block in a_story.get("content", []):
        block_type = block.get("type")

        # 文字區塊
        if block_type == "text":
            text_html = render_story_text(block.get("value", ""))

            st.markdown(
                f"""
                <div style="
                    font-size: 16px;
                    line-height: 1.6;
                    color: #444;
                    margin-bottom: 1.2em;
                ">
                    {text_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 圖片區塊
        elif block_type == "image":

            pictures = block.get("value", [])
            if len(pictures)==1:
                image_path = os.path.join("pictures", pictures[0])
                st.image(image_path, use_container_width=True)

    #----------------------------------------------------------

    # 淡淡的滿版分隔線
    st.markdown(
        """
        <div style="
            width: 100%;
            height: 1px;
            background: linear-gradient(
                to right,
                rgba(0,0,0,0),
                rgba(0,0,0,0.15),
                rgba(0,0,0,0)
            );
            margin: 1.2em 0 1em 0;
        "></div>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"第 {q_index + 1} / {len(questions)} 題")
    st.markdown(f"**{q_data['question']}**")


    selected = st.radio(
        label=f"q{st.session_state.page}_options",
        options=list(q_data["options"].keys()),
        index=None,
        key=f"q{st.session_state.page}",
        label_visibility="collapsed"
    )
    
    
    ########################################### pre calculate ###########################################

    if st.session_state.page == len(stories)-1:
    # if st.session_state.page == 2:

        if not st.session_state.result_ready:
            st.session_state.is_calculating = True

        # === 所有重算只做一次 ===

            attrs_sort = sorted(
                st.session_state.scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            st.session_state.top_attrs = [k for k, v in attrs_sort if v > 0]

            group_scores = score_to_group(st.session_state.scores, group_config)
            st.session_state.normalize_group_scores = normalize_group_scores(group_scores, group_config)

            best_elf, best_row, best_score, recommand_sort = find_best_elf(
                st.session_state.normalize_group_scores, elves_data, features
            )

            st.session_state.best_elf = best_elf
            st.session_state.best_score = best_score
            st.session_state.recommand_sort = recommand_sort

            my_r = [st.session_state.normalize_group_scores[f] for f in features]
            best_r = [best_row[f] for f in features]
            # 雷達圖
            my_df = pd.DataFrame(dict(
                r=my_r + [my_r[0]],
                theta=features + [features[0]]
            ))

            best_df = pd.DataFrame(dict(
                r=best_r + [best_r[0]],
                theta=features + [features[0]]
            ))

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=my_df['r'],
                theta=my_df['theta'],
                name="你的屬性",
                line=dict(shape='linear', color='red'),
                fill='none'
            ))
            fig.add_trace(go.Scatterpolar(
                r=best_df['r'],
                theta=best_df['theta'],
                name=st.session_state.best_elf,
                line=dict(shape='linear'),
                fill='none'
            ))

            fig.update_layout(
                width=420,
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    font=dict(size=15, color="#222"),
                    orientation="h",
                    yanchor="bottom",
                    y=1.35,          # ⭐ 拉到圖外
                    xanchor="center",
                    x=0.5
                ),
                polar=dict(
                    radialaxis=dict(
                        showticklabels=False,      # ❌ 不顯示內圈數字
                        ticks='',                  # ❌ 不顯示刻度短線
                        range=[0, 10],
                        gridcolor="rgba(0,0,0,0.15)",
                        tickfont=dict(size=14),   # ← 半徑刻度字
                        autorange=False
                    ),
                    angularaxis=dict(
                        rotation=90,
                        showticklabels=True,         # ⭐ 關掉屬性文字
                        tickfont=dict(
                            size=18,            # ⭐ 屬性名稱字體大小
                            color="#222",
                            family="Noto Sans TC SemiBold, Noto Sans TC, Arial"
                        )
                    )
                ),
                margin=dict(
                    l=140,
                    r=140,
                    t=160,   # 上面要留 legend
                    b=140
                ),
                showlegend=True
            )


            # 圖片 base64
            # st.session_state.radar_base64 = fig_to_base64(fig)
            st.session_state.radar_fig = fig
            st.session_state.result_base64 = img_to_base64(os.path.join("pictures", "result.png"))
            elf_path = os.path.join("pictures/elfs", f"{best_elf}.png")
            # elf_path = f"pictures/elfs/{best_elf}.png"
            st.session_state.elf_base64 = img_to_base64(elf_path) if os.path.exists(elf_path) else None
            st.session_state.result_ready = True
            st.session_state.is_calculating = False
    
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "上一題",
            disabled=(
                st.session_state.page == 0
                or st.session_state.is_calculating
                or st.session_state.result_ready
            ),
            on_click=lambda: setattr(
                st.session_state, "page", st.session_state.page - 1
            )
        )

    with col2:
        st.button(
            "下一題",
            disabled=(selected is None),
            on_click=go_next,
            args=(selected, q_data)
        )


# === 結果頁 ===
# elif st.session_state.page == 3:
elif st.session_state.page == len(stories):
    q_index = st.session_state.page - 1
    a_story = stories[q_index]
    # a_story = stories[11]
    optional_map = a_story.get("optional", None)
        
    for block in a_story.get("content", []):
        block_type = block.get("type")

        # 文字區塊
        if block_type == "text":
            text_html = render_story_text(block.get("value", ""))
            if optional_map is not None:
                if st.session_state.best_elf in ("八上菟比", "蕾古露絲"):
                    optional_text = optional_map["一期生"]
                elif st.session_state.best_elf in optional_map.keys():
                    optional_text = optional_map[st.session_state.best_elf]
                else:
                    optional_text = optional_map["二期生"]

                if optional_text:
                    text_html = text_html.replace("optional", optional_text)
                else:
                    text_html = text_html.replace("optional", "")

            text_html = text_html.replace("elf", st.session_state.best_elf)


            st.markdown(
                f"""
                <div style="
                    font-size: 16px;
                    line-height: 1.6;
                    color: #444;
                    margin-bottom: 1em;
                ">
                    {text_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 圖片區塊
        elif block_type == "image":
            pictures = a_story.get("picture", None)
            if pictures is not None and len(pictures)==1:
                image_path = os.path.join("pictures", pictures[0])
                st.image(image_path, use_container_width=True)

        

#---------------------------------------------------------------------------------------------
    
    # if not st.session_state.show_result:
        result_placeholder = st.empty()
        with result_placeholder:
            if st.button("🔍 讀取測驗結果"):
                st.markdown("⏳ 正在打開測驗結果......")
                st.session_state.show_result = True

    radar_html = st.session_state.radar_fig.to_html(
        include_plotlyjs="cdn",
        full_html=False
    )
    if st.session_state.show_result:
        result_placeholder.empty()

        if st.session_state.view_mode == "手機版":
            components.html(
            f"""
                <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>

                    <button
                        onclick="captureResult()"
                        style="
                            position: fixed;
                            bottom: 20px;
                            left: 50%;
                            transform: translateX(-50%);
                            z-index: 9999;
                            padding: 12px 18px;
                            border-radius: 999px;
                            border: none;
                            background: #222;
                            color: #fff;
                            font-size: 14px;
                            cursor: pointer;
                        "
                    >
                    📸 儲存結果
                    </button>

                    <script>
                    function captureResult() {{
                        const target = document.getElementById("result-capture");

                        html2canvas(target, {{
                            backgroundColor: "#ffffff",
                            scale: 2,          // ⭐ 解析度（手機一定要 >=2）
                            useCORS: true
                        }}).then(canvas => {{
                            const link = document.createElement("a");
                            link.download = "測驗結果.png";
                            link.href = canvas.toDataURL("image/png");
                            link.click();
                        }});
                    }}
                </script>

            <style>
                .result-root {{
                position: relative;
                padding: 30px;
                }}

                .mobile-card {{
                padding: 20px;
                text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC", sans-serif;
                }}

                .mobile-title {{
                font-size: 28px;
                font-weight: 900;
                margin-bottom: 12px;
                }}

                .mobile-name {{
                font-size: 16px;
                margin-bottom: 6px;
                }}

                .mobile-main {{
                font-size: 18px;
                font-weight: 800;
                margin: 14px 0;
                }}

                .mobile-sub {{
                font-size: 16px;
                color: #555;
                line-height: 1.6;
                }}

                .mobile-icons {{
                margin-top: 20px;
                display: flex;
                justify-content: center;
                gap: 20px;
                }}

                .mobile-elf {{
                margin: 16px auto;
                width: 70%;
                max-width: 320px;
                }}

                .mobile-elf img {{
                width: 100%;
                height: auto;
                display: block;
                }}

            </style>

            <div class="result-root">

            <!-- ===================== 手機版 ===================== -->
            <div class="mobile-card" id="result-capture">
                <div class="mobile-card">

                    <div class="mobile-title">測驗結果</div>

                    <div class="mobile-name">{st.session_state.user_name}<br>{'、'.join(st.session_state.top_attrs)}</div>

                    <div class="mobile-main">
                        守護精靈主推<br>
                        <strong>{st.session_state.best_elf}</strong> | 適配度：{int(round(st.session_state.best_score*100,0))}%
                    </div>

                    <div class="mobile-name">{elves_info[st.session_state.best_elf]["slogan"]}</div>

                    <div class="mobile-elf">
                        <img src="data:image/png;base64,{st.session_state.elf_base64}">
                    </div>

                    <div class="mobile-icons">
                        <a href="{elves_info[st.session_state.best_elf]["social"]["youtube"]}" target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_yt_base64}" width="36">
                        </a>
                        <a href="{elves_info[st.session_state.best_elf]["social"]["twitch"]}" target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_twitch_base64}" width="36">
                        </a>
                        <a href="{elves_info[st.session_state.best_elf]["social"]["twitter"]}" target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_x_base64}" width="36">
                        </a>
                    </div>

                </div>
            </div>

            """,
            height=1000
        )

        else:
            components.html(
                f"""

                <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>

                    <button
                        onclick="captureResult()"
                        style="
                            position: fixed;
                            bottom: 20px;
                            left: 50%;
                            transform: translateX(-50%);
                            z-index: 9999;
                            padding: 12px 18px;
                            border-radius: 999px;
                            border: none;
                            background: #222;
                            color: #fff;
                            font-size: 14px;
                            cursor: pointer;
                        "
                    >
                    📸 儲存結果
                    </button>

                    <script>
                        function captureResult() {{
                            const target = document.getElementById("result-capture");

                            html2canvas(target, {{
                                backgroundColor: "#ffffff",
                                scale: 2,
                                useCORS: true,

                                // ⭐⭐⭐ 關鍵修正 ⭐⭐⭐
                                scrollY: -window.scrollY,
                                scrollX: -window.scrollX,
                                y: 60,
                                windowWidth: document.documentElement.clientWidth,
                                windowHeight: target.scrollHeight
                            }}).then(canvas => {{
                                const link = document.createElement("a");
                                link.download = "測驗結果.png";
                                link.href = canvas.toDataURL("image/png");
                                link.click();
                            }});
                        }}
                    </script>

                <style>
                    .result-root {{
                    position: relative;
                    padding: 30px;
                    }}
                </style>

                <div class="result-root" id="result-capture">

                <!-- ===================== 桌機版 ===================== -->
                <div>
                    <div style="
                        position: relative;
                        padding: 30px;          /* ⭐ 關鍵：安全邊距 */
                    ">
                    <img src="data:image/png;base64,{st.session_state.result_base64}"
                        style="
                            width:100%;
                            display:block;
                            transform: scale(1.15);
                            transform-origin: center top;
                            z-index:1;
                    ">
                    
                    <!-- 標題 -->
                    <div style="
                        position:absolute;
                        top:17%;
                        left:10%;
                        width:80%;
                        text-align:center;
                        font-size:40px;
                        font-weight:1000;
                        color:#1f1f1f;
                        z-index: 2;
                    ">
                        測驗結果
                    </div>

                    <!-- 名字 & 屬性 -->
                    <div style="
                        position:absolute;
                        top:24%;
                        left:12%;
                        font-size:22px;
                        color:#333;
                        z-index: 2;
                    ">
                        <strong>名字：</strong><span style="font-size:24px;">{st.session_state.user_name}</span> <br>
                        <strong>屬性：</strong><span style="font-size:24px;">{'、'.join(st.session_state.top_attrs)}</span> <br>
                        <strong>守護精靈主推：</strong><span style="font-size:24px;">{st.session_state.best_elf} | 適配度 {int(round(st.session_state.best_score*100,0))}%</span> <br>
                        <span style="font-size:18px;">（其他推薦：{st.session_state.recommand_sort[1]["elf_name"]} | 適配度 {int(round(st.session_state.recommand_sort[1]["suit_score"]*100,0))}%、{st.session_state.recommand_sort[2]["elf_name"]} | 適配度 {int(round(st.session_state.recommand_sort[2]["suit_score"]*100,0))}%）</span> <br>

                    </div>
                        
                    <!-- 雷達圖 -->
                    <div style="
                        position:absolute;
                        bottom:3%;
                        left:37%;
                        width:58%;
                        max-width:420px;   /* ⭐ 關鍵 */
                        transform: scale(0.9);       /* ⭐ 微調 */
                        transform-origin: center;
                        z-index:2;
                        overflow: visible;
                    ">
                        {radar_html}
                    </div>

                    

                    <!-- 精靈圖片 -->
                    <img src="data:image/png;base64,{st.session_state.elf_base64}"
                        style="
                            position:absolute;
                            bottom:3%;
                            left:-15%;
                            width:90%;
                            z-index: 2;
                        ">

                    <!-- 精靈標語 -->
                    <div style="
                        position:absolute;
                        top:45%;
                        right:-6%;
                        width:50%;
                        font-size:24px;
                        line-height:1.6;
                        color:#333;
                        z-index: 2;
                    ">
                        <strong>{st.session_state.best_elf}：</strong><br>
                    </div>
                    <!-- 精靈標語 -->
                    <div style="
                        position:absolute;
                        top:49%;
                        right:-6%;
                        width:50%;
                        font-size:18px;
                        line-height:1.6;
                        color:#333;
                        z-index: 2;
                    ">
                        {elves_info[st.session_state.best_elf]["slogan"]}
                    </div>

                    <div style="
                        position:absolute;
                        bottom:6%;
                        right:38%;
                        display:flex;
                        gap:18px;
                        z-index:5;
                        transform: scale(1.5);
                    ">
                        <a href={elves_info[st.session_state.best_elf]["social"]["youtube"]} target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_yt_base64}"
                                width="32">
                        </a>
                    </div>
                    <div style="
                        position:absolute;
                        bottom:6%;
                        right:26%;
                        display:flex;
                        gap:18px;
                        z-index:5;
                        transform: scale(1.15);
                    ">
                        <a href={elves_info[st.session_state.best_elf]["social"]["twitch"]} target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_twitch_base64}"
                                width="32">
                        </a>
                    </div>
                    <div style="
                        position:absolute;
                        bottom:6%;
                        right:15%;
                        display:flex;
                        gap:18px;
                        z-index:5;
                        transform: scale(1.1);
                    ">
                        <a href={elves_info[st.session_state.best_elf]["social"]["twitter"]} target="_blank">
                            <img src="data:image/png;base64,{st.session_state.icon_x_base64}"
                                width="32">
                        </a>
                    </div>
                </div>

                """,
                height=1400
            )

        st.markdown(
            """
            <div style="
                text-align: center;
                font-size: 16px;
                color: #555;
                margin-top: 16px;
            ">
                重新測驗請直接重整網頁
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        

        st.markdown(
                f"""
                <div style="
                    font-size: 16px;
                    line-height: 1.6;
                    color: #444;
                    margin-bottom: 1em;
                ">
                    Released by L.M. Live <br>
                    Designed & built by Tipsyuu <br>
                    Visual & assets assisted by AI <br>
                    Special thanks to Eric and everyone at L.M.

                </div>
                """,
                unsafe_allow_html=True
            )