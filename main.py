import streamlit as st
import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(my_scores, cutee_scores, features):
    v1 = np.array([float(my_scores[f]) for f in features])
    v2 = np.array([float(cutee_scores[f]) for f in features])

    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # print(f"餘弦相似度: {cos_sim:.3f}")
    return cos_sim


# === 功能：根據答案重新計算分數 ===
def recalc_scores():
    st.session_state.scores = default_scores
    for ans in st.session_state.answers.values():
        for attr, val in ans.items():
            st.session_state.scores[attr] += val


def find_best_cutee(my_scores,features):
    best_score = 0
    best_cutee = None
    best_row = None
    recommand_sort = []

    with open('cutee.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            cutee_name = row['cutee_name']

            # score 1
            # diff_score = 0
            # for feature in features:
            #     diff_score += abs(float(my_scores[feature])-float(row[feature]))
            
            # score 2 - cos
            suit_score = cosine_similarity(my_scores, row, features)
            recommand_sort.append({"cutee_name":cutee_name,"suit_score":suit_score})

            # if diff_score < best_score:
            if suit_score > best_score:
                best_score = suit_score
                best_cutee = cutee_name
                best_row = row

            
            print(cutee_name, f"餘弦相似度: {suit_score:.3f}")

    recommand_sort = sorted(recommand_sort, key=lambda x: x['suit_score'], reverse=True)
    # print("row format", best_row)
    cutee_name = best_row.pop('cutee_name', None)
    return best_cutee, best_row, best_score, recommand_sort

# st.markdown("<h1 style='text-align: center;'> ✨L.M. Live—誰是你的守護精靈？✨ </h1>", unsafe_allow_html=True)


# st.set_page_config(page_title="多屬性加分問卷", page_icon="✨", layout="centered")


questions = load_json("questions.json")
default_scores = load_json("attributes.json")
features = list(default_scores.keys())


# === 初始化+題目設定 ===

total_pages = len(questions) + 2  # 包含開頭(0) + 結果頁(最後一頁)

if "page" not in st.session_state:
    st.session_state.page = 0

if "scores" not in st.session_state:
    st.session_state.scores = default_scores.copy()

if "answers" not in st.session_state:
    st.session_state.answers = [None] * total_pages


# === 頁面 0：開始畫面 ===

# if st.session_state.page == 0:
#     st.markdown(
#         """
#         <div style='text-align: center;'>
#             <h1 style='color:#ffb6c1; font-size: 42px;'>
#                 ✨ 誰會成為你的連結者？ ✨
#             </h1>
#             <p style='font-size:18px; color:#888;'>L.M. Live 是法文 <b>“Lien Monde Live”</b> 的縮寫，<br>
#             譯「連結世界的直播」。</p>
#             <p style='font-size:16px; color:#555;'>
#                 快來測驗看看 L.M. Live 中，<br>
#                 誰最適合你吧！
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     if st.button("開始測驗 🚀"):
#         st.session_state.page += 1


if st.session_state.page == 0:
    st.title("💫 L.M. Live 守護精靈測驗")
    st.subheader("✨ 誰會成為你的連結者？")
    st.write("")
    st.write("L.M. Live 是法文 **Lien Monde Live** 的縮寫，譯「連結世界的直播」。")
    st.write("快來測驗看看 L.M. Live 中，誰最適合你吧！")
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("👇點擊下方按鈕開始你的測驗旅程")

    if st.button("開始測驗"):
        st.session_state.page += 1


# === 問題頁面 ===
elif 1 <= st.session_state.page <= len(questions):
    q_index = st.session_state.page - 1
    q_data = questions[q_index]

    st.header(f"第 {q_index + 1} / {len(questions)} 題")
    st.subheader(q_data["question"])

    # choice = st.radio(
    #     "請選擇一個答案：",
    #     list(q_data["options"].keys()),
    #     key=f"q_{q_index}"
    # )

    # # 儲存選擇
    # if choice:
    #     st.session_state.answers[q_index] = q_data["options"][choice]

    selected = st.radio("請選擇：", list(q_data["options"].keys()), index=None, key=f"q{st.session_state.page}")
    # 算分
    col1, col2 = st.columns(2)
    with col1:
        prev_clicked = st.button("上一題", disabled=(st.session_state.page == 0))
    with col2:
        next_clicked = st.button("下一題", disabled=(selected is None))

    st.write("page",st.session_state.page)
    # 處理按鈕事件
    if next_clicked and selected:
    # 撤銷上一個選項的分數（若有）
        prev_answer = st.session_state.answers[st.session_state.page]
        if prev_answer is not None:
            for attr, val in q_data["options"][prev_answer].items():
                st.session_state.scores[attr] -= val

    # 更新答案並加入新分數
        st.session_state.answers[st.session_state.page] = selected
        for attr, val in q_data["options"][selected].items():
            st.session_state.scores[attr] += val

    # 換頁
        st.session_state.page += 1

    elif prev_clicked:
        st.session_state.page -= 1

    # 頁面按鈕列
    # cols = st.columns([1, 2, 1])
    # with cols[0]:
    #     if st.button("⬅ 上一題", disabled=st.session_state.page == 1):
    #         st.session_state.page -= 1
    #         st.rerun()
    # with cols[2]:
    #     if st.button("下一題 ➡"):
    #         st.session_state.page += 1
    #         st.rerun()


# === 結果頁 ===
elif st.session_state.page == len(questions) + 1:
    st.title("🌟 結果頁 🌟")
    st.write("根據你的選擇，我們計算出以下屬性分數：")

    print("測驗分數", st.session_state.scores)

# -----------------------------------------------------------------------------------------


    best_cutee, best_row, best_score, recommand_sort = find_best_cutee(st.session_state.scores,features)
    # 轉為 DataFrame
    my_df = pd.DataFrame(dict(
        r = list(st.session_state.scores.values()) + [list(st.session_state.scores.values())[0]],  # 雷達圖需首尾相接
        theta = features + [features[0]]
    )) 

    print("best_row",best_row)
    print(best_row.values())
    print(features)

    best_row
    best_df = pd.DataFrame(dict(
        r = list(best_row.values()) + [list(best_row.values())[0]],  # 雷達圖需首尾相接
        theta = features + [features[0]]
    )) 

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=my_df['r'],
        theta=my_df['theta'],
        name='我的喜好',
        line=dict(shape='linear',color='red'),
        fill='none'
    ))

    fig.add_trace(go.Scatterpolar(
        r=best_df['r'],
        theta=best_df['theta'],
        name=best_cutee,
        line=dict(shape='linear'),
        fill='none'
    ))

    fig.update_layout(
    title=f'最適合你的人：{best_cutee}（適合度{round(best_score*100,1)}%）<br>其他推薦：{recommand_sort[1]["cutee_name"]}（適合度{round(recommand_sort[1]["suit_score"]*100,1)}%）或 {recommand_sort[2]["cutee_name"]}（適合度{round(recommand_sort[2]["suit_score"]*100,1)}%）',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        ),
        angularaxis=dict(
            rotation=90  # 這裡調整角度
        )
    ),
    showlegend=True
)
    st.plotly_chart(fig)

# -----------------------------------------------------------------------------------------

    # 重新計算分數
    # recalc_scores()

    # 顯示分數
    st.json(st.session_state.scores)

    total = sum(st.session_state.scores.values())
    st.markdown(f"### 總分：**{total}**")

    descriptions = {
        "可愛": "你散發出讓人想保護的魅力 💖",
        "漂亮": "你的外表令人驚艷 ✨",
        "有趣": "你是團體的開心果 😄",
        "氣質": "你給人一種沉靜優雅的感覺 🍃",
        "知性": "你的智慧讓人著迷 📘",
    }

    # st.markdown(f"**你的專屬連結者是：{top_attr}！**\n\n{descriptions[top_attr]}")

    st.markdown("---")

    if st.button("重新開始"):
        st.session_state.page = 0
        st.session_state.scores = default_scores.copy()
        st.session_state.answers = [None] * len(questions)


# ------------------------------------------------------------------------------------- 

# #若送出則處理邏輯與繪圖
# if submit_all:

    

#     image_filepath = "cutee_info"

#     if os.path.exists(os.path.join(image_filepath,f"{best_cutee}.webp")):
#         cutee_image_path = os.path.join(image_filepath,f"{best_cutee}.webp")
#     elif os.path.exists(os.path.join(image_filepath,f"{best_cutee}.png")):
#         cutee_image_path = os.path.join(image_filepath,f"{best_cutee}.png")
#     elif os.path.exists(os.path.join(image_filepath,f"{best_cutee}.gif")):
#         cutee_image_path = os.path.join(image_filepath,f"{best_cutee}.gif")
#     else:
#         cutee_image_path = None

#     if cutee_image_path is not None:
#         st.image(cutee_image_path, caption=f"{best_cutee}", use_container_width=True)

    
    

#     st.success("歡迎追隨，Enjoy your day!")