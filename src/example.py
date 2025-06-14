import os
import io
import time
import zipfile
import base64
import requests
import numpy as np
import PIL.Image
import streamlit as st
from openai import OpenAI
import cohere


# ------------ 配置API KEY ------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "你的api key")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的api key")

# 模型选用
EMBED_MODEL = "embed-v4.0"
QUERY_MODEL = "qwen3-235b-a22b"
VISION_MODEL = "qvq-max-2025-03-25"

#css样式
st.markdown(
    """
    <style>
    body {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 100%);
    color: #e6e6e6;
}

/* 容器玻璃化 */
.main .block-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 15px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
    backdrop-filter: blur(10px);
    padding: 2rem;
}

/* 侧边栏 */
.sidebar .sidebar-content {
    width: 100%;
    background: radial-gradient(circle at 20% 20%, #1c1f26 0%, #0f1118 90%);
    padding: 1.5rem;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* 卡片组件 */
.card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
    transition: all 0.3s ease-in-out;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 15px rgba(0, 255, 255, 0.05);
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0, 255, 255, 0.1);
}

/* 按钮样式 */
.stButton button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
}

/* 输入框样式 */
.stTextInput input {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0, 114, 255, 0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #00ffff;
    box-shadow: inset 0 0 5px rgba(0, 255, 255, 0.2);
}

.stTextInput input:focus {
    outline: none;
    border-color: #0072ff;
    box-shadow: 0 0 8px rgba(0, 114, 255, 0.5);
}

/* 图像预览增强 */
.image-preview {
    border: 2px dashed rgba(0, 114, 255, 0.3);
    border-radius: 10px;
    padding: 1rem;
    transition: all 0.3s ease;
}

.image-preview:hover {
    border-color: #00ffff;
    background-color: rgba(0, 114, 255, 0.05);
}

/* 自定义加载动画 */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #00ffff;
    border-radius: 50%;
    border-top-color: transparent;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* 响应式网格 */
@media (min-width: 768px) {
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1.5rem;
    }
}
</style>
    """ , unsafe_allow_html=True)



# 初始化客户端
def init_clients():
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    qwen_client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return co, qwen_client

co, qwen_client = init_clients()

MAX_PIXELS = 1568 * 1568
SIM_THRESHOLD = 0.3  # 相似度阈值，可根据需要调整
# ------------一些功能性函数 ------------
def resize_image(pil_image: PIL.Image.Image):
    w, h = pil_image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        pil_image.thumbnail((int(w * scale), int(h * scale)))


def base64_from_pil(pil_image: PIL.Image.Image) -> str:
    fmt = pil_image.format or "PNG"
    resize_image(pil_image)
    with io.BytesIO() as buf:
        pil_image.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{data}"



def embed_document(img: PIL.Image.Image):
    b64 = base64_from_pil(img)
    doc_in = {"content": [{"type": "image", "image": b64}]}
    resp = co.embed(model=EMBED_MODEL,
                    input_type="search_document",
                    embedding_types=["float"],
                    inputs=[doc_in])
    vec = np.array(resp.embeddings.float[0])
    return vec / np.linalg.norm(vec)


def embed_query(text: str):
    resp = co.embed(model=EMBED_MODEL,
                    input_type="search_query",
                    embedding_types=["float"],
                    texts=[text])
    vec = np.array(resp.embeddings.float[0])
    return vec / np.linalg.norm(vec)




def generate_answer(question: str, img: PIL.Image.Image = None):
    """支持流式输出"""
    if img is not None:
        # 处理图文输入（使用流式模式）
        b64_image = base64_from_pil(img)

        messages = [
            {"role": "system", "content": "根据下面的图片回答问题。"},
             {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]
        }]
        response = qwen_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            stream=True,  # 启用流式输出

        )
    else:
        # 处理纯文本输入
        messages = [
            {"role": "system", "content": "请仅根据文字问题回答"},
            {"role": "user", "content": question}
        ]
        response = qwen_client.chat.completions.create(
            model=QUERY_MODEL,
            messages=messages,
            stream=True,  # 启用流式输出

        )


    def stream_response():
        try:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            st.error(f"流式输出错误: {str(e)}")
            yield "抱歉，流式输出过程中发生错误。"

    return stream_response()

# ------------ 会话状态 ------------
if 'img_paths' not in st.session_state:
    st.session_state.img_paths = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = []


st.sidebar.title("导航栏")
page = st.sidebar.radio("功能", ["查询", "上传", "图库"], index=0)

# ------------ 上传页面 ------------
if page == "上传":
    st.header("上传图片资源")
    uploaded = st.file_uploader("上传单张图片", type=['png','jpg','jpeg'])
    url = st.text_input("或输入图片 URL")
    zipf = st.file_uploader("上传 ZIP 压缩包 (仅图片)", type=['zip'])

    if uploaded:
        img = PIL.Image.open(uploaded)
        path = f"uploaded_{int(time.time())}.png"
        img.save(path)
        st.session_state.img_paths.append(path)
        st.session_state.doc_embeddings.append(embed_document(img))
        st.success(f"已添加: {path}")
    if url:
        try:
            r = requests.get(url)
            r.raise_for_status()
            img = PIL.Image.open(io.BytesIO(r.content))
            path = f"url_{int(time.time())}.png"
            img.save(path)
            st.session_state.img_paths.append(path)
            st.session_state.doc_embeddings.append(embed_document(img))
            st.success(f"已下载并添加成功: {path}")
        except Exception as e:
            st.error(f"URL 下载失败👀: {e}")
    if zipf:
        with zipfile.ZipFile(zipf) as z:
            for fname in z.namelist():
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    data = z.read(fname)
                    img = PIL.Image.open(io.BytesIO(data))
                    path = f"zip_{int(time.time())}_{os.path.basename(fname)}"
                    img.save(path)
                    st.session_state.img_paths.append(path)
                    st.session_state.doc_embeddings.append(embed_document(img))
            st.success("ZIP 中的所有图片已添加")

# ------------ 查询页面 ------------
elif page == "查询":
    st.header("🔍 图像RAG-文字检索图片")
    question = st.text_input("请输入您的问题✍️：")
    if st.button("检索并回答"):
        if not question:
            st.warning("请输入问题后再检索🤓👆")
        else:
            q_emb = embed_query(question)
            if st.session_state.doc_embeddings:
                docs = np.vstack(st.session_state.doc_embeddings)
                sims = docs.dot(q_emb)
                max_sim = float(np.max(sims))
                idx = int(np.argmax(sims))
            else:
                max_sim = 0
            answer_container = st.empty()
            full_answer = ""
            # 判断是否命中
            if max_sim < SIM_THRESHOLD:
            # 文字回答流式渲染
              ans_generator = generate_answer(question)
              for token in ans_generator:
                full_answer += token
                answer_container.markdown(f"**模型文字回答（无相关图片）：** {full_answer}")
            else:

            # 显示图片
              best = st.session_state.img_paths[idx]
              img = PIL.Image.open(best)
              st.image(img, caption=f"最相关图片（相关度：{max_sim:.2f}）", use_column_width=True)

            # 图文回答流式渲染
              ans_generator = generate_answer(question, img)
              for token in ans_generator:
                full_answer += token
                answer_container.markdown(f"**模型回答：** {full_answer}")

# ------------ 图库页面 ------------
elif page == "图库":
    st.header("📸 已录入图片图库")
    paths = st.session_state.img_paths
    if not paths:
        st.info("""
啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊！
宝宝肚肚打雷啦😡。
""")
        st.info("""一张图片都没有
**❗️请先在“上传”页面添加图片❗️**""")
        st.info("""啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊！""")
    else:
        for i in range(0, len(paths), 3):
            cols = st.columns(3)
            for j, path in enumerate(paths[i:i+3]):
                with cols[j]:
                    img = PIL.Image.open(path)
                    st.image(img, width=150, caption=os.path.basename(path))