import gradio as gr
from src.nodes.preprocess import pre_processing_interview
from src.nodes.graph import build_graph

graph = build_graph()

def initialize_session():
    return {"state": None, "started": False, "chat": []}

def start_interview(file_obj, sess):
    if not file_obj:
        return sess, [["AI 면접관", "이력서를 먼저 업로드해주세요."]]

    state = pre_processing_interview(file_obj.name)
    sess["state"] = state
    sess["started"] = True
    sess["chat"] = [["AI 면접관", state["current_question"]]]
    return sess, sess["chat"]

def respond(message, sess):
    if not sess.get("started"):
        sess["chat"].append(["AI 면접관", "먼저 이력서를 업로드하고 시작하세요."])
        return sess, sess["chat"], ""

    sess["chat"].append(["지원자", message])
    st = sess["state"]
    st["current_answer"] = message  # ✅ 사용자 답변을 state에 넣고

    st = graph.invoke(st, config={"recursion_limit": 50})  # ✅ 한 턴 처리

    sess["state"] = st

    if st.get("next_step") == "end":
        sess["chat"].append(["AI 면접관", st.get("final_report", "종료되었습니다.")])
        return sess, sess["chat"], ""

    sess["chat"].append(["AI 면접관", st.get("current_question", "다음 질문을 준비 중입니다.")])
    return sess, sess["chat"], ""

def build_demo():
    with gr.Blocks(theme="soft", title="AI 면접관") as demo:
        sess = gr.State(initialize_session())

        file_input = gr.File(label="이력서 업로드 (PDF/DOCX)", file_types=[".pdf", ".docx"], type="filepath")
        start_btn = gr.Button("인터뷰 시작", variant="primary")

        chatbot = gr.Chatbot(height=520)
        user_input = gr.Textbox(placeholder="답변을 입력하고 Enter...")

        start_btn.click(start_interview, inputs=[file_input, sess], outputs=[sess, chatbot])
        user_input.submit(respond, inputs=[user_input, sess], outputs=[sess, chatbot, user_input])

    return demo
