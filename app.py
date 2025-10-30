
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# app_llm_expert_switch.py
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# 設定
# =========================
DEFAULT_MODEL = "gpt-4o-mini"  # 必要に応じて変更可

# LLM本体（温度などはUIから変更できるようにしてもOK）
def get_llm(model_name: str, temperature: float):
    return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=900)

# 選択肢ごとのシステムメッセージ
SYSTEM_PROMPTS = {
    "心理": (
        "あなたは心理療法の領域における有資格の専門家としてふるまいます。"
        "安全・倫理・多様性の尊重を最優先とし、診断名の断定や医療行為は行いません。"
        "面接の基本（受容・共感・明確化・要約）と、CBT/MI/危機介入/ストレス対処/睡眠衛生などの"
        "エビデンスに基づく一般的助言を、相談者の語りを尊重しながら段階的に提示してください。"
        "自傷他害や虐待が疑われる場合は、緊急性評価と専門窓口/医療機関への受診を明確に促してください。"
        "専門用語は平易に言い換え、宿題やセルフモニタリング案は具体的に。"
        "最後に『次の一歩』として小さな行動目標を1つ提案してください。"
    ),
    "薬": (
        "あなたは精神科領域の薬物療法に詳しい専門家としてふるまいます。"
        "ただし診断・処方は行わず、一般的情報の提供と受診推奨に徹します。"
        "作用機序・一般的な有効性・代表的副作用・相互作用・服薬アドヒアランス・中止リスク・妊娠授乳への配慮"
        "などを、ガイドライン整合的かつ平易に説明してください。"
        "具体薬の個別用量や指示は出さず、不安が強い場合は主治医/薬剤師への確認を促してください。"
        "安全第一で、急変・重篤副作用が疑われる場合は救急受診の基準を明示してください。"
        "最後に『受診時に主治医へ確認すると良い質問』を3つ提案してください。"
    ),
}

# =========================
# LLM実行関数
# =========================
def run_llm(input_text: str, selected_role: str, model_name: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    """
    入力テキストとラジオボタンの選択値を受け取り、LLMの回答（str）を返す。
    - input_text: ユーザーからの相談/質問
    - selected_role: "心理" または "薬"
    """
    system_msg = SYSTEM_PROMPTS.get(selected_role, "You are a helpful assistant.")

    llm = get_llm(model_name=model_name, temperature=temperature)
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=input_text),
    ]
    result = llm.invoke(messages)
    return result.content

# =========================
# UI
# =========================
st.title("サンプルアプリ③: 専門家切替 LLMアシスタント")
st.write("このWebアプリでは、**ラジオボタンで専門家の種類（心理／薬）**を選び、入力した相談内容に対して、選択した専門家の視点でLLMが応答します。")
st.write("- 「心理」: 心理療法の枠組みで、支持的・段階的な助言と小さな行動目標の提案")
st.write("- 「薬」: 薬物療法の一般的情報（作用機序・副作用など）と、受診時に確認すると良い質問の提案")
st.info("※ 本アプリは一般的情報の提供を目的とし、診断・処方は行いません。緊急時は医療機関へ。", icon="⚠️")

st.divider()

# 操作説明
with st.expander("操作方法"):
    st.markdown(
        """
1. **専門家の種類**を選択（「心理」または「薬」）。
2. **相談内容**を入力（自由記述）。
3. **オプション**でモデル名や温度を調整（必要に応じて）。
4. **「実行」**を押すと、LLMの回答が表示されます。
        """
    )

# 選択UI
selected_item = st.radio(
    "専門家の種類を選んでください。",
    ["心理", "薬"],
    horizontal=True,
)

# 入力
input_text = st.text_area(
    "相談内容（自由記述）",
    placeholder="例）最近眠れず、日中の不安が強いです。仕事の集中力も落ちています…",
    height=160,
)

# オプション（モデル/温度）
with st.expander("詳細設定（任意）"):
    model_name = st.text_input("モデル名", value=DEFAULT_MODEL, help="例: gpt-4o-mini / gpt-4o / o4-mini など")
    temperature = st.slider("温度（創造性）", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
else:
    # 旧Streamlitとの互換用（expander外で参照可能に）
    model_name = locals().get("model_name", DEFAULT_MODEL)
    temperature = locals().get("temperature", 0.2)

st.divider()

# 実行ボタン
if st.button("実行"):
    st.divider()
    if not input_text.strip():
        st.error("相談内容を入力してください。")
    else:
        try:
            with st.spinner("LLMが回答を作成中…"):
                answer = run_llm(
                    input_text=input_text.strip(),
                    selected_role=selected_item,
                    model_name=model_name,
                    temperature=temperature,
                )
            st.subheader("回答")
            st.write(answer)

            # 参考として、実際に使われたシステムメッセージも確認可能に
            with st.expander("（参考）この回答で使われたシステムメッセージ"):
                st.code(SYSTEM_PROMPTS[selected_item], language="markdown")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.info("APIキー（OPENAI_API_KEY）が設定されているか、モデル名の綴りやレート制限をご確認ください。")
