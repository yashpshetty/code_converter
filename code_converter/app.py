import gradio as gr
from openai import OpenAI
import os

# ── Config ────────────────────────────────────────────────
# On HuggingFace Spaces, set HF_API_TOKEN in Settings → Variables and secrets
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

MODEL    = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
BASE_URL = "https://router.huggingface.co/v1"

LANGUAGES = ["Python", "JavaScript", "Java", "C++", "C"]

# ── LLM Client ────────────────────────────────────────────
def get_client():
    return OpenAI(base_url=BASE_URL, api_key=HF_API_TOKEN)

# ── Step 1: Validate input language ───────────────────────
def validate_code(source_code: str, source_lang: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason)
    reason: "VALID" | "NOT_CODE" | "WRONG_LANGUAGE:<detected>"
    """
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict code language detector. "
                        "You will receive a piece of text and a programming language name. "
                        "Determine:\n"
                        "1. Is the input actual programming code (not plain English, random words)?\n"
                        "2. If it is code, is it written in the specified language?\n"
                        "Reply with ONLY one of these (no extra text):\n"
                        "VALID\n"
                        "NOT_CODE\n"
                        "WRONG_LANGUAGE:<detected_language>"
                    )
                },
                {
                    "role": "user",
                    "content": f"Language specified: {source_lang}\n\nInput:\n{source_code}"
                }
            ],
            max_tokens=20,
            temperature=0.0
        )
        verdict = response.choices[0].message.content.strip()

        if verdict == "VALID":
            return True, "VALID"
        elif verdict == "NOT_CODE":
            return False, "NOT_CODE"
        elif verdict.startswith("WRONG_LANGUAGE"):
            parts = verdict.split(":")
            detected = parts[1].strip() if len(parts) > 1 else "Unknown"
            return False, f"WRONG_LANGUAGE:{detected}"
        else:
            return True, "VALID"  # fallback: allow
    except Exception:
        return True, "VALID"      # if validation errors, allow conversion

# ── Step 2: Convert code ──────────────────────────────────
def convert_code(source_code: str, source_lang: str, target_lang: str) -> str:
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert programmer. Convert the given {source_lang} code to {target_lang}. "
                        "Return ONLY the converted code. No explanation, no markdown, no code fences."
                    )
                },
                {
                    "role": "user",
                    "content": f"Convert this {source_lang} code to {target_lang}:\n\n{source_code}"
                }
            ],
            max_tokens=1024,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# ── Main handler ──────────────────────────────────────────
def handle_conversion(source_code: str, source_lang: str, target_lang: str):
    # Basic checks
    if not HF_API_TOKEN:
        return (
            "❌ HF_API_TOKEN not set.\n"
            "Go to Settings → Variables and secrets on HuggingFace Spaces and add your token."
        )
    if not source_code.strip():
        return "⚠️ Please paste your source code first."
    if source_lang == target_lang:
        return "⚠️ Source and Target languages must be different."

    # Validate
    is_valid, reason = validate_code(source_code, source_lang)

    if not is_valid:
        if reason == "NOT_CODE":
            return (
                f"❌ Invalid Input — Not a Programming Code\n\n"
                f"The text you entered does not appear to be any programming code.\n"
                f"Please paste valid {source_lang} source code and try again."
            )
        elif reason.startswith("WRONG_LANGUAGE"):
            detected = reason.split(":")[1] if ":" in reason else "a different language"
            return (
                f"⚠️ Wrong Source Language Selected\n\n"
                f"The code you entered appears to be {detected}, "
                f"but you selected {source_lang} as the source language.\n\n"
                f"Please either:\n"
                f"  • Paste {source_lang} code, OR\n"
                f"  • Change the source language to {detected}"
            )

    # Convert
    result = convert_code(source_code, source_lang, target_lang)

    if result.startswith("ERROR"):
        return f"❌ {result}"

    return result

# ── Gradio UI ─────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

body, .gradio-container {
    background: #0d0d14 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
}

/* Header */
#header {
    text-align: center;
    padding: 36px 0 20px;
}

#header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #a78bfa, #60e8d4, #f9a87a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin: 0 !important;
}

#header p {
    color: #5a5a72 !important;
    font-size: 0.85rem !important;
    margin-top: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Panels */
.panel-box {
    background: #13131f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 14px !important;
    padding: 0 !important;
}

/* Labels */
label span {
    color: #a78bfa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Dropdowns */
.gr-dropdown select, select {
    background: #1a1a2e !important;
    color: #e0e0f0 !important;
    border: 1px solid #3a3a5c !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}

/* Textareas */
textarea {
    background: #0f0f1a !important;
    color: #c8c8e0 !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}

textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.15) !important;
    outline: none !important;
}

/* Convert button */
#convert-btn {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 14px 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
}

#convert-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.55) !important;
}

#convert-btn:active {
    transform: translateY(0) !important;
}

/* Output label color */
#output-box label span {
    color: #60e8d4 !important;
}

/* Footer */
#footer {
    text-align: center;
    color: #3a3a52;
    font-size: 0.75rem;
    padding: 16px 0 24px;
}
"""

with gr.Blocks(css=css) as demo:

    gr.HTML("""
    <div id="header">
        <h1>⟨ CODE CONVERTER ⟩</h1>
        <p>AI-powered · HuggingFace Router · Llama-3.1-8B · 5 Languages</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            source_lang = gr.Dropdown(
                choices=LANGUAGES,
                value="Python",
                label="📥  Source Language",
            )
        with gr.Column(scale=1):
            target_lang = gr.Dropdown(
                choices=LANGUAGES,
                value="JavaScript",
                label="📤  Target Language",
            )

    source_code = gr.Code(
        label="📝  Paste Your Source Code",
        language="python",
        lines=16,
    )

    convert_btn = gr.Button(
        "▶   Convert Code",
        elem_id="convert-btn",
        variant="primary",
    )

    output_code = gr.Code(
        label="✅  Converted Code",
        lines=16,
        interactive=False,
        elem_id="output-box",
    )

    gr.HTML('<div id="footer">Built with HuggingFace · Gradio · Llama 3.1</div>')

    # Dynamic language highlight for source editor
    def update_source_lang(lang):
        lang_map = {
            "Python": "python",
            "JavaScript": "javascript",
            "Java": "java",
            "C++": "cpp",
            "C": "c",
        }
        return gr.Code(language=lang_map.get(lang, "python"))

    def update_target_lang(lang):
        lang_map = {
            "Python": "python",
            "JavaScript": "javascript",
            "Java": "java",
            "C++": "cpp",
            "C": "c",
        }
        return gr.Code(language=lang_map.get(lang, "python"))

    source_lang.change(fn=update_source_lang, inputs=source_lang, outputs=source_code)
    target_lang.change(fn=update_target_lang, inputs=target_lang, outputs=output_code)

    convert_btn.click(
        fn=handle_conversion,
        inputs=[source_code, source_lang, target_lang],
        outputs=output_code,
    )

if __name__ == "__main__":
    demo.launch()
