"""VAKSHA Streamlit interface with Firestore logging support."""
import uuid
import streamlit as st

from core.firebase_store import fetch_recent_runs, save_run_log
from core.pipeline import run_baseline, run_protected

st.set_page_config(page_title="VAKSHA | Multilingual AI Safety", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background: #0b1220; color: #edf2ff;}
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.8rem; border: 1px solid #263653; border-radius: 18px;
background: linear-gradient(120deg,#14233d,#101a2e); margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.7rem; margin: 0; color: #fff; letter-spacing: .08em;}
.hero p {color:#a9bbd7; margin:.35rem 0 0;}
.notice {padding: .8rem 1rem; border-radius: 10px; background:#352b16; color:#ffe3a3;
border:1px solid #725923; margin: .8rem 0 1rem;}
[data-testid="stMetricValue"] {color:#fff;}
</style>
<div class="hero"><h1>🛡️ VAKSHA</h1><p>Multilingual AI Safety &amp; Protection Pipeline</p></div>
<div class="notice"><b>DEMO ADAPTER MODE:</b> Uses synthetic mock adapters until live translation, safety-classifier, and target AI adapters are registered.</div>
""", unsafe_allow_html=True)


def make_translation(text: str):
    return {"translated_text": f"[Demo translation] {text}", "model": "demo-translator (mock)"}


def make_classifier(label: str):
    def classify(text: str):
        return {
            "label": label,
            "category": "synthetic_demo_case" if label == "unsafe" else None,
            "model": "demo-classifier (mock)",
        }
    return classify


def demo_target(prompt: str):
    return {
        "response": f"[Mock target response] The demo target received your original prompt: {prompt}",
        "model": "demo-target-llm (mock)",
    }


left, right = st.columns([1.25, .75], gap="large")
with left:
    st.subheader("Analyze a prompt")
    language = st.selectbox("Input language", ["Kannada", "Kanglish / Romanized Kannada", "English", "Other / mixed"])
    prompt = st.text_area("Your prompt", height=150, placeholder="Type or paste a Kannada, Kanglish, or English prompt…")
    mode = st.radio("Execution route", ["Protected pipeline", "Baseline (unprotected)"], horizontal=True)
    demo_label = st.selectbox(
        "Demo classifier outcome",
        ["safe", "unsafe", "unknown"],
        help="Simulation control only. This is not an actual model prediction.",
    )
    persist_to_db = st.checkbox("Log run to Firebase Firestore", value=True)
    run = st.button("▶  Analyze prompt", type="primary", use_container_width=True)

with right:
    st.subheader("What VAKSHA does")
    st.markdown("**Protected route**")
    st.write("1. Translate the input to English for safety inspection.\n2. Classify the translated prompt.\n3. ALLOW, BLOCK, or require human review.\n4. Call the target model only when allowed.")
    st.markdown("**Baseline route**")
    st.write("Sends the original prompt directly to the target model, without the protection checks.")
    st.caption("The target model receives the original prompt in both routes when it is called.")

if run:
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        with st.spinner("Running VAKSHA pipeline…"):
            if mode == "Protected pipeline":
                result = run_protected(
                    prompt,
                    prompt_id=run_id,
                    translation_adapter=make_translation,
                    classifier_adapter=make_classifier(demo_label),
                    target_adapter=demo_target,
                )
            else:
                result = run_baseline(
                    prompt,
                    prompt_id=run_id,
                    target_adapter=demo_target,
                )

        enriched_result = {**result, "language": language}

        if persist_to_db:
            ok, db_err = save_run_log(enriched_result)
            if ok:
                st.toast("Saved run to Firestore", icon="🔥")
            elif db_err:
                st.caption(f"Firestore log skipped: {db_err}")

        st.divider()
        st.subheader("Pipeline result")
        decision = result.get("decision", "UNKNOWN")
        color = {"ALLOW": "#1b9b69", "BLOCK": "#c2414b", "REVIEW_REQUIRED": "#b7791f"}.get(decision, "#64748b")
        a, b, c, d = st.columns(4)
        a.metric("Decision", decision)
        b.metric("Status", result.get("status", "—"))
        c.metric("Target model called", "Yes" if result.get("target_called") else "No")
        d.metric("Latency", f"{result.get('latency_ms', 0):.2f} ms")
        st.markdown(f"<div style='height:6px;border-radius:6px;background:{color};margin:.2rem 0 1rem'></div>", unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["Safety & translation", "Model response", "Raw pipeline data", "Firestore history"])
        with t1:
            trans = result.get("translation") or {}
            guard = result.get("guard") or {}
            x, y = st.columns(2)
            with x:
                st.markdown("**English translation**")
                st.write(trans.get("translated_text", "Not performed on baseline route"))
                st.caption(f"Translator: {trans.get('model', '—')}")
            with y:
                st.markdown("**Safety classification**")
                st.write(guard.get("label", "Not performed on baseline route"))
                st.write("Category:", guard.get("category") or "None reported")
                st.caption(f"Classifier: {guard.get('model', '—')}")
            st.info("A demo outcome is manually selected above; it is not a real safety assessment.")
        with t2:
            if result.get("target_called"):
                st.success("Target adapter was invoked.")
                st.write(result.get("target_response"))
                st.caption(f"Target: {result.get('target_model') or '—'}")
            else:
                st.warning("Target adapter was not called.")
                st.write("Decision:", decision)
        with t3:
            st.json(enriched_result)
        with t4:
            history, hist_err = fetch_recent_runs(limit=15)
            if hist_err:
                st.info(f"Firestore history unavailable: {hist_err}")
            elif not history:
                st.write("No runs logged yet.")
            else:
                st.dataframe(history, use_container_width=True)

        if result.get("errors"):
            st.error("Pipeline notes: " + "; ".join(result["errors"]))

st.divider()
st.caption("VAKSHA · Team GLITCH · Streamlit + Firebase Firestore")