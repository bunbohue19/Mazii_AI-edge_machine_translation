import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime


class TranslationClient:
    """Client for communicating with the translation server"""
    
    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url
    
    def translate(self, text: str, target_lang: str, 
                temperature: float = 0.20, max_tokens: int = 1024) -> Dict[str, Any]:
        """Send translation request"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/translate",
                json={
                    "text": text,
                    "target_language_code": target_lang,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def initialize_session_state():
    """Initialize session state variables"""
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []


def main_translation_interface():
    """Main translation interface"""
    st.title("🌐 Mazii Machine Translation (MMT)")
    st.markdown("*Post-trained by using Supervised Fine-Tuning (SFT)")
    
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # Language selection
        source_lang = st.selectbox(
            "From:",
            ["Auto"],
            key="source_lang"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            help="Lower values make translations more deterministic, higher values add creativity."
        )
        
        # Input text
        input_text = st.text_area(
            "Text to translate:",
            height=200,
            placeholder="Enter text here...",
            key="input_text"
        )
    
    with col2:
        st.subheader("Output")
        
        target_lang = st.selectbox(
            "To:",
            ["Japanese", "Vietnamese", "Traditional Chinese", "Indonesian"],
            index=0,
            key="target_lang"
        )
        
        # Translation button
        if st.button("Translate", type="primary", use_container_width=True):
            if not input_text.strip():
                st.warning("Please enter text to translate!")
            else:
                with st.spinner("Translating..."):
                    client = TranslationClient()
                    start_time = time.time()
                    
                    result = client.translate(
                        input_text.strip(),
                        target_lang,
                        temperature
                    )
                    
                    end_time = time.time()
                
                if "error" in result:
                    st.error(f"Translation failed: {result['error']}")
                else:
                    # Display translation result
                    st.text_area(
                        "Translation:",
                        value=result.get("translated_text", ""),
                        height=200,
                        key="output_text"
                    )
                    
                    # Show metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("⏱️ Time", f"{end_time - start_time:.2f}s")
                    with col_b:
                        st.metric("📊 Tokens", result.get("usage", {}).get("total_tokens", 0))
                    
                    # Add to history
                    st.session_state.translation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "target_lang": target_lang,
                        "original": input_text.strip(),
                        "translated": result.get("translated_text", ""),
                        "time": f"{end_time - start_time:.2f}s",
                        "tokens": result.get("usage", {}).get("total_tokens", 0)
                    })


def translation_history():
    """Show translation history"""
    st.header("📈 Translation History")
    
    if not st.session_state.translation_history:
        st.info("No translations yet. Try translating some text!")
        return
    
    # Summary metrics
    total_translations = len(st.session_state.translation_history)
    total_tokens = sum(t.get("tokens", 0) for t in st.session_state.translation_history)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Translations", total_translations)
    with col2:
        st.metric("Total Tokens", total_tokens)
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.rerun()
    
    st.subheader("Recent Translations")
    
    # Show recent translations (last 10)
    for i, trans in enumerate(reversed(st.session_state.translation_history[-10:])):
        with st.expander(f"🕐 {trans['timestamp']} → {trans['target_lang']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text:**")
                st.text(trans['original'])
            
            with col2:
                st.markdown("**Translation:**")
                st.text(trans['translated'])
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.text(f"⏱️ {trans['time']}")
            with col_b:
                st.text(f"🔢 {trans['tokens']} tokens")
            with col_c:
                st.text(f"🌐 → {trans['target_lang']}")


def main():
    """Main application"""
    st.set_page_config(
        page_title="Mazii Machine Translation",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🌐 Translate", "📈 History"])
    
    with tab1:
        main_translation_interface()
    
    with tab2:
        translation_history()


if __name__ == "__main__":
    main()