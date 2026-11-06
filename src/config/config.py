import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class Config:

    GROQ_API_KEY=os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

    TAVILY_API_KEY=os.getenv("TAVILY_API_KEY") or st.secrets["TAVILY_API_KEY"]

    # Groq retired the Llama 3.x models on 16 Aug 2026. These are Groq's
    # recommended replacements. Override them in .env (LLM_MODEL=...,
    # EVALUATOR_MODEL=...) without touching code when Groq changes models again.
    LLM_MODEL=os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    EVALUATOR_MODEL=os.getenv("EVALUATOR_MODEL", "openai/gpt-oss-20b")

    # Used automatically when the main model fails, e.g. when it hits Groq's
    # daily token limit. Each Groq model has its own daily budget, so this
    # roughly doubles how many questions the app can answer per day.
    # Qwen 3.8 27B is a Groq *preview* model: if Groq removes it, set
    # FALLBACK_MODEL to another model, or to "none" to turn the backup off.
    FALLBACK_MODEL=os.getenv("FALLBACK_MODEL", "qwen/qwen3.8-27b")

    EMBEDDING_MODEL="all-MiniLM-L6-v2"

    CHUNK_SIZE=1200

    CHUNK_OVERLAP=200

    @classmethod
    def validate(cls):

        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing")

        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY missing")

    @classmethod
    def _chat(cls, model, retries=2):
        extra = {}
        if model.startswith("qwen/"):
            # Qwen thinks out loud in <think> tags by default; keep only the answer.
            extra["reasoning_format"] = "hidden"
        return ChatGroq(
            model=model,
            api_key=cls.GROQ_API_KEY,
            temperature=0,
            max_retries=retries,
            **extra
        )

    @classmethod
    def get_llm(cls):

        cls.validate()

        os.environ["TAVILY_API_KEY"]=cls.TAVILY_API_KEY

        fallback = cls.FALLBACK_MODEL
        if not fallback or fallback.lower() == "none" or fallback == cls.LLM_MODEL:
            return cls._chat(cls.LLM_MODEL)
        # Few retries on the main model so a daily-limit error switches to the
        # backup quickly instead of waiting.
        return cls._chat(cls.LLM_MODEL, retries=1).with_fallbacks([cls._chat(fallback)])

    @classmethod
    def get_evaluator_llm(cls):

        cls.validate()

        return cls._chat(cls.EVALUATOR_MODEL)