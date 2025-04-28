# exercises.py
"""
Exercise/feedback generator for ChunkRecallTrainer (LangChain + OpenAI).

Usage:
    from exercises import ExerciseGenerator

    gen = ExerciseGenerator(api_key=session_state["api_key"])
    ex  = gen.create_exercise(jp="お知らせします", en_chunk="I'll keep you posted")

    # ex.question   -> 英語で出題された問題文
    # ex.answer_key -> 模範解答
    # ex.model_json -> 元の LLM 出力全文

    fb = gen.review_answer(
        user_answer="I'll keep you posted about tomorrow's schedule.",
        ex=ex,
        jp="お知らせします",
        en_chunk="I'll keep you posted"
    )
    # fb.score, fb.comment_md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ────────────────────── Data containers ──────────────────────
@dataclass
class Exercise:
    question: str
    answer_key: str
    model_json: Dict[str, Any]


@dataclass
class Feedback:
    score: int
    comment_md: str
    raw: str


# ────────────────────── Prompt templates ──────────────────────

_PROMPT_EXERCISE = """You are an encouraging English coach.

Japanese phrase (JP): {jp}
Target chunk (EN): {en_chunk}
---
Create one short practice question **in ENGLISH** that requires the learner to use
the target chunk exactly once in a natural, everyday context.

Return JSON with keys:
- "question": the English prompt for the learner
- "answer": one model answer that correctly uses the chunk
"""


class ExSchema(BaseModel):
    question: str = Field(..., description="English prompt for the learner")
    answer: str = Field(..., description="One model answer using the chunk")


_PROMPT_REVIEW = """You are a strict but kind English proof-reader.

Target chunk: {en_chunk}
Learner answer: {user_answer}
Model answer: {model_answer}
---
(1) Score the learner's answer from 0-5.
(2) Suggest a more natural alternative if needed.
(3) Explain the main improvement points in Japanese.

Return in markdown:
1. `Score: <n>/5`
2. `Better: <suggestion>` (single line, EN, omit if perfect)
3. Bulleted JP feedback
"""


class FBSchema(BaseModel):
    score: int = Field(..., description="Score 0-5")
    better: str | None = Field(None, description="Better alternative if any")
    comment: str = Field(..., description="JP feedback")


# ──────────────────────────── Core ────────────────────────────
class ExerciseGenerator:
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7
    ):
        llm_raw = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
        self.llm_ex = llm_raw.with_structured_output(ExSchema)
        self.template_ex = PromptTemplate.from_template(_PROMPT_EXERCISE)

        self.llm_fb = llm_raw.with_structured_output(FBSchema)
        self.template_fb = PromptTemplate.from_template(_PROMPT_REVIEW)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def create_exercise(self, jp: str, en_chunk: str) -> Exercise:
        data = (self.template_ex | self.llm_ex).invoke({"jp": jp, "en_chunk": en_chunk})
        # data は ExSchema → dict と同等
        return Exercise(
            question=data.question,
            answer_key=data.answer,
            model_json=data.model_dump(),
        )

    def review_answer(self, user_answer: str, ex: Exercise, en_chunk: str) -> Feedback:
        fb_data = (self.template_fb | self.llm_fb).invoke(
            {
                "en_chunk": en_chunk,
                "user_answer": user_answer,
                "model_answer": ex.answer_key,
            }
        )
        comment_md = (
            f"**Score:** {fb_data.score}/5\n\n"
            + (f"**Better:** {fb_data.better}\n\n" if fb_data.better else "")
            + fb_data.comment
        )
        return Feedback(
            score=fb_data.score,
            comment_md=comment_md,
            raw=fb_data.model_dump_json(indent=2),
        )
