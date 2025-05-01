# graph.py
from typing import TypedDict, Optional
from exercises import _PROMPT_EXERCISE, _PROMPT_REVIEW
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# -------------------- IO Schemas --------------------
class ExerciseSchema(BaseModel):
    question: str = Field(...)
    answer_key: str = Field(...)


class ReviewSchema(BaseModel):
    score: int
    better: Optional[str] = None
    comment: str = Field(...)


class InputDict(TypedDict):
    jp_prompt: str
    en_answer: str
    question: str
    answer_key: str
    user_input: str


class OutputDict(InputDict):
    feedback: str


# -------------------- LLM Models --------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_ex = llm.with_structured_output(ExerciseSchema)
llm_fb = llm.with_structured_output(ReviewSchema)


# -------------------- Graph Nodes --------------------
def generate_exercise(state: InputDict) -> InputDict:
    template_ex = PromptTemplate.from_template(_PROMPT_EXERCISE)
    result = (template_ex | llm_ex).invoke(
        {
            "jp": state["jp_prompt"],
            "en_chunk": state["en_answer"],
        }
    )
    return {
        **state,
        "question": result.question,
        "answer_key": result.answer_key,
    }


def review_output(state: InputDict) -> OutputDict:
    template_fb = PromptTemplate.from_template(_PROMPT_REVIEW)
    result = (template_fb | llm_fb).invoke(
        {
            "en_chunk": state["en_answer"],
            "user_answer": state["user_input"],
            "model_answer": state["answer_key"],
        }
    )

    comment_md = f"**Score:** {result.score}/5\n\n"
    if result.better:
        comment_md += f"**Better:** {result.better}\n\n"
    comment_md += result.comment

    return {
        **state,
        "feedback": comment_md,
    }


# -------------------- Graph Assembly --------------------
graph = StateGraph(input=InputDict, output=OutputDict)
graph.add_node("generate_exercise", generate_exercise)
graph.add_node("review_output", review_output)
graph.set_entry_point("generate_exercise")
graph.add_edge("generate_exercise", "review_output")
graph.set_finish_point("review_output")

app = graph.compile()
