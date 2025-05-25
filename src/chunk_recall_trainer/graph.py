"""
LangGraph definition for the Chunk Recall Trainer.

This module defines a stateful graph using LangGraph to manage the flow of
exercise generation and user answer review. The graph can:
1. Generate a new exercise question and its answer key.
2. Optionally, review a user's answer to a previously generated question and provide feedback.

The flow is conditional: if user input is provided, the graph proceeds to review;
otherwise, it stops after exercise generation.
"""
# graph.py
from typing import TypedDict, Optional, Any # Any is used for the graph state values, though more specific types are preferred if possible
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# PydanticOutputParser is implicitly used by .with_structured_output()
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from .config import app_config
from .logger import logger # Import the logger

# ────────────────────── Prompt Templates ──────────────────────
# These templates define the instructions for the LLM for different tasks.

_PROMPT_EXERCISE = """You are an encouraging English coach.

Japanese phrase (JP): {jp}
Target chunk (EN): {en_chunk}
---
Create one short practice question **in ENGLISH** that requires the learner to use
the target chunk exactly once in a natural, everyday context.

Return JSON with keys:
- "question": the English prompt for the learner
- "answer_key": one model answer that correctly uses the chunk
"""

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

# -------------------- Pydantic Schemas for LLM Output Parsing --------------------
# These schemas define the expected structure of the JSON output from the LLM.

class ExerciseSchema(BaseModel):
    """Pydantic schema for parsing the output of exercise generation LLM call."""
    question: str = Field(..., description="English prompt for the learner")
    answer_key: str = Field(..., description="One model answer using the chunk")

class ReviewSchema(BaseModel):
    """Pydantic schema for parsing the output of answer review LLM call."""
    score: int = Field(..., description="Score 0-5 based on the user's answer")
    better: Optional[str] = Field(None, description="A suggested better alternative answer, if any")
    comment: str = Field(..., description="Feedback commentary in Japanese")


# -------------------- Graph State Definition --------------------
class GraphState(TypedDict):
    """
    Defines the state structure for the LangGraph application.
    
    This TypedDict holds all the data that is passed between nodes in the graph.
    Fields are optional where appropriate to allow for conditional execution paths
    (e.g., feedback is only present after the review node).
    """
    jp_prompt: str                 # Original Japanese prompt for the chunk.
    en_answer: str                 # Original English answer/chunk.
    user_input: Optional[str]      # User's attempted answer to the exercise.
    question: Optional[str]        # Generated English question for the user.
    answer_key: Optional[str]      # Generated model answer for the question.
    feedback: Optional[str]        # Feedback provided after reviewing user_input.


# -------------------- LLM Configuration --------------------
# Initialize the ChatOpenAI model instance.
# The API key is sourced from app_config, set at application startup.
# Temperature is set for a balance of creativity and predictability.
# This LLM instance is shared by different chains/nodes.
llm = ChatOpenAI(
    api_key=app_config.openai_api_key, model="gpt-4o-mini", temperature=0.7
)

# Create LLM chains by attaching the Pydantic schemas for structured output.
# llm_ex will output JSON matching ExerciseSchema.
# llm_fb will output JSON matching ReviewSchema.
llm_ex = llm.with_structured_output(ExerciseSchema)
llm_fb = llm.with_structured_output(ReviewSchema)


# -------------------- Graph Node Functions --------------------
# These functions define the individual processing steps (nodes) in the graph.

def generate_exercise_node(state: GraphState) -> GraphState:
    """
    Generates an exercise question and its answer key using the LLM.

    This node is skipped if a question and answer_key are already present in the state,
    allowing for reuse of previously generated exercises.

    Args:
        state: The current graph state. Must contain 'jp_prompt' and 'en_answer'.

    Returns:
        An updated graph state with 'question' and 'answer_key' populated.
    """
    logger.info("Executing generate_exercise_node.")
    # Skip generation if question and answer_key are already provided in the input state.
    # This is useful when feedback is requested for an existing question.
    if state.get("question") and state.get("answer_key"):
        logger.info("Skipping exercise generation as question and answer_key already in state.")
        return state

    # Define the chain for exercise generation: PromptTemplate -> LLM (with ExerciseSchema)
    template_ex = PromptTemplate.from_template(_PROMPT_EXERCISE)
    chain = template_ex | llm_ex

    try:
        logger.info(f"Generating exercise for JP: '{state['jp_prompt']}', EN: '{state['en_answer']}'")
        # Invoke the chain with necessary inputs from the state
        result = chain.invoke({"jp": state["jp_prompt"], "en_chunk": state["en_answer"]})

        # Update the state with the generated question and answer key.
        # A copy of the state is made to ensure immutability of the input state.
        new_state = state.copy()
        new_state["question"] = result.question
        new_state["answer_key"] = result.answer_key
        logger.info(f"Exercise generated: Q='{result.question}', AK='{result.answer_key}'")
        return new_state
    except Exception as e:
        logger.error(f"Error in generate_exercise_node: {e}")
        # Return state with error message or raise exception
        error_state = state.copy()
        error_state["question"] = "Error generating question."
        error_state["answer_key"] = "Error generating answer key."
        return error_state


def review_output_node(state: GraphState) -> GraphState:
    """
    Reviews the user's input against the model answer and generates feedback.

    Args:
        state: The current graph state. Must contain 'en_answer' (target chunk),
               'user_input', and 'answer_key' (model answer for the question).

    Returns:
        An updated graph state with 'feedback' populated.
    """
    logger.info("Executing review_output_node.")
    # Safeguard: Ensure necessary fields for review are present.
    # This should ideally be guaranteed by the graph's conditional logic.
    if not state.get("user_input") or not state.get("answer_key"):
        logger.warning("User input or answer key missing for review.")
        error_state = state.copy()
        error_state["feedback"] = "Error: User input or model answer key missing for review."
        return error_state

    # Define the chain for feedback generation: PromptTemplate -> LLM (with ReviewSchema)
    template_fb = PromptTemplate.from_template(_PROMPT_REVIEW)
    chain = template_fb | llm_fb

    try:
        logger.info(f"Reviewing user input: '{state['user_input']}' against answer key: '{state['answer_key']}' for chunk: '{state['en_answer']}'")
        # Invoke the chain with necessary inputs from the state
        result = chain.invoke(
            {
                "en_chunk": state["en_answer"],      # The original English chunk being practiced
                "user_answer": state["user_input"],  # The user's attempt
                "model_answer": state["answer_key"], # The model's answer to the generated question
            }
        )

        # Format the feedback into a markdown string
        comment_md = f"**Score:** {result.score}/5\n\n"
        if result.better: # Add suggested better answer if provided by LLM
            comment_md += f"**Better:** {result.better}\n\n"
        comment_md += result.comment

        # Update the state with the generated feedback.
        feedback_state = state.copy()
        feedback_state["feedback"] = comment_md
        logger.info(f"Feedback generated: Score={result.score}, Better='{result.better}', Comment='{result.comment}'")
        return feedback_state
    except Exception as e:
        logger.error(f"Error in review_output_node: {e}")
        error_state = state.copy()
        error_state["feedback"] = "Error generating feedback."
        return error_state

# -------------------- Conditional Edge Logic --------------------
# This function determines the next step in the graph after exercise generation.

def should_review(state: GraphState) -> str:
    """
    Determines the next node based on the presence of user input.

    If `user_input` is present and non-empty, the graph transitions to the
    `review_output_node`. Otherwise, the graph ends.

    Args:
        state: The current graph state.

    Returns:
        A string indicating the name of the next node or END.
    """
    user_input_present = state.get("user_input") and state.get("user_input","").strip()
    logger.info(f"Conditional edge 'should_review': User input present = {bool(user_input_present)}. Transitioning to '{'review_output_node' if user_input_present else 'END'}'")
    if user_input_present:
        return "review_output_node"  # Proceed to review if user input exists
    else:
        return END  # End the graph if no user input is provided

# -------------------- Graph Assembly --------------------
# Construct the LangGraph StateGraph.

# Initialize a new StateGraph with the defined GraphState.
workflow = StateGraph(GraphState)

# Add the defined functions as nodes in the graph.
workflow.add_node("generate_exercise_node", generate_exercise_node)
workflow.add_node("review_output_node", review_output_node)

# Set the entry point for the graph.
# All executions will begin at the 'generate_exercise_node'.
workflow.set_entry_point("generate_exercise_node")

# Define conditional edges from 'generate_exercise_node'.
# The 'should_review' function will determine the next step:
# - If 'should_review' returns "review_output_node", transition to 'review_output_node'.
# - If 'should_review' returns END, the graph execution finishes.
workflow.add_conditional_edges(
    "generate_exercise_node",  # Source node
    should_review,             # Function to determine the route
    {                          # Mapping of function return values to next nodes
        "review_output_node": "review_output_node",
        END: END
    }
)

# Define the edge from 'review_output_node'.
# After review, the graph always ends.
workflow.add_edge("review_output_node", END)


# Compile the graph into a runnable application.
# This compiled 'app' can be invoked with an initial GraphState.
app = workflow.compile()
logger.info("LangGraph application compiled and ready.")
