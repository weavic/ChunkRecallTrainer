import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys

# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock app_config before graph module imports it
# This is important because graph.py uses app_config.openai_api_key at the module level
# when defining `llm = ChatOpenAI(...)`
mock_app_config_instance = MagicMock()
mock_app_config_instance.openai_api_key = "fake_test_api_key" 

# Apply the mock to the actual location where graph.py will import it from
# This assumes graph.py does `from .config import app_config`
# If AppConfig is imported and then app_config is an instance, the target is different.
# Let's assume `app_config` is the direct import from `.config`
# The key is that the mock must be in place *before* `graph.py` is loaded by the test runner.
# This can be tricky. A common way is to ensure this mock is active when Python loads graph.py.
# One way is to patch it in `__init__.py` of the tests or at the top of the test file.
# For simplicity here, we'll patch it where it's looked up by graph.py.
# The target for patch should be 'chunk_recall_trainer.graph.app_config'
# because graph.py is `src/chunk_recall_trainer/graph.py` and it imports `.config.app_config`
# so inside graph.py, `app_config` refers to `chunk_recall_trainer.config.app_config`.

# However, the LLM objects (llm, llm_ex, llm_fb) are created at module level in graph.py.
# So we need to mock ChatOpenAI itself when graph.py is imported.
# This is best done by patching 'langchain_openai.ChatOpenAI' *before* importing graph.py elements.

# Mock ChatOpenAI before importing graph elements
mock_chat_open_ai_class = MagicMock()
# mock_chat_open_ai_instance = MagicMock()
# mock_chat_open_ai_class.return_value = mock_chat_open_ai_instance # When ChatOpenAI() is called

# Patch 'langchain_openai.ChatOpenAI' globally for the duration of this test module loading.
# This ensures that when graph.py is imported, it uses our mock.
# This is a bit of a "heavy hammer" but necessary for module-level initializations.

# It's often easier to refactor the SUT (system under test) to allow dependency injection
# for LLMs, but given the current structure, we patch.

# We need to patch where ChatOpenAI is looked up *by graph.py*.
# graph.py: from langchain_openai import ChatOpenAI
# So, the target for patching is 'chunk_recall_trainer.graph.ChatOpenAI'
# or more globally 'langchain_openai.ChatOpenAI' if that's easier.
# Let's try patching it directly where it's imported in graph.py for more targeted mocking.

with patch('chunk_recall_trainer.graph.ChatOpenAI', new=mock_chat_open_ai_class):
    # Now import the graph elements. They will be defined using the mocked ChatOpenAI.
    from chunk_recall_trainer.graph import (
        GraphState,
        generate_exercise_node,
        review_output_node,
        should_review,
        app as graph_app, # Compiled graph
        ExerciseSchema, # For mock return values
        ReviewSchema   # For mock return values
    )
    # Also, the llm, llm_ex, llm_fb in graph.py are now instances of our MagicMock() class,
    # or rather, their .with_structured_output methods are called on it.
    # We need to make sure `mock_chat_open_ai_class().with_structured_output()` returns a callable mock.
    mock_llm_instance = mock_chat_open_ai_class.return_value
    mock_llm_instance.with_structured_output.return_value = MagicMock() # This mock will be `llm_ex` and `llm_fb`


class TestGraphLogic(unittest.TestCase):

    def setUp(self):
        # Reset mocks for llm_ex and llm_fb for each test if they are used directly.
        # These were created at import time using the patched ChatOpenAI.
        # We need to control their `invoke` method.
        # `llm_ex` and `llm_fb` are the result of `mock_llm_instance.with_structured_output()`
        # So, we re-assign the mock for each test.
        self.mock_llm_ex = mock_llm_instance.with_structured_output(ExerciseSchema)
        self.mock_llm_fb = mock_llm_instance.with_structured_output(ReviewSchema)

        # If the graph nodes use `template | llm_ex_instance_in_graph_module`,
        # then we need to mock that `llm_ex_instance_in_graph_module.invoke`.
        # The instances `llm_ex` and `llm_fb` in `graph.py` are what we need to mock.
        # Since they are top-level in graph.py, we can patch them directly for tests.
        
        # This is getting complicated because of module-level instantiation.
        # A cleaner way is to ensure that `llm_ex` and `llm_fb` *inside graph.py* are mocks.
        # The `with patch` at import time should handle this.
        # `llm_ex` and `llm_fb` in graph.py are now MagicMocks.
        
        # Let's re-evaluate. The `llm_ex` and `llm_fb` in graph.py are already
        # `MagicMock` instances because `with_structured_output` returned a `MagicMock`.
        # So we can just refer to them via the imported graph module if needed,
        # or ensure our mocks passed to nodes are used.

        # For the node tests, we'll directly mock the `invoke` calls.
        # The `chain.invoke` pattern is `(template | llm_structured_output_mock).invoke`
        # We need to mock the result of the `invoke` call on the llm_structured_output_mock.
        pass


    def test_generate_exercise_node_generates(self):
        """Test generate_exercise_node generates question and answer_key."""
        initial_state: GraphState = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": None, "question": None, "answer_key": None, "feedback": None
        }
        
        # Mock the return value of llm_ex.invoke (which is used inside the node)
        # This assumes llm_ex is the one imported from graph.py and is a mock
        # from our initial patch.
        # We need to access the `llm_ex` that the node `generate_exercise_node` sees.
        
        # To do this, we patch `llm_ex` within the `chunk_recall_trainer.graph` module for this test.
        with patch('chunk_recall_trainer.graph.llm_ex') as mock_graph_llm_ex:
            mock_graph_llm_ex.invoke.return_value = ExerciseSchema(question="Generated Q", answer_key="Generated AK")
            
            result_state = generate_exercise_node(initial_state)

            self.assertEqual(result_state["question"], "Generated Q")
            self.assertEqual(result_state["answer_key"], "Generated AK")
            self.assertEqual(result_state["jp_prompt"], "こんにちは") # Ensure others are passed through
            mock_graph_llm_ex.invoke.assert_called_once_with(
                {"jp": "こんにちは", "en_chunk": "Hello"}
            )

    def test_generate_exercise_node_skips_if_present(self):
        """Test generate_exercise_node skips if question and answer_key are present."""
        initial_state: GraphState = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": None, 
            "question": "Existing Q", "answer_key": "Existing AK", 
            "feedback": None
        }
        with patch('chunk_recall_trainer.graph.llm_ex') as mock_graph_llm_ex:
            result_state = generate_exercise_node(initial_state)
            
            self.assertEqual(result_state["question"], "Existing Q")
            self.assertEqual(result_state["answer_key"], "Existing AK")
            mock_graph_llm_ex.invoke.assert_not_called()


    def test_review_output_node_generates_feedback(self):
        """Test review_output_node generates feedback."""
        initial_state: GraphState = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": "Hallo", 
            "question": "Q1", "answer_key": "Hello", 
            "feedback": None
        }
        with patch('chunk_recall_trainer.graph.llm_fb') as mock_graph_llm_fb:
            mock_graph_llm_fb.invoke.return_value = ReviewSchema(score=3, better="Hola", comment="Good try")
            
            result_state = review_output_node(initial_state)

            expected_feedback = "**Score:** 3/5\n\n**Better:** Hola\n\nGood try"
            self.assertEqual(result_state["feedback"], expected_feedback)
            mock_graph_llm_fb.invoke.assert_called_once_with({
                "en_chunk": "Hello",
                "user_answer": "Hallo",
                "model_answer": "Hello"
            })

    def test_review_output_node_error_if_missing_input(self):
        """Test review_output_node handles missing user_input or answer_key."""
        state_no_input: GraphState = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": None, "question": "Q1", "answer_key": "Hello", "feedback": None
        }
        state_no_ak: GraphState = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": "Hi", "question": "Q1", "answer_key": None, "feedback": None
        }
        with patch('chunk_recall_trainer.graph.llm_fb') as mock_graph_llm_fb:
            result_state_1 = review_output_node(state_no_input)
            self.assertEqual(result_state_1["feedback"], "Error: User input or model answer key missing for review.")
            mock_graph_llm_fb.invoke.assert_not_called()

            result_state_2 = review_output_node(state_no_ak)
            self.assertEqual(result_state_2["feedback"], "Error: User input or model answer key missing for review.")
            mock_graph_llm_fb.invoke.assert_not_called()


    def test_should_review_with_input(self):
        """Test should_review returns 'review_output_node' if user_input is present."""
        state: GraphState = {"user_input": "Some answer", "jp_prompt": "", "en_answer": "", "question": "", "answer_key": "", "feedback": ""}
        self.assertEqual(should_review(state), "review_output_node")

    def test_should_review_without_input(self):
        """Test should_review returns END if user_input is None or empty."""
        state_none: GraphState = {"user_input": None, "jp_prompt": "", "en_answer": "", "question": "", "answer_key": "", "feedback": ""}
        state_empty: GraphState = {"user_input": "  ", "jp_prompt": "", "en_answer": "", "question": "", "answer_key": "", "feedback": ""}
        
        # Assuming END is imported from langgraph.graph
        from langgraph.graph import END as LANGGRAPH_END 
        self.assertEqual(should_review(state_none), LANGGRAPH_END)
        self.assertEqual(should_review(state_empty), LANGGRAPH_END)


    # Testing the compiled graph `app`
    # We need to mock the `invoke` methods of `llm_ex` and `llm_fb` that are part of the compiled graph.
    # These are the instances created in graph.py at module level.
    # The patch at the top of the file should ensure these are mocks.
    @patch('chunk_recall_trainer.graph.llm_ex') # patch the object within graph.py
    @patch('chunk_recall_trainer.graph.llm_fb')  # patch the object within graph.py
    def test_graph_app_question_generation_only(self, mock_llm_fb_in_graph, mock_llm_ex_in_graph):
        """Test compiled graph for question generation path."""
        mock_llm_ex_in_graph.invoke.return_value = ExerciseSchema(question="Test Q", answer_key="Test AK")
        
        input_state = {
            "jp_prompt": "こんにちは", "en_answer": "Hello", 
            "user_input": None # No user input
        }
        # For a compiled graph, input only needs defined fields. Optional can be omitted.
        # GraphState allows them to be optional.
        
        result = graph_app.invoke(input_state)
        
        self.assertEqual(result["question"], "Test Q")
        self.assertEqual(result["answer_key"], "Test AK")
        self.assertIsNone(result.get("feedback")) # No feedback should be generated
        mock_llm_ex_in_graph.invoke.assert_called_once()
        mock_llm_fb_in_graph.invoke.assert_not_called()

    @patch('chunk_recall_trainer.graph.llm_ex')
    @patch('chunk_recall_trainer.graph.llm_fb')
    def test_graph_app_feedback_generation(self, mock_llm_fb_in_graph, mock_llm_ex_in_graph):
        """Test compiled graph for feedback generation path."""
        # generate_exercise_node will be called, but it should skip LLM call
        # because question and answer_key are provided.
        mock_llm_fb_in_graph.invoke.return_value = ReviewSchema(score=4, better=None, comment="OK")
        
        input_state = {
            "jp_prompt": "こんにちは", "en_answer": "Hello",
            "user_input": "Hallo",
            "question": "Provided Q", # Skips generation
            "answer_key": "Provided AK" # Skips generation
        }
        
        result = graph_app.invoke(input_state)
        
        self.assertEqual(result["question"], "Provided Q") # Passed through
        self.assertEqual(result["answer_key"], "Provided AK") # Passed through
        self.assertIn("Score:** 4/5", result["feedback"])
        self.assertIn("OK", result["feedback"])
        
        mock_llm_ex_in_graph.invoke.assert_not_called() # Should be skipped
        mock_llm_fb_in_graph.invoke.assert_called_once()

if __name__ == "__main__":
    unittest.main()
