# test_exercise.py
from exercises import ExerciseGenerator


def test_create_exercise():
    gen = ExerciseGenerator(api_key="dummy")
    ex = gen.create_exercise("ごきげんよう", "How are you?")
    assert "How are you?" in ex.question
    assert ex.answer_key == "How are you?"


def test_review_feedback():
    gen = ExerciseGenerator(api_key="dummy")
    ex = gen.create_exercise("最近どうよ？", "How's it going?")
    fb = gen.review_answer("How's it going?", ex, en_chunk="How's it going?")
    assert fb.score == 4
    assert "Perfect" in fb.comment_md
