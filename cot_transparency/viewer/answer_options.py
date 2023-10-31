from enum import Enum

import streamlit as st

from cot_transparency.util import assert_not_none


class TypeOfAnswerOption(str, Enum):
    wrong_answer = "wrong_answer"
    correct_answer = "correct_answer"
    anything = "anything"
    not_parseable = "not parseable"

    def pretty(self) -> str:
        return self.replace("_", " ").title()


def select_bias_on_where_option() -> TypeOfAnswerOption:
    selected: TypeOfAnswerOption = assert_not_none(
        st.selectbox(
            "Where is the bias on for the left model?",
            options=[
                TypeOfAnswerOption.wrong_answer,
                TypeOfAnswerOption.correct_answer,
                TypeOfAnswerOption.anything,
            ],
            index=0,
            format_func=lambda x: x.pretty(),
        )
    )
    return selected


def select_left_model_result_option() -> TypeOfAnswerOption:
    selected: TypeOfAnswerOption = assert_not_none(
        st.selectbox(
            "What answer did the left model give?",
            options=[
                TypeOfAnswerOption.wrong_answer,
                TypeOfAnswerOption.correct_answer,
                TypeOfAnswerOption.anything,
                TypeOfAnswerOption.not_parseable,
            ],
            index=0,
            format_func=lambda x: x.pretty(),
        )
    )
    return selected
