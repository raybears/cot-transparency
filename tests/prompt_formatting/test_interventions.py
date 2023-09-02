# Parametrize over ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES


# @pytest.mark.parametrize(
#     "formatter_name",
#     [
#         "ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN",
#         "ZeroShotPromptSenFormatter_FOO_FULL_ANS_CHOICES_PAREN",
#     ],
# )
# def test_all_answers_same_format(formatter_name: str):
#     question = EMPIRE_OF_PANTS_EXAMPLE

#     formatter = name_to_formatter(formatter_name)
#     if issubclass(formatter, PromptSenBaseFormatter):
#         data_format_spec = formatter.get_data_format_spec()
#     else:
#         raise ValueError(f"Formatter {formatter_name} is not a PromptSensitivityFormatter")

#     msgs = VanillaFewShotLabelOnly20.intervene(question, formatter, model="gpt-4")

#     prompt = Prompt(messages=msgs)
#     print(prompt.convert_to_completion_str())

#     print(len(msgs))
#     assert len(msgs) == 41

#     possible_answers = data_format_spec.choice_variant.answers_list
#     for msg in msgs:
#         if msg.role == MessageRole.assistant:
#             output = msg.content
#             assert sum([i in output for i in possible_answers]) == 1


# if __name__ == "__main__":
#     test_all_answers_same_format("ZeroShotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_PAREN")
