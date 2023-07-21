from typing import Any, Union
import fire
from cot_transparency.model_apis import format_for_completion
from cot_transparency.data_models.models import ExperimentJsonFormat, StageTwoExperimentJsonFormat
from cot_transparency.data_models.io import LoadedJsonType, ExpLoader

from pathlib import Path
from tkinter import LEFT, Frame, Tk, Label, Button, Text, END, OptionMenu, StringVar
from random import choice

LoadedJsonTupleType = Union[
    dict[tuple[str, str, str], ExperimentJsonFormat], dict[tuple[str, str, str], StageTwoExperimentJsonFormat]
]


class GUI:
    def __init__(self, master: Tk, json_dict: LoadedJsonTupleType):
        width = 150
        self.master = master
        self.json_dict = json_dict
        self.keys = list(self.json_dict.keys())
        self.index = 0
        self.master.title("JSON Viewer")
        self.fontsize = 16
        self.alignment = "center"  # change to "center" to center align

        self.file_label = Label(master, text="Select a file:", font=("Arial", self.fontsize))
        self.file_label.pack(anchor=self.alignment)

        self.task_var = StringVar(master)
        self.task_var.set(self.keys[0][0])  # default value

        self.model_var = StringVar(master)
        self.model_var.set(self.keys[0][1])  # default value

        self.formatter_var = StringVar(master)
        self.formatter_var.set(self.keys[0][2])  # default value

        self.task_dropdown = OptionMenu(master, self.task_var, *{key[0] for key in self.keys}, command=self.select_json)
        self.task_dropdown.config(font=("Arial", self.fontsize))
        self.task_dropdown.pack(anchor=self.alignment)

        self.model_dropdown = OptionMenu(
            master, self.model_var, *{key[1] for key in self.keys}, command=self.select_json
        )
        self.model_dropdown.config(font=("Arial", self.fontsize))
        self.model_dropdown.pack(anchor=self.alignment)

        self.formatter_dropdown = OptionMenu(
            master, self.formatter_var, *{key[2] for key in self.keys}, command=self.select_json
        )
        self.formatter_dropdown.config(font=("Arial", self.fontsize))
        self.formatter_dropdown.pack(anchor=self.alignment)

        self.label = Label(master, text="Config:", font=("Arial", self.fontsize))
        self.label.pack(anchor=self.alignment)

        self.config_text = Text(master, width=width, height=10, font=("Arial", self.fontsize))
        self.config_text.pack(anchor=self.alignment)

        self.label2 = Label(master, text="Messages:", font=("Arial", self.fontsize))
        self.label2.pack(anchor=self.alignment)

        self.messages_text = Text(master, width=width, height=20, font=("Arial", self.fontsize))
        self.messages_text.pack(anchor=self.alignment)

        self.label3 = Label(
            master,
            text="Model Output:",
            font=("Arial", self.fontsize),
        )
        self.label3.pack(anchor=self.alignment)

        self.output_text = Text(master, width=width, height=5, font=("Arial", self.fontsize))
        self.output_text.pack(anchor=self.alignment)

        self.parsed_ans_label = Label(master, text="Parsed Answer:", font=("Arial", self.fontsize))
        self.parsed_ans_label.pack(anchor=self.alignment)

        self.ground_truth_label = Label(master, text="Ground Truth:", font=("Arial", self.fontsize))
        self.ground_truth_label.pack(anchor=self.alignment)

        self.buttons_frame = Frame(master)
        self.buttons_frame.pack(anchor=self.alignment)

        self.prev_button = Button(
            self.buttons_frame, text="Prev", command=self.prev_output, font=("Arial", self.fontsize)
        )
        self.prev_button.pack(side=LEFT)

        self.next_button = Button(
            self.buttons_frame, text="Next", command=self.next_output, font=("Arial", self.fontsize)
        )
        self.next_button.pack(side=LEFT)

        self.random_button = Button(
            self.buttons_frame, text="Random", command=self.random_output, font=("Arial", self.fontsize)
        )
        self.random_button.pack(side=LEFT)

        # Display the first JSON
        self.select_json()
        self.display_output()

    def select_json(self, *args: Any):
        selected_key = (self.task_var.get(), self.model_var.get(), self.formatter_var.get())
        self.selected_exp = self.json_dict[selected_key]
        self.display_output()

    def prev_output(self):
        self.index = (self.index - 1) % len(self.selected_exp.outputs)
        self.display_output()

    def next_output(self):
        self.index = (self.index + 1) % len(self.selected_exp.outputs)
        self.display_output()

    def random_output(self):
        len_of_current_exp = len(self.selected_exp.outputs)
        self.index = choice(range(len_of_current_exp))
        self.display_output()

    def display_output(self):
        experiment = self.selected_exp

        # Clear previous text
        self.config_text.delete("1.0", END)
        self.messages_text.delete("1.0", END)
        self.output_text.delete("1.0", END)

        # Insert new text
        output = experiment.outputs[self.index]

        formatted_output = format_for_completion(output.task_spec.messages)
        self.config_text.insert(END, str(output.task_spec.model_config.json(indent=2)))
        self.messages_text.insert(END, formatted_output)
        self.output_text.insert(END, str(output.first_raw_response))
        self.parsed_ans_label.config(text=f"Parsed Answer: {output.first_parsed_response}")
        self.ground_truth_label.config(text=f"Ground Truth: {output.ground_truth}")


# Add your load_jsons function here


def convert_loaded_json_keys_to_tuples(
    loaded_dict: LoadedJsonType,
) -> LoadedJsonTupleType:
    # path_format is exp_dir/task_name/model/formatter.json
    out = {}
    for path, exp in loaded_dict.items():
        out[(path.parent.parent.name, path.parent.name, path.name)] = exp
    return out


def sort_stage1(loaded_dict: dict[Path, ExperimentJsonFormat]):
    for exp in loaded_dict.values():
        exp.outputs.sort(key=lambda x: x.task_spec_uid())


def sort_stage2(loaded_dict: dict[Path, StageTwoExperimentJsonFormat]):
    for exp in loaded_dict.values():
        exp.outputs.sort(key=lambda x: (x.stage_one_hash, x.step_in_cot_trace))  # type: ignore


def main(exp_dir: str):
    # Load the JSONs here
    # establish if this stage one or stage two
    stage = ExpLoader.get_stage(exp_dir)
    if stage == 1:
        loaded_jsons = ExpLoader.stage_one(exp_dir)
        sort_stage1(loaded_jsons)  # type: ignore
    else:
        loaded_jsons = ExpLoader.stage_two(exp_dir)
        sort_stage2(loaded_jsons)  # type: ignore

    loaded_jsons_with_tuples = convert_loaded_json_keys_to_tuples(loaded_jsons)

    root = Tk()
    GUI(root, loaded_jsons_with_tuples)
    root.mainloop()


if __name__ == "__main__":
    fire.Fire(main)
