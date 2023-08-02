from typing import Any, Callable, Optional, Union
import fire
from cot_transparency.model_apis import convert_to_completion_str, convert_to_strict_messages
from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    TaskOutput,
    TaskSpec,
    StageTwoTaskSpec,
)
from cot_transparency.data_models.io import LoadedJsonType, ExpLoader

from tkinter import LEFT, Frame, Tk, Label, Button, Text, END, OptionMenu, StringVar
from random import choice

LoadedJsonTupleType = Union[
    dict[tuple[str, str, str], ExperimentJsonFormat], dict[tuple[str, str, str], StageTwoExperimentJsonFormat]
]


class GUI:
    def __init__(
        self,
        frame: Frame,
        json_dict: LoadedJsonTupleType,
        width: int = 150,
        update_callback: Optional[Callable[..., bool]] = None,
    ):
        config_width = width // 3
        self.frame = frame
        self.json_dict = json_dict
        self.keys = list(self.json_dict.keys())
        self.index = 0
        self.fontsize = 16
        self.alignment = "center"  # change to "center" to center align
        self.task_var = self.keys[0][0]
        self.update_callback = update_callback or (lambda: None)

        self.file_label = Label(frame, text="Select a file:", font=("Arial", self.fontsize))
        self.file_label.pack(anchor=self.alignment)

        self.model_var = StringVar(frame)
        self.model_var.set(self.keys[0][1])  # default value

        self.formatter_var = StringVar(frame)
        self.formatter_var.set(self.keys[0][2])  # default value

        self.model_dropdown = OptionMenu(
            frame, self.model_var, *{key[1] for key in self.keys}, command=self.select_json
        )
        self.model_dropdown.config(font=("Arial", self.fontsize))
        self.model_dropdown.pack(anchor=self.alignment)

        self.formatter_dropdown = OptionMenu(
            frame, self.formatter_var, *{key[2] for key in self.keys}, command=self.select_json
        )
        self.formatter_dropdown.config(font=("Arial", self.fontsize))
        self.formatter_dropdown.pack(anchor=self.alignment)

        self.label2 = Label(frame, text="Messages:", font=("Arial", self.fontsize))
        self.label2.pack(anchor=self.alignment)

        self.messages_text = Text(frame, width=width, height=20, font=("Arial", self.fontsize))
        self.messages_text.pack(anchor=self.alignment)

        self.config_and_output_frame = Frame(frame)
        self.config_and_output_frame.pack(anchor=self.alignment)

        self.config_frame = Frame(self.config_and_output_frame, width=config_width)
        self.label = Label(self.config_frame, text="Config:", font=("Arial", self.fontsize))
        self.label.pack(anchor=self.alignment)
        self.config_text = Text(self.config_frame, width=config_width, height=10, font=("Arial", self.fontsize))
        self.config_text.pack(anchor=self.alignment)

        self.config_frame.pack(side=LEFT)

        self.output_frame = Frame(self.config_and_output_frame, width=width - config_width)
        self.label3 = Label(
            self.config_and_output_frame,
            text="Model Output:",
            font=("Arial", self.fontsize),
        )
        self.label3.pack(anchor=self.alignment)
        self.output_text = Text(self.output_frame, width=width - config_width, height=10, font=("Arial", self.fontsize))
        self.output_text.pack(anchor=self.alignment)

        self.parsed_ans_label = Label(frame, text="Parsed Answer:", font=("Arial", self.fontsize))
        self.parsed_ans_label.pack(anchor=self.alignment)
        self.parsed_ans_text = Text(
            self.output_frame, width=width - config_width, height=1, font=("Arial", self.fontsize)
        )
        self.output_frame.pack(side=LEFT)

        # Display the first JSON
        self.select_json()
        self.display_output()

    def select_json(self, *args: Any):
        key = (self.task_var, self.model_var.get(), self.formatter_var.get())
        try:
            self.selected_exp = self.json_dict[key]
            self.display_output()

            # Do something with the data
        except KeyError:
            self.clear_fields()
            self.display_error()

    def prev_output(self):
        self.index = (self.index - 1) % len(self.selected_exp.outputs)
        self.display_output()

    def next_output(self):
        self.index = (self.index + 1) % len(self.selected_exp.outputs)
        self.display_output()

    def select_index(self, idx: int):
        self.index = idx
        self.display_output()

    def clear_fields(self):
        self.messages_text.delete("1.0", END)
        self.config_text.delete("1.0", END)
        self.output_text.delete("1.0", END)
        # clear other fields as necessary

    def display_error(self):
        self.messages_text.insert(END, "DATA NOT FOUND")
        self.config_text.insert(END, "DATA NOT FOUND")
        self.output_text.insert(END, "DATA NOT FOUND")
        # add to other fields as necessary

    def display_output(self):
        self.update_callback()
        if not self.update_callback():
            self.display_error()
        experiment = self.selected_exp

        # Clear previous text
        self.config_text.delete("1.0", END)
        self.messages_text.delete("1.0", END)
        self.output_text.delete("1.0", END)

        # Insert new text
        output = experiment.outputs[self.index]

        strict_messages = convert_to_strict_messages(output.task_spec.messages, output.task_spec.model_config.model)
        formatted_output = convert_to_completion_str(strict_messages)
        self.config_text.insert(END, str(output.task_spec.model_config.json(indent=2)))
        self.messages_text.insert(END, formatted_output)
        self.output_text.insert(END, str(output.first_raw_response))


class CompareGUI:
    def __init__(self, master: Tk, json_dict: LoadedJsonTupleType, width: int = 150, n_compare: int = 2):
        width_of_each = width // n_compare
        self.fontsize = 16
        self.alignment = "center"  # change to "center" to center align
        self.keys = list(json_dict.keys())

        self.base_guis: list[GUI] = []
        for i in range(n_compare):
            frame = Frame(master)
            # add the callback to the last base_gui
            if i == n_compare - 1:
                callback = self.check_all_on_the_same_task
            else:
                callback = None

            self.base_guis.append(GUI(frame, json_dict, width_of_each, callback))
            frame.grid(row=1, column=i)

        self.task_var = StringVar(master)
        self.task_var.set(self.keys[0][0])  # default value

        self.task_dropdown = OptionMenu(
            master,
            self.task_var,
            *{key[0] for key in self.keys},
            command=self.select_task,  # type: ignore
        )
        self.task_dropdown.config(font=("Arial", self.fontsize))
        self.task_dropdown.grid(row=0, column=0, columnspan=n_compare)

        self.ground_truth_label = Label(master, text="Ground Truth:", font=("Arial", self.fontsize))
        self.ground_truth_label.grid(row=2, column=0, columnspan=n_compare)

        self.buttons_frame = Frame(master)
        self.buttons_frame.grid(row=3, column=0, columnspan=n_compare)

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

        self.display_output()

    def check_all_on_the_same_task(self) -> bool:
        # check that all gui's are on the same index and have the same task_hash
        task_hashes = []
        for gui in self.base_guis:
            task_spec = gui.selected_exp.outputs[gui.index].task_spec
            if isinstance(task_spec, StageTwoTaskSpec):
                task_hashes.append(task_spec.stage_one_output.task_spec.task_hash)
        unique_task_hashes = set(task_hashes)
        if len(unique_task_hashes) <= 1:
            return True
        else:
            print("All GUIs must be on the same task_hash")
            return False

    def display_output(self):
        # Insert new text
        exp: Union[ExperimentJsonFormat, StageTwoExperimentJsonFormat] = self.base_guis[0].selected_exp
        index: int = self.base_guis[0].index
        output: Union[TaskOutput, StageTwoTaskOutput] = exp.outputs[index]

        if isinstance(output.task_spec, TaskSpec):
            self.ground_truth_label.config(text=f"Ground Truth: {output.task_spec.ground_truth}")
        elif isinstance(output.task_spec, StageTwoTaskSpec):
            self.ground_truth_label.config(
                text=f"Ground Truth: {output.task_spec.stage_one_output.task_spec.ground_truth}"
            )

    def prev_output(self):
        for gui in self.base_guis:
            gui.prev_output()
        self.display_output()

    def next_output(self):
        for gui in self.base_guis:
            gui.next_output()
        self.display_output()

    def random_output(self):
        idx = choice(range(len(self.base_guis[0].selected_exp.outputs)))
        for gui in self.base_guis:
            gui.select_index(idx)
        self.display_output()

    def select_task(self, task_name: str):
        for gui in self.base_guis:
            gui.task_var = task_name
            gui.select_json()
        self.display_output()


def convert_loaded_json_keys_to_tuples(
    loaded_dict: LoadedJsonType,
) -> LoadedJsonTupleType:
    # path_format is exp_dir/task_name/model/formatter.json
    out = {}
    for path, exp in loaded_dict.items():
        out[(path.parent.parent.name, path.parent.name, path.name)] = exp
    return out


def sort_stage1(loaded_dict: dict[tuple[str, str, str], ExperimentJsonFormat]):
    for exp in loaded_dict.values():
        exp.outputs.sort(key=lambda x: x.task_spec_uid())


def sort_stage2(loaded_dict: dict[tuple[str, str, str], StageTwoExperimentJsonFormat]):
    for exp in loaded_dict.values():
        exp.outputs.sort(
            key=lambda x: (
                x.task_spec.stage_one_output.task_spec.task_hash,
                x.task_spec.stage_one_output.uid(),
                x.task_spec.step_in_cot_trace,
            )
        )  # type: ignore


def main(exp_dir: str, width: int = 175, n_compare: int = 1):
    # Load the JSONs here
    # establish if this stage one or stage two

    stage = ExpLoader.get_stage(exp_dir)
    if stage == 1:
        loaded_jsons = ExpLoader.stage_one(exp_dir)
        loaded_jsons_with_tuples = convert_loaded_json_keys_to_tuples(loaded_jsons)
        sort_stage1(loaded_jsons_with_tuples)  # type: ignore
    else:
        loaded_jsons = ExpLoader.stage_two(exp_dir)
        loaded_jsons_with_tuples = convert_loaded_json_keys_to_tuples(loaded_jsons)
        sort_stage2(loaded_jsons_with_tuples)  # type: ignore

    root = Tk()
    CompareGUI(root, loaded_jsons_with_tuples, width, n_compare)
    root.mainloop()


if __name__ == "__main__":
    fire.Fire(main)
