from enum import Enum

from cot_transparency.data_models.hashable import HashableBaseModel

from pydantic import ConfigDict


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # If you are OpenAI chat, you need to add this back into the previous user message
    # Anthropic can handle it as per normal like an actual assistant
    assistant_if_completion = "assistant_preferred"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class StrictMessageRole(str, Enum):
    # Stricter set of roles that doesn't allow assistant_preferred
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str
    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def remove_role(self) -> "ChatMessage":
        return ChatMessage(role=MessageRole.none, content=self.content)

    def add_question_prefix(self) -> "ChatMessage":
        if self.content.startswith("Question: "):
            return self
        return ChatMessage(role=self.role, content=f"Question: {self.content}")

    def add_answer_prefix(self) -> "ChatMessage":
        if self.content.startswith("Answer: "):
            return self
        return ChatMessage(role=self.role, content=f"Answer: {self.content}")


class StrictChatMessage(HashableBaseModel):
    role: StrictMessageRole
    content: str
    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)
