"""An ordered user message: text and image parts, rendered in exactly their order.

A conversation is ``[(text, image_or_None), ...]``: each tuple is one user
message, and each provider service decides where the tuple's images sit
(after the text on the SLURM / vLLM, OpenAI, OpenAI-compatible and Bedrock
routes, before it on the Claude and Google routes). That rendering is kept
byte for byte. A caller that needs a fixed order puts an ``OrderedParts`` in
the message slot instead of the tuple::

    conversations = [("q1", [OrderedParts([image, "What does this show?"])])]

Every service renders its parts in the order given. A ``str`` part is text;
any other part is one image (a ``PIL.Image.Image`` or an ``os.PathLike`` file
path, as the tuple's image slot accepts; a bare ``str`` is always text, so a
path goes as a ``pathlib.Path``). The local text-only service keeps the text
parts in order and drops the images, as it drops a tuple's.

Why a type of its own, not a plain list in the message slot: a plain list is
what a caller who builds ``[text, image]`` instead of ``(text, image)`` passes
today, and reading it as ordered parts would silently reorder that caller's
images on the Claude and Google routes. ``OrderedParts`` is not iterable, so
code that only knows the tuple form fails loudly on it instead of reading its
first two parts as ``(text, image)``.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple


class OrderedParts:
    """One user message as an ordered sequence of parts (``str`` text, or one image)."""

    __slots__ = ("parts",)

    def __init__(self, parts: Iterable[Any]):
        parts = tuple(parts)
        if not parts:
            raise ValueError("OrderedParts needs at least one part")
        for index, part in enumerate(parts):
            if part is None or isinstance(part, (list, tuple, dict, bytes, OrderedParts)):
                raise TypeError(
                    f"OrderedParts part {index} is {type(part).__name__}; each part is a str "
                    "(text) or one image (PIL.Image.Image or a pathlib.Path)")
        object.__setattr__(self, "parts", parts)

    def __setattr__(self, name, value):
        raise AttributeError("OrderedParts is immutable")

    def __eq__(self, other):
        return isinstance(other, OrderedParts) and self.parts == other.parts

    def __hash__(self):
        return hash(("OrderedParts", self.parts))

    def __repr__(self):
        kinds = ", ".join(repr(p) if isinstance(p, str) else f"<image {type(p).__name__}>"
                          for p in self.parts)
        return f"OrderedParts([{kinds}])"

    def __reduce__(self):
        return (OrderedParts, (self.parts,))

    @property
    def texts(self) -> List[str]:
        """The text parts, in order."""
        return [p for p in self.parts if isinstance(p, str)]

    @property
    def images(self) -> List[Any]:
        """The image parts, in order."""
        return [p for p in self.parts if not isinstance(p, str)]


def is_text(part: Any) -> bool:
    """Whether a part of an ``OrderedParts`` is text (else it is one image)."""
    return isinstance(part, str)


def image_path_or_object(part: Any) -> Any:
    """An image part as the tuple form's image slot holds it (a path as ``str``)."""
    return os.fspath(part) if isinstance(part, os.PathLike) else part


def tuple_view(
    conversations: Sequence[Tuple[str, Sequence[Any]]],
) -> Sequence[Tuple[str, List[Tuple[str, Optional[Any]]]]]:
    """The conversations with each ``OrderedParts`` read as a ``(text, images)`` tuple.

    For code that only sizes a job (the native-batch cost estimate): the text
    parts joined by a newline, the images in a list. Never what a provider
    receives. A job with no ``OrderedParts`` is returned as the same object.
    """
    if not any(isinstance(m, OrderedParts) for _cid, msgs in conversations for m in msgs):
        return conversations

    def as_tuple(message: Any) -> Any:
        if not isinstance(message, OrderedParts):
            return message
        images = [image_path_or_object(i) for i in message.images]
        return ("\n".join(message.texts), images or None)

    return [(cid, [as_tuple(m) for m in msgs]) for cid, msgs in conversations]


def openai_content(message: OrderedParts) -> List[dict]:
    """The OpenAI chat-completions content list of one ordered message (the
    OpenAI, OpenAI-compatible and SLURM / vLLM routes): each part as the
    block the tuple form renders it as, in the message's order."""
    from ..media_utils import encode_image_to_b64

    content: List[dict] = []
    for part in message.parts:
        if is_text(part):
            content.append({"type": "text", "text": part})
        else:
            b64, mime = encode_image_to_b64(image_path_or_object(part))
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
    return content
