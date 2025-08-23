from io import BytesIO
from typing import ClassVar

import PIL.Image
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.widgets import ContentSwitcher
from textual_image.widget import AutoImage, HalfcellImage

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels import BasePanel


class ImageViewer(BasePanel):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "sensor_msgs/msg/CompressedImage",  # ROS2
        "sensor_msgs/CompressedImage",  # ROS1
    }

    DEFAULT_CSS = """
    ContentSwitcher * {
        width: auto;
        height: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("a", "set_renderer('auto')", "Auto"),
        ("h", "set_renderer('cell')", "Halfcell"),
    ]

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with ContentSwitcher(initial="cell"):
            yield HalfcellImage(id="cell")
            yield AutoImage(id="auto")

    def watch_data(self, _data: MessageEvent | None) -> None:
        img = None
        if self.data:
            img = PIL.Image.open(BytesIO(self.data.message.data))

        content = self.query_one(ContentSwitcher)
        if content.current == "cell":
            img_container = self.query_one(HalfcellImage)
        else:
            img_container = self.query_one(AutoImage)
        img_container.image = img

    def action_set_renderer(self, renderer: str) -> None:
        """Set the image renderer."""
        content = self.query_one(ContentSwitcher)
        content.current = renderer
