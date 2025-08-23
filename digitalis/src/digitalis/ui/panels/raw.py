from datetime import datetime, timezone
from typing import Any, ClassVar

from rich.highlighter import ISO8601Highlighter, ReprHighlighter
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.validation import ValidationResult, Validator
from textual.widgets import Input, Tree
from textual.widgets.tree import TreeNode

from digitalis.grammar import ParsedMessagePath, parse_message_path
from digitalis.grammar.query import QueryError, apply_query
from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.base import SCHEMA_ANY, BasePanel
from digitalis.utilities import (
    NANOSECONDS_PER_SECOND,
    STRFTIME_FORMAT,
    nanoseconds_to_iso,
    quaternion_to_euler,
)

highlighter = ReprHighlighter()
iso_highlighter = ISO8601Highlighter()


def add_node(
    name: str, node: TreeNode, obj: Any, expand_depth: int, auto_expand: bool = True
) -> None:
    """Adds a node to the tree.

    Args:
        name (str): Name of the node.
        node (TreeNode): Parent node.
        obj (object): Data associated with the node.
        expand_depth (int): How deep to expand.
        auto_expand (bool): Whether to auto-expand nodes.
    """
    # Store the object's string representation for comparison
    node.data = repr(obj)

    if not hasattr(obj, "__slots__"):
        node.allow_expand = False
        if isinstance(obj, bytes):
            # only show the first 100 bytes of bytes objects
            start_bytes = obj[:100]
            obj_repr = f"bytes({len(obj)}): {start_bytes!r}..."
        else:
            obj_repr = repr(obj)
        if name:
            label = Text.assemble(Text.from_markup(f"[b]{name}[/b]="), highlighter(obj_repr))
        else:
            label = Text(obj_repr)
        node.set_label(label)
        return

    if obj.__slots__ == ["sec", "nanosec"]:
        # Convert timestamp to ISO format
        timestamp = iso_highlighter(
            datetime.fromtimestamp(
                obj.sec + obj.nanosec / NANOSECONDS_PER_SECOND, tz=timezone.utc
            ).strftime(STRFTIME_FORMAT)
        )
        stamp_label = Text.assemble(Text.from_markup(f"[b]{name}[/b]="), timestamp)
        node.set_label(stamp_label)
    elif obj.__slots__ == ["x", "y", "z", "w"]:
        # convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(obj.x, obj.y, obj.z, obj.w)
        euler_label = Text.assemble(
            Text.from_markup(f"[b]{name}[/b]="),
            highlighter(f"r={roll:.2f}, p={pitch:.2f}, y={yaw:.2f}"),
        )
        node.set_label(euler_label)
    elif expand_depth >= 0 and auto_expand:
        node.expand()

    for slot in obj.__slots__:
        data = getattr(obj, slot)
        child = node.add(slot)
        if isinstance(data, (list, tuple)):
            # Store array info as data for comparison
            child.data = f"array:{len(data)}"
            child.set_label(Text(f"{slot}[{len(data)}]"))
            for index, value in enumerate(data):
                new_node = child.add(f"[{index}]")
                if expand_depth >= 0 and auto_expand:
                    new_node.expand()
                add_node(str(index), new_node, value, expand_depth - 1, auto_expand)
                if index > 25:  # TODO: lazy load these?
                    add_node("Truncated", new_node, "...", expand_depth - 1, auto_expand)
                    break
        else:
            add_node(slot, child, data, expand_depth - 1, auto_expand)


class QueryValidator(Validator):
    """Validator for query syntax using parse_message_path."""

    def __init__(self) -> None:
        super().__init__()
        self._cached_parsed: ParsedMessagePath | None = None
        self._cached_query: str = ""

    def validate(self, value: str) -> ValidationResult:
        """Validate query syntax and cache parsed result.

        Args:
            value: The query string to validate

        Returns:
            ValidationResult indicating success or failure
        """
        if not value.strip():
            self._cached_parsed = None
            self._cached_query = value
            return self.success()

        try:
            parsed = parse_message_path(f"/dummy{value}")
            self._cached_parsed = parsed
            self._cached_query = value
            return self.success()
        except ValueError as e:
            self._cached_parsed = None
            self._cached_query = value
            return self.failure(f"Invalid query syntax: {e}")

    def get_cached_parsed(self, query: str) -> ParsedMessagePath | None:
        """Get cached parsed result if query matches.

        Args:
            query: The query string to check

        Returns:
            Cached parsed result or None if not available
        """
        if query == self._cached_query:
            return self._cached_parsed
        return None


class TreeView(Tree):
    data: reactive[MessageEvent | None] = reactive(None)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_message_string: str | None = None

    def _has_message_changed(self, message: MessageEvent) -> bool:
        """Check if the message has changed since last update."""
        new_message_string = repr(message)
        if self._current_message_string != new_message_string:
            self._current_message_string = new_message_string
            return True
        return False

    def _create_header_node(self, message: MessageEvent) -> None:
        """Create the header node."""
        timestamp = nanoseconds_to_iso(message.timestamp_ns)
        timestamp_rich = iso_highlighter(timestamp)
        schema_name = "n/a"
        if message.schema_name:
            schema_name = message.schema_name

        dummy_label = Text.assemble(
            Text.from_markup(f"[bold]{schema_name}[/bold] @ "),
            timestamp_rich,
        )
        header_node = self.root.add("")
        header_node.set_label(dummy_label)
        header_node.allow_expand = False

    def _update_tree_incrementally(self, new_message: MessageEvent) -> None:
        """Update tree incrementally by comparing each node's data."""
        # Check if we have existing structure
        if not self.root.children:
            # No existing structure - build from scratch
            self._build_tree_from_scratch(new_message)
            return

        # Update header (always check this since timestamp changes)
        self._update_header_node(new_message)

        # Update content incrementally
        if len(self.root.children) < 2:
            # Missing content node - rebuild content
            if not hasattr(new_message.message, "__slots__"):
                primitive_node = self.root.add("")
                primitive_node.data = repr(new_message.message)
                label = highlighter(repr(new_message.message))
                primitive_node.set_label(label)
                primitive_node.allow_expand = False
            else:
                add_node("root", self.root, new_message.message, 3, auto_expand=False)
        else:
            # Update existing content node
            content_node = self.root.children[1]
            self._update_node_incrementally(content_node, new_message.message, "root", 3)

    def _update_header_node(self, message: MessageEvent) -> None:
        """Update header node with new timestamp/schema."""
        if not self.root.children:
            self._create_header_node(message)
            return

        header_node = self.root.children[0]

        # Create new header content
        timestamp = nanoseconds_to_iso(message.timestamp_ns)
        schema_name = "n/a" if not message.schema_name else message.schema_name
        new_header_data = f"{schema_name}@{message.timestamp_ns}"

        # Only update if changed
        if header_node.data != new_header_data:
            header_node.data = new_header_data
            timestamp_rich = iso_highlighter(timestamp)
            dummy_label = Text.assemble(
                Text.from_markup(f"[bold]{schema_name}[/bold] @ "),
                timestamp_rich,
            )
            header_node.set_label(dummy_label)

    def _update_node_incrementally(
        self, node: TreeNode, new_obj: Any, path: str, expand_depth: int
    ) -> None:
        """Update a single node incrementally based on data comparison."""
        new_data = repr(new_obj)

        # If data hasn't changed, leave node untouched (preserves expansion state)
        if node.data == new_data:
            return

        # Data changed - update this node
        node.data = new_data

        # Update the label based on object type
        if not hasattr(new_obj, "__slots__"):
            # Primitive - just update label
            if isinstance(new_obj, bytes):
                start_bytes = new_obj[:100]
                obj_repr = f"bytes({len(new_obj)}): {start_bytes!r}..."
            else:
                obj_repr = repr(new_obj)

            # Extract name from path for label
            name = path.split("/")[-1] if "/" in path else path
            if name and name != "root":
                label = Text.assemble(Text.from_markup(f"[b]{name}[/b]="), highlighter(obj_repr))
            else:
                label = highlighter(obj_repr)
            node.set_label(label)
            node.allow_expand = False

            # Remove any children since this is now a primitive
            for child in list(node.children):
                child.remove()
            return

        # Object with slots - handle special cases first
        name = path.split("/")[-1] if "/" in path else path
        if new_obj.__slots__ == ["sec", "nanosec"]:
            timestamp = iso_highlighter(
                datetime.fromtimestamp(
                    new_obj.sec + new_obj.nanosec / NANOSECONDS_PER_SECOND, tz=timezone.utc
                ).strftime(STRFTIME_FORMAT)
            )
            if name and name != "root":
                label = Text.assemble(Text.from_markup(f"[b]{name}[/b]="), timestamp)
            else:
                label = timestamp
            node.set_label(label)
        elif new_obj.__slots__ == ["x", "y", "z", "w"]:
            roll, pitch, yaw = quaternion_to_euler(new_obj.x, new_obj.y, new_obj.z, new_obj.w)
            euler_repr = f"r={roll:.2f}, p={pitch:.2f}, y={yaw:.2f}"
            if name and name != "root":
                label = Text.assemble(
                    Text.from_markup(f"[b]{name}[/b]="),
                    highlighter(euler_repr),
                )
            else:
                label = highlighter(euler_repr)
            node.set_label(label)

        # Update children
        self._update_children_incrementally(node, new_obj, path, expand_depth)

    def _update_children_incrementally(
        self, parent_node: TreeNode, new_obj: Any, path: str, expand_depth: int
    ) -> None:
        """Update children of a node incrementally."""
        existing_children = {
            str(child.label) if child.label else "": child for child in parent_node.children
        }
        new_slots = list(new_obj.__slots__)

        # Process each slot in the new object
        for slot in new_slots:
            new_slot_data = getattr(new_obj, slot)
            child_path = f"{path}/{slot}"

            if slot in existing_children:
                # Update existing child
                child_node = existing_children[slot]

                if isinstance(new_slot_data, (list, tuple)):
                    # Handle array updates
                    self._update_array_node(
                        child_node, new_slot_data, slot, child_path, expand_depth - 1
                    )
                else:
                    # Handle object/primitive updates
                    self._update_node_incrementally(
                        child_node, new_slot_data, child_path, expand_depth - 1
                    )
            else:
                # Add new child
                child_node = parent_node.add(slot)
                if isinstance(new_slot_data, (list, tuple)):
                    child_node.data = f"array:{len(new_slot_data)}"
                    child_node.set_label(Text(f"{slot}[{len(new_slot_data)}]"))
                    for index, value in enumerate(new_slot_data):
                        item_node = child_node.add(f"[{index}]")
                        add_node(str(index), item_node, value, expand_depth - 1, auto_expand=False)
                        if index > 25:
                            truncated_node = child_node.add("Truncated")
                            truncated_node.data = "..."
                            truncated_node.set_label(Text("..."))
                            truncated_node.allow_expand = False
                            break
                else:
                    add_node(slot, child_node, new_slot_data, expand_depth - 1, auto_expand=False)

        # Remove children that no longer exist
        for child_label, child_node in existing_children.items():
            if child_label not in new_slots:
                child_node.remove()

    def _update_array_node(
        self, array_node: TreeNode, new_array: list | tuple, slot: str, path: str, expand_depth: int
    ) -> None:
        """Update an array node incrementally, comparing min(old_len, new_len) items."""
        new_array_data = f"array:{len(new_array)}"

        # Always update array label to show current length
        array_node.data = new_array_data
        array_node.set_label(Text(f"{slot}[{len(new_array)}]"))

        # Get existing array items (skip truncation markers)
        existing_items = {}
        truncation_node = None

        for child in array_node.children:
            if child.label:
                label_str = str(child.label)
                if label_str.startswith("[") and label_str.endswith("]"):
                    try:
                        index = int(label_str.strip("[]"))
                        existing_items[index] = child
                    except ValueError:
                        pass
                elif "Truncated" in label_str:
                    truncation_node = child

        # Compare up to min(existing_count, new_count, 25)
        max_items = min(len(new_array), 25)  # Respect truncation limit
        existing_count = len(existing_items)
        min_count = min(existing_count, max_items)

        # Update existing items up to min_count
        for i in range(min_count):
            if i in existing_items:
                existing_node = existing_items[i]
                self._update_node_incrementally(
                    existing_node, new_array[i], f"{path}/{i}", expand_depth
                )

        # Add new items if array grew
        for i in range(existing_count, max_items):
            item_node = array_node.add(f"[{i}]")
            add_node(str(i), item_node, new_array[i], expand_depth, auto_expand=False)

        # Remove excess items if array shrunk
        for i in range(max_items, existing_count):
            if i in existing_items:
                existing_items[i].remove()

        # Handle truncation marker
        if len(new_array) > 25:
            if not truncation_node:
                # Add truncation marker
                truncation_node = array_node.add("Truncated")
                truncation_node.data = "..."
                truncation_node.set_label(Text("..."))
                truncation_node.allow_expand = False
        elif truncation_node:
            # Remove truncation marker
            truncation_node.remove()

    def _build_tree_from_scratch(self, message: MessageEvent) -> None:
        """Build tree from scratch when no existing structure."""
        self.clear()

        # Create header
        self._create_header_node(message)

        # Create content
        if not hasattr(message.message, "__slots__"):
            primitive_node = self.root.add("")
            primitive_node.data = repr(message.message)
            label = highlighter(repr(message.message))
            primitive_node.set_label(label)
            primitive_node.allow_expand = False
        else:
            add_node("root", self.root, message.message, 3, auto_expand=False)

    def watch_data(self, channel_message: MessageEvent | None) -> None:
        """Updates tree with per-node string comparison to preserve user expansion state.

        Args:
            channel_message: MessageEvent containing schema, timestamp, and message data.
        """
        if not channel_message:
            self.clear()
            self._current_message_string = None
            return

        # Check if entire message changed (quick check)
        if not self._has_message_changed(channel_message):
            return  # No change, preserve current tree state including expansion

        # Do incremental update - only changes what actually changed
        self._update_tree_incrementally(channel_message)


class Raw(BasePanel):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {SCHEMA_ANY}
    PRIORITY: ClassVar[int] = 1000  # Should be the last panel

    parsed_query: reactive[ParsedMessagePath | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self._validator = QueryValidator()

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        yield Input(placeholder="Filter", compact=True, validators=[self._validator])
        yield TreeView("root").data_bind(guide_depth=2, show_root=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle query changes from Input."""
        if event.validation_result and event.validation_result.is_valid:
            self.parsed_query = self._validator.get_cached_parsed(event.value)
            self._update_date()

    def _update_date(self) -> None:
        tree = self.query_one(TreeView)

        if self.parsed_query and self.data and self.parsed_query:
            try:
                filtered_message = apply_query(self.parsed_query, self.data.message)
                tree.data = MessageEvent(
                    topic=self.data.topic,
                    message=filtered_message,
                    timestamp_ns=self.data.timestamp_ns,
                    schema_name=self.data.schema_name,
                )
            except QueryError:
                # Ignore for now
                pass
        else:
            tree.data = self.data

    def watch_data(self, _data: MessageEvent | None) -> None:
        self._update_date()
