import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from rich.highlighter import ISO8601Highlighter, ReprHighlighter
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from digitalis.reader.types import MessageEvent
from digitalis.ui.panels.base import BasePanel
from digitalis.utilities import nanoseconds_to_iso, quaternion_to_euler


class Vector3(Protocol):
    """https://docs.ros2.org/foxy/api/geometry_msgs/msg/Vector3.html"""

    x: float
    y: float
    z: float


class Quaternion(Protocol):
    """https://docs.ros2.org/foxy/api/geometry_msgs/msg/Quaternion.html"""

    x: float
    y: float
    z: float
    w: float


class Transform(Protocol):
    """https://docs.ros2.org/foxy/api/geometry_msgs/msg/Transform.html"""

    translation: Vector3
    rotation: Quaternion


class TfTransformStamped(Protocol):
    """https://docs.ros2.org/foxy/api/geometry_msgs/msg/TransformStamped.html"""

    header: Any
    child_frame_id: str
    transform: Transform


class TFMessage(Protocol):
    """https://docs.ros2.org/foxy/api/tf2_msgs/msg/TFMessage.html"""

    transforms: list[TfTransformStamped]


@dataclass
class TransformData:
    frame_id: str
    child_frame_id: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    static: bool
    timestamp_ns: int


@dataclass
class TfTreeNode:
    frame_id: str
    level: int
    transform_data: TransformData | None = None


class TfTree(Tree[str]):
    data: reactive[dict[str, TransformData]] = reactive({})

    def __init__(self) -> None:
        super().__init__("root")
        self.show_root = False

    def watch_data(self, data: dict[str, TransformData]) -> None:
        # Build hierarchical tree structure
        tree_nodes = self._build_tree_structure(data)

        # Update tree efficiently
        self._update_tree_nodes(tree_nodes)

    def _build_tree_structure(self, data: dict[str, TransformData]) -> list[TfTreeNode]:
        nodes: list[TfTreeNode] = []

        if not data:
            return nodes

        # Find root frames (parents that are not children)
        parents = {t.frame_id for t in data.values()}
        children = {t.child_frame_id for t in data.values()}
        root_frames = sorted(parents - children)

        # Build tree starting from root frames
        for root_frame in root_frames:
            nodes.append(TfTreeNode(frame_id=root_frame, level=0))
            nodes.extend(self._build_subtree(root_frame, data, 1))

        return nodes

    def _build_subtree(
        self, frame_id: str, data: dict[str, TransformData], level: int
    ) -> list[TfTreeNode]:
        nodes: list[TfTreeNode] = []

        # Find all transforms where this frame is the parent
        child_transforms = [t for t in data.values() if t.frame_id == frame_id]
        child_transforms.sort(key=lambda t: t.child_frame_id)

        for transform in child_transforms:
            nodes.append(
                TfTreeNode(frame_id=transform.child_frame_id, level=level, transform_data=transform)
            )
            nodes.extend(self._build_subtree(transform.child_frame_id, data, level + 1))

        return nodes

    def _update_tree_nodes(self, tree_nodes: list[TfTreeNode]) -> None:
        # Build expected root structure
        expected_roots = {}
        for node in tree_nodes:
            if node.level == 0:
                has_children = any(
                    n.transform_data and n.transform_data.frame_id == node.frame_id
                    for n in tree_nodes
                )
                expected_roots[node.frame_id] = (Text(node.frame_id, style="bold"), has_children)

        # Get current root nodes
        current_roots = {child.data: child for child in self.root.children if child.data}
        current_keys = list(current_roots.keys())
        expected_keys = list(expected_roots.keys())

        # Remove stale roots that no longer exist
        stale_keys = set(current_keys) - set(expected_keys)
        for stale_key in stale_keys:
            current_roots[stale_key].remove()

        # Update or add roots
        for frame_id, (label, has_children) in expected_roots.items():
            if frame_id in current_roots:
                # Update existing root
                root_node = current_roots[frame_id]
                if str(root_node.label) != str(label):
                    root_node.set_label(label)
            elif has_children:
                # Add new root with children
                root_node = self.root.add(label, data=frame_id)
            else:
                # Add new leaf root
                root_node = self.root.add_leaf(label, data=frame_id)

            # Update children and expand if needed
            if has_children:
                root_node.expand()
                self._update_children(root_node, frame_id, tree_nodes)

    def _update_children(
        self, parent_tree_node: TreeNode[str], parent_frame_id: str, all_nodes: list[TfTreeNode]
    ) -> None:
        # Build expected children structure
        expected_children = {}
        child_nodes = [
            node
            for node in all_nodes
            if node.transform_data and node.transform_data.frame_id == parent_frame_id
        ]
        child_nodes.sort(key=lambda n: n.frame_id)

        for child_node in child_nodes:
            expected_children[child_node.frame_id] = self._create_node_label(child_node)

        # Get current children
        current_children = {child.data: child for child in parent_tree_node.children if child.data}
        current_keys = list(current_children.keys())
        expected_keys = list(expected_children.keys())

        # Remove stale children
        stale_keys = set(current_keys) - set(expected_keys)
        for stale_key in stale_keys:
            current_children[stale_key].remove()

        # Update or add children
        for frame_id, label in expected_children.items():
            if frame_id in current_children:
                # Update existing child
                tree_child = current_children[frame_id]
                if str(tree_child.label) != str(label):
                    tree_child.set_label(label)
            else:
                # Add new child - always expandable since we add transform details
                tree_child = parent_tree_node.add(label, data=frame_id)

            # Find the corresponding tf_node and update its children
            child_tf_node = next((n for n in child_nodes if n.frame_id == frame_id), None)
            if child_tf_node:
                self._update_transform_children(tree_child, child_tf_node, all_nodes)
                tree_child.expand()

    def _update_transform_children(
        self, parent_node: TreeNode[str], tf_node: TfTreeNode, all_nodes: list[TfTreeNode]
    ) -> None:
        """Update all children of a transform node using efficient in-place updates."""
        if not tf_node.transform_data:
            return

        highlighter = ReprHighlighter()
        transform = tf_node.transform_data

        # Build expected children - transform details first, then frame children
        expected_children = {}

        # Add transform details
        x, y, z = transform.translation
        translation_label = Text.assemble(
            Text.from_markup("[b]translation[/b]="), highlighter(f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
        )
        expected_children["__translation__"] = translation_label

        qx, qy, qz, qw = transform.rotation
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        rotation_text = (
            f"r={math.degrees(roll):.1f}°, p={math.degrees(pitch):.1f}°, y={math.degrees(yaw):.1f}°"
        )
        rotation_label = Text.assemble(
            Text.from_markup("[b]rotation[/b]="),
            highlighter(rotation_text),
        )
        expected_children["__rotation__"] = rotation_label

        # Add frame children
        frame_children = [
            node
            for node in all_nodes
            if node.transform_data and node.transform_data.frame_id == tf_node.frame_id
        ]
        frame_children.sort(key=lambda n: n.frame_id)

        for child_node in frame_children:
            expected_children[child_node.frame_id] = self._create_node_label(child_node)

        # Get current children by their stable keys
        current_children = {child.data: child for child in parent_node.children if child.data}
        current_keys = list(current_children.keys())
        expected_keys = list(expected_children.keys())

        # If structure matches, just update labels (efficient in-place update)
        if current_keys == expected_keys:
            for key, label in expected_children.items():
                if str(current_children[key].label) != str(label):
                    current_children[key].set_label(label)
            # Recursively update frame children
            for child_node in frame_children:
                if child_node.frame_id in current_children:
                    child_tree_node = current_children[child_node.frame_id]
                    self._update_transform_children(child_tree_node, child_node, all_nodes)
        else:
            # Structure changed, rebuild children
            parent_node.remove_children()
            for key, label in expected_children.items():
                if key.startswith("__") and key.endswith("__"):
                    # Transform detail - add as leaf
                    parent_node.add_leaf(label, data=key)
                else:
                    # Frame child - add as expandable node
                    child_tree_node = parent_node.add(label, data=key)
                    # Find the corresponding tf_node and recursively update its children
                    child_tf_node = next((n for n in frame_children if n.frame_id == key), None)
                    if child_tf_node:
                        self._update_transform_children(child_tree_node, child_tf_node, all_nodes)
                        child_tree_node.expand()

    def _create_node_label(self, node: TfTreeNode) -> Text:
        if not node.transform_data:
            return Text(node.frame_id, style="bold")

        transform = node.transform_data
        color = "green" if transform.static else "red"

        # Format timestamp like raw panel
        iso_highlighter = ISO8601Highlighter()
        timestamp = nanoseconds_to_iso(transform.timestamp_ns)
        timestamp_text = iso_highlighter(timestamp)

        return Text.assemble((f"{node.frame_id}", color), (" @ ", ""), timestamp_text)


class Tf(BasePanel[TFMessage]):
    SUPPORTED_SCHEMAS: ClassVar[set[str]] = {
        "tf2_msgs/msg/TFMessage",  # ROS2
        "tf2_msgs/TFMessage",  # ROS1
    }

    tf_data: reactive[dict[str, TransformData]] = reactive({})
    _last_channel_id: str | None = None
    _pending_update: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.set_interval(0.05, self._flush_updates)  # Update UI at 20Hz

    def compose(self) -> ComposeResult:
        yield TfTree()

    def _update_tree(self) -> None:
        tree = self.query_one(TfTree)
        tree.data = self.tf_data
        tree.mutate_reactive(TfTree.data)

    def _flush_updates(self) -> None:
        """Periodically flush pending updates to the UI (throttled at 20Hz)."""
        if self._pending_update:
            self._update_tree()
            self._pending_update = False

    def watch_data(self, data: MessageEvent | None) -> None:
        if data is None:
            return

        # Clear data if channel changed
        if data.topic != self._last_channel_id:
            self.tf_data.clear()
            self._last_channel_id = data.topic

        # TODO: Subscribe to both topics and show correctly
        is_static = False

        # Process each transform in the message
        for transform_stamped in data.message.transforms:
            key = f"{transform_stamped.header.frame_id}_{transform_stamped.child_frame_id}"

            trans = transform_stamped.transform.translation
            rot = transform_stamped.transform.rotation

            self.tf_data[key] = TransformData(
                frame_id=transform_stamped.header.frame_id,
                child_frame_id=transform_stamped.child_frame_id,
                translation=(trans.x, trans.y, trans.z),
                rotation=(rot.x, rot.y, rot.z, rot.w),
                static=is_static,
                timestamp_ns=data.timestamp_ns,
            )

        # Mark that we need to update UI (will happen in next flush)
        self._pending_update = True
