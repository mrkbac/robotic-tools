"""Tests for message path query execution.

Tests the parse_query functionality that applies parsed message paths
to Python objects with __slots__ or primitive types.
"""

from dataclasses import dataclass

import pytest

from digitalis.grammar.parser import parse_message_path
from digitalis.grammar.query import QueryError, apply_query, parse_query


# Test data structures using __slots__
class Vector3:
    """3D vector with __slots__."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class Pose:
    """Pose with position and orientation using __slots__."""

    __slots__ = ("orientation", "position")

    def __init__(self, position: Vector3, orientation: Vector3) -> None:
        self.position = position
        self.orientation = orientation


@dataclass
class Marker:
    """Marker dataclass for testing."""

    id: int
    type: str
    pose: Pose
    active: bool
    scale: float


@dataclass
class TrajectoryPoint:
    """Trajectory point dataclass."""

    position: Vector3
    velocity: Vector3
    time: float


class Trajectory:
    """Trajectory with list of points using __slots__."""

    __slots__ = ("name", "points")

    def __init__(self, points: list[TrajectoryPoint], name: str) -> None:
        self.points = points
        self.name = name


class RobotState:
    """Complete robot state using __slots__."""

    __slots__ = ("active", "markers", "pose", "trajectory")

    def __init__(
        self, pose: Pose, trajectory: Trajectory, markers: list[Marker], active: bool
    ) -> None:
        self.pose = pose
        self.trajectory = trajectory
        self.markers = markers
        self.active = active


# Test fixtures
@pytest.fixture
def sample_vector() -> Vector3:
    """Create a sample Vector3."""
    return Vector3(1.0, 2.0, 3.0)


@pytest.fixture
def sample_pose() -> Pose:
    """Create a sample Pose."""
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Vector3(0.0, 0.0, 1.57)
    return Pose(position, orientation)


@pytest.fixture
def sample_markers() -> list[Marker]:
    """Create sample markers."""
    return [
        Marker(
            id=1,
            type="arrow",
            pose=Pose(Vector3(1.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)),
            active=True,
            scale=1.0,
        ),
        Marker(
            id=2,
            type="cube",
            pose=Pose(Vector3(2.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.57)),
            active=False,
            scale=0.5,
        ),
        Marker(
            id=3,
            type="arrow",
            pose=Pose(Vector3(0.0, 2.0, 1.0), Vector3(0.0, 0.0, 3.14)),
            active=True,
            scale=2.0,
        ),
    ]


@pytest.fixture
def sample_trajectory() -> Trajectory:
    """Create a sample trajectory."""
    points = [
        TrajectoryPoint(
            position=Vector3(0.0, 0.0, 0.0),
            velocity=Vector3(1.0, 0.0, 0.0),
            time=0.0,
        ),
        TrajectoryPoint(
            position=Vector3(1.0, 0.0, 0.0),
            velocity=Vector3(1.0, 1.0, 0.0),
            time=1.0,
        ),
        TrajectoryPoint(
            position=Vector3(2.0, 1.0, 0.0),
            velocity=Vector3(0.0, 1.0, 0.0),
            time=2.0,
        ),
    ]
    return Trajectory(points, "test_trajectory")


@pytest.fixture
def robot_state(
    sample_pose: Pose, sample_trajectory: Trajectory, sample_markers: list[Marker]
) -> RobotState:
    """Create a complete robot state."""
    return RobotState(sample_pose, sample_trajectory, sample_markers, True)


class TestBasicFieldAccess:
    """Test basic field access on objects."""

    def test_simple_field_access(self, sample_vector: Vector3) -> None:
        """Test accessing a simple field."""
        # This test will fail until we implement parse_query

        result = parse_query("x", sample_vector)
        assert result == 1.0

        result = parse_query("y", sample_vector)
        assert result == 2.0

        result = parse_query("z", sample_vector)
        assert result == 3.0

    def test_nested_field_access(self, sample_pose: Pose) -> None:
        """Test accessing nested fields."""

        result = parse_query("position.x", sample_pose)
        assert result == 1.0

        result = parse_query("position.y", sample_pose)
        assert result == 2.0

        result = parse_query("orientation.z", sample_pose)
        assert result == 1.57

    def test_deep_nested_access(self, robot_state: RobotState) -> None:
        """Test deeply nested field access."""

        result = parse_query("pose.position.x", robot_state)
        assert result == 1.0

        result = parse_query("trajectory.name", robot_state)
        assert result == "test_trajectory"

        result = parse_query("active", robot_state)
        assert result is True


class TestArrayAccess:
    """Test array indexing operations."""

    def test_simple_array_index(self, robot_state: RobotState) -> None:
        """Test simple array indexing."""

        # Access first marker
        result = parse_query("markers[0].id", robot_state)
        assert result == 1

        # Access second marker
        result = parse_query("markers[1].type", robot_state)
        assert result == "cube"

        # Access trajectory point
        result = parse_query("trajectory.points[0].time", robot_state)
        assert result == 0.0

    def test_negative_array_index(self, robot_state: RobotState) -> None:
        """Test negative array indexing."""

        # Access last marker
        result = parse_query("markers[-1].id", robot_state)
        assert result == 3

        # Access last trajectory point
        result = parse_query("trajectory.points[-1].time", robot_state)
        assert result == 2.0

    def test_array_index_out_of_bounds(self, robot_state: RobotState) -> None:
        """Test array index out of bounds handling."""

        with pytest.raises(QueryError):
            parse_query("markers[10].id", robot_state)

        with pytest.raises(QueryError):
            parse_query("trajectory.points[-10].time", robot_state)


class TestSliceAccess:
    """Test slice notation operations."""

    def test_simple_slice(self, robot_state: RobotState) -> None:
        """Test simple slice operations."""

        # Get first two markers
        result = parse_query("markers[0:2]", robot_state)
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

        # Get trajectory points slice
        result = parse_query("trajectory.points[1:3]", robot_state)
        assert len(result) == 2
        assert result[0].time == 1.0
        assert result[1].time == 2.0

    def test_open_ended_slices(self, robot_state: RobotState) -> None:
        """Test open-ended slice operations."""

        # Get all markers from index 1
        result = parse_query("markers[1:]", robot_state)
        assert len(result) == 2
        assert result[0].id == 2
        assert result[1].id == 3

        # Get first two markers
        result = parse_query("markers[:2]", robot_state)
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_full_slice(self, robot_state: RobotState) -> None:
        """Test full slice operation."""

        result = parse_query("markers[:]", robot_state)
        assert len(result) == 3
        assert [m.id for m in result] == [1, 2, 3]


class TestFilterOperations:
    """Test filtering operations."""

    def test_equality_filter(self, robot_state: RobotState) -> None:
        """Test equality filtering."""

        # Filter markers by type
        result = parse_query('markers[:]{type=="arrow"}', robot_state)
        assert len(result) == 2
        assert all(m.type == "arrow" for m in result)
        assert [m.id for m in result] == [1, 3]

        # Filter by boolean value
        result = parse_query("markers[:]{active==true}", robot_state)
        assert len(result) == 2
        assert all(m.active for m in result)

    def test_comparison_filters(self, robot_state: RobotState) -> None:
        """Test comparison filtering operations."""

        # Filter by numeric value
        result = parse_query("markers[:]{scale>0.5}", robot_state)
        assert len(result) == 2
        assert all(m.scale > 0.5 for m in result)

        # Filter trajectory points by time
        result = parse_query("trajectory.points[:]{time>=1.0}", robot_state)
        assert len(result) == 2
        assert all(p.time >= 1.0 for p in result)

    def test_nested_field_filter(self, robot_state: RobotState) -> None:
        """Test filtering on nested fields."""

        # Filter markers by nested pose position
        result = parse_query("markers[:]{pose.position.x>1.0}", robot_state)
        assert len(result) == 1
        assert result[0].id == 2

        # Filter trajectory points by position
        result = parse_query("trajectory.points[:]{position.x==1.0}", robot_state)
        assert len(result) == 1
        assert result[0].time == 1.0

    def test_multiple_filters(self, robot_state: RobotState) -> None:
        """Test applying multiple filters."""

        # Chain multiple filters
        result = parse_query('markers[:]{type=="arrow"}{active==true}', robot_state)
        assert len(result) == 2
        assert all(m.type == "arrow" and m.active for m in result)


class TestVariableSubstitution:
    """Test variable substitution in queries."""

    def test_variable_in_slice(self, robot_state: RobotState) -> None:
        """Test variable substitution in slice operations."""

        parsed = parse_message_path("markers[$start:$end]")
        result = apply_query(parsed, robot_state, variables={"start": 0, "end": 2})
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    def test_variable_in_filter(self, robot_state: RobotState) -> None:
        """Test variable substitution in filter operations."""

        parsed = parse_message_path("markers[:]{id==$target_id}")
        result = apply_query(parsed, robot_state, variables={"target_id": 2})
        assert len(result) == 1
        assert result[0].id == 2

    def test_variable_in_array_index(self, robot_state: RobotState) -> None:
        """Test variable substitution in array indexing."""

        parsed = parse_message_path("markers[$index].type")
        result = apply_query(parsed, robot_state, variables={"index": 1})
        assert result == "cube"


class TestErrorHandling:
    """Test error handling for various edge cases."""

    def test_missing_field(self, sample_vector: Vector3) -> None:
        """Test handling of missing fields."""

        with pytest.raises(QueryError, match="Field 'w' not found"):
            parse_query("w", sample_vector)

    def test_invalid_array_access(self, sample_vector: Vector3) -> None:
        """Test invalid array access on non-sequence objects."""

        with pytest.raises(QueryError, match="Cannot index"):
            parse_query("x[0]", sample_vector)

    def test_empty_object(self) -> None:
        """Test querying empty objects."""

        with pytest.raises(QueryError):
            parse_query("field", None)

    def test_type_mismatch_in_filter(self, robot_state: RobotState) -> None:
        """Test type mismatch in filter comparisons."""

        # Try to compare string with number
        with pytest.raises(QueryError):
            parse_query("markers[:]{type>5}", robot_state)


class TestComplexQueries:
    """Test complex query combinations."""

    def test_chained_operations(self, robot_state: RobotState) -> None:
        """Test chained field access, indexing, and filtering."""

        # Complex query: get x position of first active arrow marker
        result = parse_query(
            'markers[:]{type=="arrow"}{active==true}[0].pose.position.x', robot_state
        )
        assert result == 1.0

    def test_nested_slicing_and_filtering(self, robot_state: RobotState) -> None:
        """Test nested slicing and filtering operations."""

        # Get time values of middle trajectory points
        result = parse_query("trajectory.points[1:3][:].time", robot_state)
        assert result == [1.0, 2.0]


class TestPrimitiveTypes:
    """Test querying primitive types and collections."""

    def test_dict_access(self) -> None:
        """Test querying dictionary objects."""

        data = {"position": {"x": 1.0, "y": 2.0}, "values": [1, 2, 3]}

        result = parse_query("position.x", data)
        assert result == 1.0

        result = parse_query("values[1]", data)
        assert result == 2

    def test_list_of_primitives(self) -> None:
        """Test querying lists of primitive values."""

        data = [1, 2, 3, 4, 5]

        result = parse_query("[0]", data)
        assert result == 1

        result = parse_query("[1:4]", data)
        assert result == [2, 3, 4]

    def test_mixed_data_types(self) -> None:
        """Test querying mixed data type structures."""

        data = {
            "numbers": [1, 2, 3],
            "objects": [{"id": 1, "value": 10}, {"id": 2, "value": 20}],
            "nested": {"deep": {"value": 42}},
        }

        result = parse_query("numbers[1]", data)
        assert result == 2

        result = parse_query("objects[:]{id==2}[0].value", data)
        assert result == 20

        result = parse_query("nested.deep.value", data)
        assert result == 42
