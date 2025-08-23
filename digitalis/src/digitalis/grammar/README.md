# Message Path Grammar

A parser for Foxglove/ROS message path expressions that enables navigation and filtering of nested data structures.

## Grammar Specification

### Basic Syntax

The message path grammar supports the following core constructs:

1. **Topic Reference**: `/topic_name` - References a ROS topic
2. **Field Access**: `/topic.field` - Access nested fields using dot notation
3. **Array Indexing**: `/topic.array[0]` - Access array elements by index
4. **Negative Indexing**: `/topic.array[-1]` - Access from end of array
5. **Slicing**: `/topic.array[1:5]` - Extract array slices with start:end notation
6. **Filtering**: `/topic{field==value}` - Filter data based on conditions

### Operators

- **Field Access**: `.` - Navigate to nested fields
- **Array Access**: `[index]` - Access array elements or slices
- **Filtering**: `{condition}` - Apply filters to data
- **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=` - Compare values in filters
- **Variables**: `$variable` - Reference variables in slices and filters

### Data Types

- **Integers**: `42`, `-10`
- **Strings**: `"text"`, `'text'`
- **Booleans**: `true`, `false`
- **Variables**: `$var_name`

### Examples

```
/pose                           # Simple topic reference
/pose.position.x               # Nested field access
/trajectory.points[0]          # Array indexing
/trajectory.points[-1]         # Last element
/trajectory.points[1:5]        # Slice elements 1-4
/trajectory.points[:]          # All elements
/markers{id==5}                # Filter by condition
/markers[:]{type=="arrow"}     # Filter all elements
/data.values[$start:$end]      # Variable-based slicing
```

### Grammar Structure

The parser returns structured data with the following schema:

```python
{
    "topicName": str,        # The topic name
    "topicNameRepr": str,    # String representation of topic
    "messagePath": list,     # Parsed path components
    "modifier": str | None   # Optional modifier
}
```

## Implementation

This grammar is implemented using the Lark parsing library in Python, providing:

- Fast parsing performance
- Type-safe output structures
- Comprehensive error handling
- Support for all Foxglove message path features

## Resources

- [Foxglove Message path syntax](https://docs.foxglove.dev/docs/visualization/message-path-syntax)
- [LichtblickSuite grammar.ne](https://github.com/lichtblick-suite/lichtblick/blob/main/packages/message-path/src/grammar.ne)
- [LichtblickSuite grammar tests](https://github.com/lichtblick-suite/lichtblick/blob/main/packages/message-path/src/parseMessagePath.test.ts)
