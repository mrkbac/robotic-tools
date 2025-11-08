# Foxglove Message Path Syntax Reference

This document describes the message path syntax used by Foxglove for accessing and filtering data from ROS topics.

## Overview

Message paths provide a concise way to navigate nested message structures, access array elements, and filter data based on conditions. The syntax is inspired by JavaScript object notation with additional features for filtering and slicing.

## Syntax Components

### 1. Topic Reference

Every message path starts with a topic name prefixed with `/`:

```
/my_topic
```

### 2. Field Access

Use dot notation to access nested fields:

```
/my_topic.field_name
/my_topic.nested.field.value
```

### 3. Array Indexing

Access specific array elements using bracket notation with zero-based indices:

```
/my_topic.array[0]           # First element
/my_topic.array[5]           # Sixth element
/my_topic.array[-1]          # Last element
/my_topic.array[-2]          # Second to last element
```

**Negative indices** count from the end of the array.

### 4. Array Slicing

Extract ranges of elements using colon syntax `[start:end]`:

```
/my_topic.array[1:3]         # Elements at indices 1, 2, and 3 (inclusive)
/my_topic.array[:]           # All elements
/my_topic.colors[:].r        # Red channel of all colors
```

**Important**: Both start and end indices are **inclusive**.

### 5. Filtering

Filter messages using curly braces with comparison expressions:

```
/my_topic{field == value}
/my_topic{stats.count > 100}
/my_topic.items[:]{active == true}
```

#### Comparison Operators

| Operator | Description           | Example                |
| -------- | --------------------- | ---------------------- |
| `==`     | Equal to              | `{id == 5}`            |
| `!=`     | Not equal to          | `{status != "error"}`  |
| `<`      | Less than             | `{temperature < 25.5}` |
| `<=`     | Less than or equal    | `{count <= 10}`        |
| `>`      | Greater than          | `{pressure > 101.3}`   |
| `>=`     | Greater than or equal | `{score >= 0.95}`      |

#### Multiple Filters (AND Logic)

Chain multiple filters together - all conditions must be true:

```
/my_topic{category == "robot"}{status == "active"}{battery > 20}
```

This is equivalent to: `category == "robot" AND status == "active" AND battery > 20`

### 6. Variables

Variables are prefixed with `$` and can be used in slices and filters:

```
/my_topic.array[$start_idx:$end_idx]
/my_topic.items[:]{id == $selected_id}
/my_topic{timestamp > $min_time}{timestamp < $max_time}
```

**Restrictions**:

- Variables can **only** appear in array slices and filter expressions
- Variables cannot be used in topic names or field names

### 7. Value Literals

Filters support different literal types:

- **Numbers**: `42`, `-3.14`, `0.001`
- **Strings**: `"hello"` or `'world'`
- **Booleans**: `true`, `false`

**String Quote Escaping**: Escape quotes by alternating single and double quotes:

```
/my_topic{name == "O'Brien"}      # Single quote inside double quotes
/my_topic{text == 'Say "hello"'}  # Double quotes inside single quotes
```

## Complete Examples

### Basic Field Access

```
/odom.pose.pose.position.x
```

Access the x-coordinate from an odometry message.

### Hierarchical Topic Names

ROS topics can have hierarchical names with multiple levels separated by slashes:

```
/vehicle/odom
/camera/left/image_color
/robot/sensors/lidar/front
```

These hierarchical topics work with all path operations:

```
/vehicle/odom.pose.position.x
/camera/left/images[0].data
/sensors/temp[:]{value>20}.reading
```

### Array Navigation

```
/sensor_data.readings[0].value
/sensor_data.readings[-1].timestamp
```

Get the first reading's value and the last reading's timestamp.

### Filtering Arrays

```
/detections.objects[:]{confidence > 0.8}.class_name
```

Get class names of all detected objects with confidence above 80%.

### Complex Nested Operations

```
/library.books[:]{genre == "sci-fi"}{pages > 200}.reviews[:]{rating >= 4}.text
```

Get review text (rating ≥ 4) for sci-fi books with more than 200 pages.

### Using Variables

```
/robots.agents[:]{id == $robot_id}.battery_level
/timeline.events[$start:$end].description
```

Filter by a specific robot ID and slice events within a time range.

### Multiple Conditions

```
/warehouse.items[:]{quantity < $min_stock}{category == $selected_category}.name
```

Find low-stock items in a specific category.

## Grammar Overview

```
message_path    : topic_ref (path_segment)*

topic_ref       : "/" identifier

path_segment    : field_access
                | array_index
                | array_slice
                | filter

field_access    : "." identifier

array_index     : "[" (integer | negative_int | variable) "]"

array_slice     : "[" slice_start? ":" slice_end? "]"

filter          : "{" field_expr operator value "}"

operator        : "==" | "!=" | "<" | "<=" | ">" | ">="

value           : number | string | boolean | variable

variable        : "$" identifier
```

## Special Rules and Edge Cases

### 1. Slice Inclusivity

Unlike Python slicing, both start and end indices are **inclusive**:

- `[1:3]` returns elements at indices 1, 2, **and** 3
- `[0:0]` returns just the first element

### 2. Empty Slices

- `[:]` is valid and returns all elements
- `[:5]` returns elements from 0 to 5 (inclusive)
- `[5:]` returns elements from 5 to the end

### 3. Filter Field Paths

Filter expressions can reference nested fields:

```
/data{stats.metrics.cpu_usage > 80}
```

### 4. Filter on Array Results

Filters can be applied after slicing:

```
/sensors[:]{type == "temperature"}.readings[:]{value > 25}
```

### 5. Variable Scope

Variables are resolved from an external context and must be provided by the system using the message path.

### 6. Quote Escaping Limitation

There is no escape sequence like `\"`. Instead, alternate quote styles:

- Use `"` around strings containing `'`
- Use `'` around strings containing `"`

## Implementation Notes

This parser will:

- Parse message paths into an AST (Abstract Syntax Tree)
- Validate syntax according to the grammar rules
- Provide structured representation for evaluation engines
- Support all operators and features defined in the Foxglove specification

## References

- [Foxglove Message Path Syntax Documentation](https://docs.foxglove.dev/docs/visualization/message-path-syntax)
