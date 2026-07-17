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

As a `ros-parser` extension, filters can also compare the current scalar value
directly by starting the expression with an operator. This is useful after
selecting a primitive field or calculating a derived value with a modifier:

```
/temperature{>=-40 && <=125}
/wheel_speed{!=0}
/imu.linear_acceleration.@norm{<=30}
/samples[:]{<=30}
/status{in [1, 3, $selected_status]}
```

For a scalar, a matching filter returns the original value and a non-matching
filter returns no value. For a primitive array, the comparison is applied to
each element. The right side of a current-value comparison may be a literal or
`$variable`, but not a field reference; use a named-field filter when comparing
fields on an object. Current-value membership uses the same literal and variable
values with `{in [...]}`.

#### Comparison Operators

| Operator | Description           | Example                |
| -------- | --------------------- | ---------------------- |
| `==`     | Equal to              | `{id == 5}`            |
| `!=`     | Not equal to          | `{status != "error"}`  |
| `<`      | Less than             | `{temperature < 25.5}` |
| `<=`     | Less than or equal    | `{count <= 10}`        |
| `>`      | Greater than          | `{pressure > 101.3}`   |
| `>=`     | Greater than or equal | `{score >= 0.95}`      |

#### Boolean Logic

Combine conditions with `&&` (AND), `||` (OR), and `!` (NOT):

```
/detections.objects[:]{confidence > 0.8 || class == "person"}
/sensors[:]{active == true && value > 0}
/data{!(type == "error")}
```

Use parentheses to control precedence (`&&` binds tighter than `||`):

```
/topic{(x > 1 || y > 1) && z == 0}
/alerts{!(severity == "low") && active == true}
```

#### Cross-Field Comparison

Compare one field against another field on the same object:

```
/odom{position.x > target.x}
/readings[:]{measured > expected}
```

#### Enum Names

For ROS fields with constants, an unquoted constant name resolves to its schema
value after validation:

```
/vehicle_state{status == MOVING}
/vehicle_state{status in [STOPPED, MOVING]}
```

Enum names are case-sensitive. A bare name that resolves to another field keeps
the cross-field behavior described above.

#### `in` Membership Operator

Test if a field's value is one of a set of values:

```
/markers[:]{type in [1, 3, 5]}
/sensors[:]{status in ["active", "calibrating"]}
```

#### Multiple Filters (AND Logic via Chaining)

Chain multiple filters together - all conditions must be true:

```
/my_topic{category == "robot"}{status == "active"}{battery > 20}
```

This is equivalent to: `category == "robot" AND status == "active" AND battery > 20`

### 6. Math Modifiers (`.@`)

Math modifiers transform numeric values using the `.@` syntax. They can be chained and support optional arguments.

```
/topic.value.@abs                  # No arguments
/topic.value.@mul(2.5)             # Single argument
/topic.value.@add($offset)         # Variable argument
/topic.value.@mul(1.8).@add(32)    # Chaining (Celsius to Fahrenheit)
/topic.value.@add(5, 10, 3)        # Multiple arguments
/topic.@product(width, height)     # Fields relative to the input object
```

Object modifiers can take bare field paths such as `width` or `shape.height`.
They are resolved relative to the object entering the modifier. A leading `$`
continues to identify a variable, so `.@product(width, $scale)` multiplies a
message field by a caller-provided value.

**Element-wise arrays**: When applied to an array, modifiers operate on each element automatically:

```
/topic.values[:].@mul(2)           # Doubles every element
/topic.readings[:].@abs            # Absolute value of each reading
```

#### Scalar Math Functions

These operate on numeric values (int/float). When applied to arrays, they work element-wise.

| Modifier | Arguments | Description | Example |
| --- | --- | --- | --- |
| `.@abs` | none | Absolute value | `/topic.value.@abs` |
| `.@negative` | none | Negate value | `/topic.value.@negative` |
| `.@sign` | none | Sign: 1, -1, or 0 | `/topic.value.@sign` |
| `.@ceil` | none | Round up to integer | `/topic.value.@ceil` |
| `.@floor` | none | Round down to integer | `/topic.value.@floor` |
| `.@trunc` | none | Truncate to integer | `/topic.value.@trunc` |
| `.@round` | `(precision?)` | Round, optional decimal places | `.@round` or `.@round(2)` |
| `.@sqrt` | none | Square root | `/topic.value.@sqrt` |
| `.@log` | none | Natural logarithm | `/topic.value.@log` |
| `.@log1p` | none | ln(1 + x), accurate for small x | `/topic.value.@log1p` |
| `.@log2` | none | Base-2 logarithm | `/topic.value.@log2` |
| `.@log10` | none | Base-10 logarithm | `/topic.value.@log10` |

#### Arithmetic Functions

| Modifier | Arguments | Description | Example |
| --- | --- | --- | --- |
| `.@add` | `(a, b?, ...)` | Add values | `.@add(10)` or `.@add(5, 10, 3)` |
| `.@sub` | `(a, b?, ...)` | Subtract values | `.@sub(5)` |
| `.@mul` | `(a, b?, ...)` | Multiply by values | `.@mul(2)` or `.@mul(2, 3)` |
| `.@div` | `(divisor)` | Divide (errors on zero) | `.@div(100)` |
| `.@min` | `(a, b?, ...)` | Minimum of value and args | `.@min(5, 2, 8)` |
| `.@max` | `(a, b?, ...)` | Maximum of value and args | `.@max(5, 2, 8)` |

#### Object Arithmetic Functions

| Modifier | Arguments | Description | Example |
| --- | --- | --- | --- |
| `.@product` | `(field, ...)` | Product of numeric field references, variables, and literals | `/image.@product(width, height)` |
| `.@sum` | `(field, ...)` | Sum of explicit numeric fields | `/message.@sum(left, right)` |
| `.@mean` | `(field, ...)` | Mean of explicit numeric fields | `/message.@mean(low, high)` |
| `.@min` | `(field, ...)` | Minimum explicit numeric field | `/message.@min(width, height)` |
| `.@max` | `(field, ...)` | Maximum explicit numeric field | `/message.@max(width, height)` |
| `.@rms` | `(field, ...)` | RMS of explicit numeric fields | `/vector.@rms(x, y, z)` |

`.@product` is schema-generic: it works with any message containing numeric
dimensions, including images, matrices, grids, and point clouds.

#### Trigonometric Functions

| Modifier | Arguments | Description |
| --- | --- | --- |
| `.@sin` | none | Sine (radians) |
| `.@cos` | none | Cosine (radians) |
| `.@tan` | none | Tangent (radians) |
| `.@asin` | none | Arcsine (returns radians) |
| `.@acos` | none | Arccosine (returns radians) |
| `.@atan` | none | Arctangent (returns radians) |
| `.@degrees` | none | Convert radians to degrees |
| `.@radians` | none | Convert degrees to radians |
| `.@wrap_angle` | none | Wrap angle to [-pi, pi] range |

#### Robotics Functions

These operate on objects with specific field structures (e.g., geometry messages), not on scalar values.

| Modifier | Input fields | Description | Output |
| --- | --- | --- | --- |
| `.@length` | sequence | Number of elements in an array or typed array | `int` |
| `.@norm` | sequence or `x, y, z?` | Euclidean norm of a numeric array or vector | `float` |
| `.@magnitude` | sequence | L2 norm of a list/array of numbers | `float` |
| `.@rpy` | `x, y, z, w` | Quaternion to Euler angles | `EulerAngles(.roll, .pitch, .yaw)` |
| `.@quat` | `roll, pitch, yaw` | Euler angles to quaternion | `Quaternion(.x, .y, .z, .w)` |

The results of `.@rpy` and `.@quat` are objects with named fields, so you can chain field access:

```
/odom.pose.orientation.@rpy.yaw    # Extract just the yaw angle
/odom.pose.orientation.@rpy.roll   # Extract just the roll angle
/topic.euler.@quat.w              # Extract the w component of the quaternion
/odom.pose.position.@norm          # Distance from origin
/joint_states.velocity.@norm       # Norm of a numeric array
/joint_states.position.@length     # Number of array elements
```

#### Aggregate Functions

Aggregate modifiers reduce one numeric array in the current message to one
scalar. Bare `.@min` and `.@max` reduce arrays, while argument-taking forms
such as `.@max(5)` continue to map over arrays element by element.

| Modifier | Output | Empty array |
| --- | --- | --- |
| `.@min` | Smallest element | no value |
| `.@max` | Largest element | no value |
| `.@sum` | Floating-point sum | `0.0` |
| `.@mean` | Arithmetic mean | no value |
| `.@rms` | Root mean square | no value |

```
/joint_states.position.@max
/samples[:]{!=0}.@rms
/detections[:]{confidence>0.8}.score.@mean
```

These functions aggregate within each message. They do not retain history or
combine values across messages or files.

#### Time-Series Functions

Cross-message modifiers use `.@@` and are evaluated with a
`MessagePathEvaluator`. Evaluator instances own their state, leaving parsed
paths reusable across topics and threads.

| Modifier | Description |
| --- | --- |
| `.@@delta` | Difference from previous value |
| `.@@derivative` | Rate of change (delta / time delta) |
| `.@@timedelta` | Seconds elapsed since previous selected message |
| `.@@unchanged_for` | Seconds the selected value has remained unchanged |

```
/odom.pose.position.x.@@delta
/odom.pose.position.x.@@derivative
/topic.status{=="OK"}.@@timedelta
/camera.header.stamp.@to_nsec.@@unchanged_for
```

#### Cross-Message Aggregates

Stream reducers consume the scalar emitted for each message and produce one
value when the stream is finalized. Empty per-message results are skipped;
arrays must be explicitly reduced within each message. All reducers use
constant memory.

| Modifier | Result |
| --- | --- |
| `.@@count` | Number of emitted message values |
| `.@@min` / `.@@max` | Stream extrema |
| `.@@sum` / `.@@mean` / `.@@rms` | Stream numeric aggregates |
| `.@@variance` / `.@@stddev` | Population dispersion |
| `.@@first` / `.@@last` | First or last emitted value |

```text
/imu.linear_acceleration.@norm.@@max{<=30}
/lidar.@product(width, height).@@min{>=1000}
/position.x.@@delta.@@max{<=0.2}
/status{=="OK"}.@@timedelta.@@max{<=0.5}
/camera.header.stamp.@to_nsec.@@unchanged_for.@@max{<=0.5}
/value.@@delta{!=0}.@@count{<=10}
```

Normal MessagePath filters and modifiers may appear between stream stages. For
example, `.value.@@delta{!=0}.@@count` counts changes without requiring a
dedicated `@@changes` operation.

The evaluator uses the timestamp supplied to `observe()`; `pymcap-cli check`
supplies MCAP log time. To measure a message field's own clock, select that
scalar and use `.@@delta`, for example `.header.stamp.@to_nsec.@@delta`. The
first `delta`, `derivative`, or `timedelta` observation emits no value.
Non-finite numeric observations fail evaluation rather than being silently
skipped.

### 7. Variables

Variables are prefixed with `$` and can be used in slices, filters, and
modifier arguments:

```
/my_topic.array[$start_idx:$end_idx]
/my_topic.items[:]{id == $selected_id}
/my_topic{timestamp > $min_time}{timestamp < $max_time}
/my_topic.temperature.@mul($scale).@add($offset)
```

**Restrictions**:

- Variables can appear in array indices, slices, filters, and modifier arguments
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

### Boolean Logic in Filters

```
/detections.objects[:]{confidence > 0.8 || class == "person"}.label
```

Get labels for high-confidence detections or anything classified as a person.

```
/sensors[:]{!(status == "offline") && value > threshold}
```

Find online sensors where value exceeds their threshold (cross-field comparison).

### Membership Testing

```
/markers[:]{type in [1, 3, 5]}.pose
```

Get poses of markers with specific type IDs.

## Grammar Overview

```
message_path    : topic_ref (path_segment)*

topic_ref       : "/" identifier

path_segment    : field_access
                | array_index
                | array_slice
                | filter
                | math_modifier

field_access    : "." identifier

math_modifier   : ".@" identifier "(" args ")"
                | ".@" identifier

array_index     : "[" (integer | negative_int | variable) "]"

array_slice     : "[" slice_start? ":" slice_end? "]"

filter          : "{" filter_expr "}"

filter_expr     : or_expr
or_expr         : and_expr ("||" and_expr)*
and_expr        : not_expr ("&&" not_expr)*
not_expr        : "!" not_expr | filter_atom
filter_atom     : comparison
                | current_value_comparison
                | current_value_in_expr
                | in_expr
                | "(" filter_expr ")"
comparison      : field_path operator filter_value
current_value_comparison
                : operator current_filter_value
current_value_in_expr
                : "in" "[" current_filter_value ("," current_filter_value)* "]"
in_expr         : field_path "in" "[" filter_value ("," filter_value)* "]"

filter_value    : number | string | boolean | variable | field_path
current_filter_value
                : number | string | boolean | variable

operator        : "==" | "!=" | "<" | "<=" | ">" | ">="

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
