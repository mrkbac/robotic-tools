import {
  IconCamera,
  IconCameraPlus,
  IconCube,
  IconCompass,
  IconArrowsMove,
  IconRoute,
  IconHierarchy2,
  IconRadar,
  IconRobot,
  IconHeartbeat,
  IconMapPin,
  IconDatabase,
  IconSettings,
  IconMessage,
  IconBox,
  type Icon,
} from "@tabler/icons-react";

interface SchemaIcon {
  Icon: Icon;
  label: string;
  color: string;
}

const SUFFIX_MAP: [string, Icon, string, string][] = [
  // sensor_msgs
  ["CompressedImage", IconCamera, "Image", "#e64980"],
  ["Image", IconCamera, "Image", "#e64980"],
  ["CameraInfo", IconCameraPlus, "Camera Info", "#d6336c"],
  ["PointCloud2", IconCube, "Point Cloud", "#7950f2"],
  ["PointCloud", IconCube, "Point Cloud", "#7950f2"],
  ["Imu", IconCompass, "IMU", "#15aabf"],
  ["LaserScan", IconRadar, "Laser Scan", "#fd7e14"],
  ["NavSatFix", IconMapPin, "GPS", "#40c057"],
  ["GpsRawInt", IconMapPin, "GPS", "#40c057"],
  ["GlobalPosition", IconMapPin, "GPS", "#40c057"],

  // geometry_msgs
  ["TwistStamped", IconArrowsMove, "Twist", "#4dabf7"],
  ["Twist", IconArrowsMove, "Twist", "#4dabf7"],
  ["Odometry", IconRoute, "Odometry", "#3bc9db"],

  // tf2_msgs
  ["TFMessage", IconHierarchy2, "TF", "#fcc419"],

  // control
  ["JointState", IconRobot, "Joint State", "#ff922b"],

  // diagnostic_msgs
  ["DiagnosticArray", IconHeartbeat, "Diagnostics", "#ff6b6b"],
  ["DiagnosticStatus", IconHeartbeat, "Diagnostics", "#ff6b6b"],

  // rcl_interfaces
  ["ParameterEvent", IconSettings, "Parameter Event", "#868e96"],
  ["Log", IconMessage, "Log", "#868e96"],
];

// std_msgs types → generic box icon
const STD_MSGS_TYPES = new Set([
  "Bool", "Byte", "Char",
  "Float32", "Float64",
  "Int8", "Int16", "Int32", "Int64",
  "UInt8", "UInt16", "UInt32", "UInt64",
  "String", "ColorRGBA", "Header",
  "Empty",
  "Float32MultiArray", "Float64MultiArray",
  "Int8MultiArray", "Int16MultiArray", "Int32MultiArray", "Int64MultiArray",
  "UInt8MultiArray", "UInt16MultiArray", "UInt32MultiArray", "UInt64MultiArray",
  "ByteMultiArray",
  "MultiArrayDimension", "MultiArrayLayout",
]);

/** Map a schema name to a Tabler icon using suffix matching. */
export function getSchemaIcon(schemaName: string | null | undefined): SchemaIcon {
  if (!schemaName) return { Icon: IconDatabase, label: "Unknown", color: "#adb5bd" };

  // Support both `/` (ROS) and `.` (Foxglove) separators
  const suffix = schemaName.split(/[./]/).pop() ?? "";

  for (const [match, Icon, label, color] of SUFFIX_MAP) {
    if (suffix === match) return { Icon, label, color };
  }

  // diagnostic_msgs package prefix
  if (schemaName.includes("diagnostic_msgs")) {
    return { Icon: IconHeartbeat, label: "Diagnostics", color: "#ff6b6b" };
  }

  // std_msgs types
  if (schemaName.includes("std_msgs") || STD_MSGS_TYPES.has(suffix)) {
    return { Icon: IconBox, label: `std: ${suffix}`, color: "#74c0fc" };
  }

  // rcl_interfaces fallback
  if (schemaName.includes("rcl_interfaces")) {
    return { Icon: IconSettings, label: suffix, color: "#868e96" };
  }

  return { Icon: IconDatabase, label: "Data", color: "#adb5bd" };
}
