import {
  IconApi,
  IconArrowsMove,
  IconBattery,
  IconBox,
  IconCamera,
  IconCameraPlus,
  IconClock,
  IconCompass,
  IconCrosshair,
  IconCube,
  IconDatabase,
  IconDeviceGamepad2,
  IconDroplet,
  IconFocusCentered,
  IconHeartbeat,
  IconHierarchy2,
  IconMagnet,
  IconMap,
  IconMapPin,
  IconMarquee,
  IconMessage,
  IconNavigation,
  IconPoint,
  IconPolygon,
  IconRadar,
  IconRobot,
  IconRotate3d,
  IconRoute,
  IconRuler,
  IconScale,
  IconSettings,
  IconShape,
  IconStereoGlasses,
  IconSun,
  IconTemperature,
  IconTimeline,
  IconVector,
  IconVectorTriangle,
  type Icon,
} from "@tabler/icons-react";

export interface SchemaIcon {
  Icon: Icon;
  color: string;
}

const SUFFIX_MAP: [string, Icon, string][] = [
  // sensor_msgs
  ["CompressedImage", IconCamera, "#e64980"],
  ["Image", IconCamera, "#e64980"],
  ["CameraInfo", IconCameraPlus, "#d6336c"],
  ["PointCloud2", IconCube, "#7950f2"],
  ["PointCloud", IconCube, "#7950f2"],
  ["Imu", IconCompass, "#15aabf"],
  ["MultiEchoLaserScan", IconRadar, "#fd7e14"],
  ["LaserScan", IconRadar, "#fd7e14"],
  ["NavSatFix", IconMapPin, "#40c057"],
  ["GpsRawInt", IconMapPin, "#40c057"],
  ["GlobalPosition", IconMapPin, "#40c057"],
  ["BatteryState", IconBattery, "#d6336c"],
  ["Temperature", IconTemperature, "#d6336c"],
  ["Illuminance", IconSun, "#d6336c"],
  ["FluidPressure", IconDroplet, "#d6336c"],
  ["RelativeHumidity", IconDroplet, "#d6336c"],
  ["MagneticField", IconMagnet, "#d6336c"],
  ["JoyFeedbackArray", IconDeviceGamepad2, "#d6336c"],
  ["JoyFeedback", IconDeviceGamepad2, "#d6336c"],
  ["Joy", IconDeviceGamepad2, "#d6336c"],
  ["Range", IconRuler, "#d6336c"],
  ["RegionOfInterest", IconFocusCentered, "#d6336c"],
  ["TimeReference", IconClock, "#d6336c"],

  // geometry_msgs — more specific suffixes first
  ["TwistWithCovarianceStamped", IconArrowsMove, "#4dabf7"],
  ["TwistWithCovariance", IconArrowsMove, "#4dabf7"],
  ["TwistStamped", IconArrowsMove, "#4dabf7"],
  ["Twist", IconArrowsMove, "#4dabf7"],
  ["Odometry", IconRoute, "#3bc9db"],
  ["PointStamped", IconPoint, "#4dabf7"],
  ["Point32", IconPoint, "#4dabf7"],
  ["PoseWithCovarianceStamped", IconCrosshair, "#4dabf7"],
  ["PoseWithCovariance", IconCrosshair, "#4dabf7"],
  ["PoseStamped", IconCrosshair, "#4dabf7"],
  ["PoseArray", IconCrosshair, "#4dabf7"],
  ["Pose2D", IconCrosshair, "#4dabf7"],
  ["Pose", IconCrosshair, "#4dabf7"],
  ["QuaternionStamped", IconRotate3d, "#4dabf7"],
  ["Quaternion", IconRotate3d, "#4dabf7"],
  ["TransformStamped", IconRotate3d, "#4dabf7"],
  ["Transform", IconRotate3d, "#4dabf7"],
  ["Vector3Stamped", IconVector, "#4dabf7"],
  ["Vector3", IconVector, "#4dabf7"],
  ["AccelWithCovarianceStamped", IconVector, "#4dabf7"],
  ["AccelWithCovariance", IconVector, "#4dabf7"],
  ["AccelStamped", IconVector, "#4dabf7"],
  ["Accel", IconVector, "#4dabf7"],
  ["WrenchStamped", IconVector, "#4dabf7"],
  ["Wrench", IconVector, "#4dabf7"],
  ["PolygonStamped", IconPolygon, "#4dabf7"],
  ["Polygon", IconPolygon, "#4dabf7"],
  ["InertiaStamped", IconScale, "#4dabf7"],
  ["Inertia", IconScale, "#4dabf7"],

  // tf2_msgs
  ["TFMessage", IconHierarchy2, "#fcc419"],

  // nav_msgs
  ["OccupancyGrid", IconMap, "#20c997"],
  ["GridCells", IconMap, "#20c997"],
  ["MapMetaData", IconMap, "#20c997"],
  ["Path", IconNavigation, "#20c997"],

  // shape_msgs
  ["MeshTriangle", IconVectorTriangle, "#845ef7"],
  ["Mesh", IconVectorTriangle, "#845ef7"],
  ["SolidPrimitive", IconShape, "#845ef7"],
  ["Plane", IconShape, "#845ef7"],

  // stereo_msgs
  ["DisparityImage", IconStereoGlasses, "#be4bdb"],

  // trajectory_msgs — more specific first
  ["MultiDOFJointTrajectoryPoint", IconTimeline, "#ff922b"],
  ["MultiDOFJointTrajectory", IconTimeline, "#ff922b"],
  ["JointTrajectoryPoint", IconTimeline, "#ff922b"],
  ["JointTrajectory", IconTimeline, "#ff922b"],

  // visualization_msgs
  ["InteractiveMarkerControl", IconMarquee, "#82c91e"],
  ["InteractiveMarkerFeedback", IconMarquee, "#82c91e"],
  ["InteractiveMarkerInit", IconMarquee, "#82c91e"],
  ["InteractiveMarkerPose", IconMarquee, "#82c91e"],
  ["InteractiveMarkerUpdate", IconMarquee, "#82c91e"],
  ["InteractiveMarker", IconMarquee, "#82c91e"],
  ["ImageMarker", IconMarquee, "#82c91e"],
  ["MarkerArray", IconMarquee, "#82c91e"],
  ["Marker", IconMarquee, "#82c91e"],
  ["MenuEntry", IconMarquee, "#82c91e"],
  ["UVCoordinate", IconMarquee, "#82c91e"],

  // control
  ["MultiDOFJointState", IconRobot, "#ff922b"],
  ["JointState", IconRobot, "#ff922b"],

  // diagnostic_msgs
  ["DiagnosticArray", IconHeartbeat, "#ff6b6b"],
  ["DiagnosticStatus", IconHeartbeat, "#ff6b6b"],

  // rcl_interfaces
  ["ParameterEvent", IconSettings, "#868e96"],
  ["Log", IconMessage, "#868e96"],

  // geometry_msgs — Point last (least specific, matches many suffixes)
  ["Point", IconPoint, "#4dabf7"],
];

// std_msgs types → generic box icon
const STD_MSGS_TYPES = new Set([
  "Bool",
  "Byte",
  "Char",
  "Float32",
  "Float64",
  "Int8",
  "Int16",
  "Int32",
  "Int64",
  "UInt8",
  "UInt16",
  "UInt32",
  "UInt64",
  "String",
  "ColorRGBA",
  "Header",
  "Empty",
  "Float32MultiArray",
  "Float64MultiArray",
  "Int8MultiArray",
  "Int16MultiArray",
  "Int32MultiArray",
  "Int64MultiArray",
  "UInt8MultiArray",
  "UInt16MultiArray",
  "UInt32MultiArray",
  "UInt64MultiArray",
  "ByteMultiArray",
  "MultiArrayDimension",
  "MultiArrayLayout",
]);

/** Map a schema name to a Tabler icon using suffix matching. */
export function getSchemaIcon(
  schemaName: string | null | undefined,
): SchemaIcon {
  if (!schemaName) return { Icon: IconDatabase, color: "#adb5bd" };

  // Support both `/` (ROS) and `.` (Foxglove) separators
  const suffix = schemaName.split(/[./]/).pop() ?? "";

  for (const [match, Icon, color] of SUFFIX_MAP) {
    if (suffix === match) return { Icon, color };
  }

  // Package prefix fallbacks (order matters — more specific first)
  if (schemaName.includes("sensor_msgs")) {
    return { Icon: IconRadar, color: "#e64980" };
  }
  if (schemaName.includes("geometry_msgs")) {
    return { Icon: IconVector, color: "#4dabf7" };
  }
  if (schemaName.includes("nav_msgs")) {
    return { Icon: IconNavigation, color: "#20c997" };
  }
  if (schemaName.includes("shape_msgs")) {
    return { Icon: IconShape, color: "#845ef7" };
  }
  if (schemaName.includes("stereo_msgs")) {
    return { Icon: IconStereoGlasses, color: "#be4bdb" };
  }
  if (schemaName.includes("trajectory_msgs")) {
    return { Icon: IconTimeline, color: "#ff922b" };
  }
  if (schemaName.includes("visualization_msgs")) {
    return { Icon: IconMarquee, color: "#82c91e" };
  }
  if (schemaName.includes("diagnostic_msgs")) {
    return { Icon: IconHeartbeat, color: "#ff6b6b" };
  }
  if (schemaName.includes("std_srvs")) {
    return { Icon: IconApi, color: "#ced4da" };
  }
  if (schemaName.includes("std_msgs") || STD_MSGS_TYPES.has(suffix)) {
    return { Icon: IconBox, color: "#74c0fc" };
  }
  if (schemaName.includes("rcl_interfaces")) {
    return { Icon: IconSettings, color: "#868e96" };
  }
  if (schemaName.includes("tf2_msgs")) {
    return { Icon: IconHierarchy2, color: "#fcc419" };
  }

  return { Icon: IconDatabase, color: "#adb5bd" };
}
