# mcap-codec-support

Reusable MCAP encoder and decoder factories used by `pymcap-cli`.

The package is intentionally factory-focused. ROS2 CDR support stays in
`mcap-ros2-support-fast`; this package composes with it when the relevant
factories are constructed.
