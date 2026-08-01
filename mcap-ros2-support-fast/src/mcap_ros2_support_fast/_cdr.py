"""CDR (Common Data Representation) constants for ROS2 message encoding."""

# CDR encapsulation kinds are encoded as a big-endian uint16 in bytes 0-1.
CDR_BIG_ENDIAN = 0x00
CDR_LITTLE_ENDIAN = 0x01

# CDR header size (encapsulation header)
CDR_HEADER_SIZE = 4

# CDR header formats
# Format: [encapsulation_kind (big-endian uint16), options (uint16)]
# Little-endian CDR: b'\x00\x01\x00\x00'
CDR_HEADER_LITTLE_ENDIAN = b"\x00\x01\x00\x00"
# Big-endian CDR: b'\x00\x00\x00\x00'
CDR_HEADER_BIG_ENDIAN = b"\x00\x00\x00\x00"
