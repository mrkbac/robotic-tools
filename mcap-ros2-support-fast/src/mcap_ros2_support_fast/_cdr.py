"""CDR (Common Data Representation) constants for ROS2 message encoding."""

# CDR header endianness markers
CDR_LITTLE_ENDIAN = 0x00
CDR_BIG_ENDIAN = 0x01

# CDR header size (encapsulation header)
CDR_HEADER_SIZE = 4

# CDR header formats
# Format: [encapsulation_kind (1 byte), options (1 byte), padding (2 bytes)]
# Little-endian: b'\x00\x01\x00\x00'
CDR_HEADER_LITTLE_ENDIAN = b"\x00\x01\x00\x00"
# Big-endian: b'\x01\x01\x00\x00'
CDR_HEADER_BIG_ENDIAN = b"\x01\x01\x00\x00"
