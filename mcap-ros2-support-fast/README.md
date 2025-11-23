# Python MCAP ROS2 support fast

This package provides fast ROS2 support for the Python MCAP file format reader.
It has no dependencies on ROS2 itself or a ROS2 environment, and can be used in any Python project.

## Benchmarks

Full benchmark suite comparing against reference mcap-ros2 implementation:

### Read Performance

```txt
--------------------------------------------------------- benchmark 'msgs-10': 2 tests --------------------------------------------------------
Name (time in ms)                                  Min                 Max                Mean             StdDev                 OPS
-----------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_decoder[mcap_ros2_fast-10]       5.9578 (1.0)        7.2528 (1.0)        6.2849 (1.0)       0.3372 (1.0)      159.1123 (1.0)
test_benchmark_decoder[mcap_ros2-10]          349.7705 (58.71)    385.8141 (53.19)    368.3301 (58.61)    17.3565 (51.48)      2.7150 (0.02)
-----------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------- benchmark 'msgs-100': 2 tests -----------------------------------------------------------
Name (time in ms)                                     Min                   Max                  Mean             StdDev                OPS
-----------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_decoder[mcap_ros2_fast-100]        36.6542 (1.0)         43.0995 (1.0)         39.4595 (1.0)       2.5661 (1.0)      25.3425 (1.0)
test_benchmark_decoder[mcap_ros2-100]          1,344.9558 (36.69)    1,410.8077 (32.73)    1,372.9393 (34.79)    24.0212 (9.36)      0.7284 (0.03)
-----------------------------------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------- benchmark 'msgs-1000': 2 tests ----------------------------------------------------------
Name (time in ms)                                      Min                   Max                  Mean             StdDev               OPS
-----------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_decoder[mcap_ros2_fast-1000]       128.1334 (1.0)        141.1800 (1.0)        136.0132 (1.0)       4.5703 (1.0)      7.3522 (1.0)
test_benchmark_decoder[mcap_ros2-1000]          4,025.6981 (31.42)    4,095.9460 (29.01)    4,058.3911 (29.84)    30.6044 (6.70)     0.2464 (0.03)
-----------------------------------------------------------------------------------------------------------------------------------------------------
```

**Read Summary:** mcap-ros2-support-fast is **30-59x faster** for reading/decoding.

### Write Performance

```txt
------------------------------------------------------------ benchmark 'write-msgs-10': 2 tests ------------------------------------------------------------
Name (time in ms)                                                Min                 Max                Mean             StdDev                OPS
------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read_and_write[mcap_ros2_fast_writer-10]      19.8460 (1.0)       21.6802 (1.0)       20.6039 (1.0)       0.5250 (1.0)      48.5346 (1.0)
test_benchmark_read_and_write[mcap_ros2_writer-10]          945.4425 (47.64)    989.6685 (45.65)    962.1396 (46.70)    21.5564 (41.06)     1.0394 (0.02)
------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------- benchmark 'write-msgs-100': 2 tests ---------------------------------------------------------------
Name (time in ms)                                                   Min                   Max                  Mean              StdDev               OPS
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read_and_write[mcap_ros2_fast_writer-100]       109.4476 (1.0)      2,299.3445 (1.0)        392.4757 (1.0)      770.5167 (4.02)     2.5479 (1.0)
test_benchmark_read_and_write[mcap_ros2_writer-100]          2,955.4963 (27.00)    3,410.1705 (1.48)     3,161.0002 (8.05)     191.4705 (1.0)      0.3164 (0.12)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------- benchmark 'write-msgs-1000': 2 tests ----------------------------------------------------------------
Name (time in ms)                                                    Min                    Max                   Mean              StdDev               OPS
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_read_and_write[mcap_ros2_fast_writer-1000]       404.2989 (1.0)         420.1088 (1.0)         411.3845 (1.0)        7.2740 (1.0)      2.4308 (1.0)
test_benchmark_read_and_write[mcap_ros2_writer-1000]          9,788.0549 (24.21)    10,766.2442 (25.63)    10,311.2563 (25.06)    407.7202 (56.05)    0.0970 (0.04)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

**Write Summary:** mcap-ros2-support-fast is **8-47x faster** for writing/encoding.

## Performance Summary

- **Read/Decode:** 30-59x faster than reference mcap-ros2
- **Write/Encode:** 8-47x faster than reference mcap-ros2

## Key Optimizations

This implementation achieves exceptional performance through several optimizations:

1. **Code Generation**: Pre-compiles message-specific decoder/encoder functions, eliminating runtime dispatch overhead
2. **Direct Memory Access**: Uses `memoryview.cast()` for zero-copy array access when endianness matches
3. **Primitive Grouping**: Batches consecutive primitive fields into single struct operations with proper alignment
4. **Minimal Allocations**: Eliminates redundant variable assignments and intermediate allocations
5. **Endianness Detection**: Generates both little-endian and big-endian decoders, dispatching based on CDR header

Based on <https://github.com/foxglove/mcap/tree/main/python> (MIT licensed)
