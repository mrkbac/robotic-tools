# MCAP Processor

The `McapProcessor` is a unified processor for MCAP files that handles recovery, filtering, merging, and rechunking through a single processing pipeline.

## Key Components

| Component           | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `InputOptions`      | Per-input file configuration: stream, time/topic filtering, content flags |
| `OutputOptions`     | Output configuration: compression, chunk size, rechunking strategy        |
| `ProcessingOptions` | Combines list of `InputOptions` with single `OutputOptions`               |
| `ProcessingStats`   | Tracks input/output message counts, chunks processed, and errors          |
| `Remapper`          | Handles ID remapping when merging multiple files                          |
| `MessageGroup`      | Manages independent chunk groups for rechunking by topic pattern          |

## Processing Flow

```mermaid
flowchart TD
    subgraph Input["Input Files"]
        F1[file1.mcap]
        F2[file2.mcap]
    end

    subgraph Init["Initialization"]
        SUM[Extract Summary<br/>schemas, channels]
        REMAP[Remap IDs<br/>for multi-file]
        CACHE[Cache Filter<br/>Decisions]
    end

    subgraph Stream["Stream Processing"]
        GEN[Generate Chunks<br/>from each file]
        MERGE[Merge by Timestamp<br/>heapq.merge]
    end

    subgraph Decision["Per-Chunk Decision"]
        DEC{Should Decode?}
        SKIP[SKIP<br/>chunk filtered out]
        COPY[COPY<br/>fast-copy unchanged]
        DECODE[DECODE<br/>decompress & filter]
    end

    subgraph MsgProc["Message Processing"]
        TIME[Time Filter]
        TOPIC[Topic Filter]
        ROUTE{Rechunking?}
        DIRECT[Write Direct]
        GROUP[Route to<br/>MessageGroup]
    end

    subgraph Output["Output"]
        WRITER[McapWriter]
        OUT[output.mcap]
    end

    F1 --> SUM
    F2 --> SUM
    SUM --> REMAP
    REMAP --> CACHE
    CACHE --> GEN
    GEN --> MERGE
    MERGE --> DEC

    DEC -->|"outside time range<br/>or all channels excluded"| SKIP
    DEC -->|"no changes needed<br/>compression matches"| COPY
    DEC -->|"needs filtering<br/>or remapping"| DECODE

    COPY --> WRITER
    DECODE --> TIME
    TIME --> TOPIC
    TOPIC --> ROUTE
    ROUTE -->|No| DIRECT
    ROUTE -->|Yes| GROUP
    DIRECT --> WRITER
    GROUP --> WRITER
    WRITER --> OUT
```

## Processing Paths

### SKIP

Chunk is discarded entirely when:

- All messages fall outside the time range
- All channels in the chunk are excluded by topic filters

### COPY (Fast Path)

Chunk is appended directly without decoding when:

- No time filtering required within the chunk
- No channel ID remapping needed
- Output compression matches chunk compression
- All channels pass topic filters

### DECODE

Chunk must be decompressed and messages processed individually when:

- Time filtering needs per-message evaluation
- Channel IDs were remapped (multi-file merge)
- Compression format differs from output
- Mixed included/excluded channels in chunk
- Rechunking is active

## Multi-File Merging

When processing multiple files:

1. Each file's schemas and channels are remapped to avoid ID conflicts
2. Chunks from all files are merged chronologically using `heapq.merge()`
3. Messages maintain global timestamp ordering in the output
