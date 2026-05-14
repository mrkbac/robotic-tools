# `core/processors/` architecture

The processor pipeline is the heart of pymcap-cli's transform commands
(`process`, `filter`, `merge`, `split`, `roscompress`, `rosdecompress`, …).
This document explains the contracts a new processor must satisfy.

## Two-tier model

Every transform is a chain of zero or more **input processors** followed by
exactly one **output router**.

| Tier              | Base class       | Responsibility                                                                 |
|-------------------|------------------|--------------------------------------------------------------------------------|
| Input processors  | `InputProcessor` | Observe / transform / drop / fan-out input records before output routing.      |
| Output router     | `OutputRouter`   | Decide which output segment(s) a surviving message or fast-copied chunk goes to. |

Both are defined in [`base.py`](base.py). The dispatcher in
`pymcap_cli.core` walks every input record through the chain of input
processors, then hands surviving messages / fast-copied chunks to the router.

## The four chunk decisions

A processor's `on_chunk()` returns a `ChunkDecision`. The dispatcher combines
decisions across the chain via "strongest wins" (decode > recompress >
skip > continue).

| Decision        | Meaning                                                                                                                  |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|
| `CONTINUE`      | Fast-copy: write the chunk's bytes verbatim to the output. No per-message work runs.                                     |
| `SKIP`          | Drop the chunk entirely. Nothing downstream sees it.                                                                     |
| `DECODE`        | Decompress and iterate messages through `on_message`. Necessary for any per-message transform or filter.                 |
| `RECOMPRESS`    | **Internal.** Decompress + re-compress with a different codec but skip message-level work. Set by the dispatcher only.   |
| `DECODE_VERIFY` | **Internal.** Channel/Schema ids were remapped; decode to check whether the in-chunk records still match the writer view. Used by `TopicRewriteProcessor`. |

Processors should return only `CONTINUE`, `SKIP`, or `DECODE`. The two internal
values exist for the dispatcher to express compression-target and
remap-verify states.

The single most important performance lever is keeping `CONTINUE` available:
a chain that returns `DECODE` for every chunk is up to 10× slower than one
that fast-copies.

## `MessageScope` — declare what you actually need

Most processors don't really need every chunk decoded — they only care about
specific channels. The default `message_scope()` returns `MessageScope.all()`,
which is conservative and forces `DECODE` everywhere. Override it:

```python
class TopicFilterProcessor(InputProcessor):
    def message_scope(self, context: ChunkContext) -> MessageScope:
        # We only need to see messages on channels we plan to drop.
        return MessageScope.channels(self._skipped_channel_ids)
```

`chunk_decision_for_message_scope()` (in `base.py`) maps a scope to a
decision using the chunk's `message_indexes`:

- `scope = NONE` → `CONTINUE` (fast-copy)
- `scope = ALL` → `DECODE` (always)
- `scope = CHANNELS(...)` → `DECODE` if the chunk contains any of those
  channels; `CONTINUE` otherwise.

If the input MCAP lacks chunk indexes, the helper falls back to `DECODE` —
correct but slow.

## `on_message` semantics

`on_message` is a generator that yields zero or more output messages:

| Behaviour              | What the next processor sees                                          |
|------------------------|-----------------------------------------------------------------------|
| `yield message`        | Pass-through.                                                         |
| (yield nothing)        | Drop. Nothing downstream sees the message.                            |
| `yield msg1; yield msg2` | Fan-out. Each yielded message enters at the **next** processor — this processor is not re-invoked for its own outputs. |
| `yield modified_msg`   | Replace.                                                              |

The "next processor, not myself" rule lets `TopicAliasProcessor` emit one
aliased copy without infinite recursion.

## Chain ordering

The chain is ordered: processors run in the order the dispatcher assembles
them. A few ordering rules are load-bearing:

1. **`AlwaysDecodeProcessor` before any per-message processor that needs to
   see *every* message** — e.g. `DedupIdenticalProcessor` only catches
   intra-input duplicates when chunks are decoded. Without
   `AlwaysDecodeProcessor`, chunks whose time ranges don't overlap another
   input fast-copy through and intra-input duplicates inside them slip past.
   (See `dedup.py`'s module docstring.)

2. **`TimeFilterProcessor` before `TopicFilterProcessor`** when both are
   active — time-pruned chunks short-circuit the topic regex work.

3. **`TopicRewriteProcessor` last among id-mutating processors** — it
   produces `DECODE_VERIFY` decisions that the dispatcher uses to short-cut
   chunks whose embedded Schema/Channel records are already clean.

4. **`OutputRouter.on_chunk` runs after every input processor's `on_chunk`** —
   the router gets the combined chunk decision and decides routing.

## Channel / Schema mutations

The two id-mutating processors are `TopicAliasProcessor` (adds new output
channels) and `TopicRewriteProcessor` (changes topic strings on existing
channels). Both interact with `InputContext.register_channel`:

```python
new_channel = context.register_channel(channel_blueprint)
# new_channel.id is assigned by the writer; emit messages on it via on_message.
```

Don't reuse input channel ids for output-only channels — the writer's
channel-id space is shared.

## Boundary processors and `OutputRouter`

Routers split an input stream across multiple output segments
(time-bucketed, size-capped, expression-driven). Three shapes exist:

| Processor / router        | Splits by                                       |
|---------------------------|-------------------------------------------------|
| `BoundarySplitProcessor`  | Caller-supplied segment boundaries.             |
| `DurationSplitRouter`     | Wall-clock duration per output.                 |
| `SizeSplitRouter`         | Byte budget per output.                         |
| `TimestampSplitRouter`    | Aligned wall-clock windows (hourly, daily, …).  |
| `ExpressionSplitRouter`   | A message-path predicate flipping `True`/`False`. |

Routers that can produce all segment boundaries upfront return them from
`output_segments()`; the dispatcher pre-creates writers and routes
`route_chunk` / `route_message` to known keys. Routers that don't know
boundaries ahead of time return `None` and rely on the
`SPLIT_REQUIRED` sentinel from `route_chunk` to ask the dispatcher to
materialise a new segment mid-stream.

## Processor catalogue (current set)

Quick reference; consult each module's docstring for full semantics.

### Decode-policy

- `AlwaysDecodeProcessor` — force `DECODE` on every chunk.

### Filtering (drop records)

- `TopicFilterProcessor` — regex include/exclude on channel topic.
- `TimeFilterProcessor` — drop messages outside `[start, end]`.
- `MetadataFilterProcessor` — keep/drop Metadata records by name regex.
- `AttachmentFilterProcessor` — keep/drop Attachment records by name regex.

### Transform (rewrite records)

- `TopicAliasProcessor` — emit a duplicate of each matching message on a new topic.
- `TopicRewriteProcessor` — change topic string on existing channels (id-stable).
- `ChannelMergeProcessor` — collapse N input channels with the same schema into one output channel.
- `TimeOffsetProcessor` — shift `log_time` / `publish_time` by a constant.

### Deduplication

- `DedupIdenticalProcessor` — drop bit-identical `(channel_id, log_time, payload)` repeats. See ordering note above.
- `NthMessageProcessor` — keep every Nth message per channel.
- `LatchingProcessor` — replay the latest message on each latched topic at segment open.

### Output routing (one per chain)

- `BoundarySplitProcessor` / `DurationSplitRouter` / `SizeSplitRouter` /
  `TimestampSplitRouter` / `ExpressionSplitRouter` — see "Boundary processors" above.

## Writing a new processor

1. Subclass `InputProcessor` (or `OutputRouter`).
2. Decide whether you need to see messages. If not, leave `on_chunk` alone
   and override `message_scope()` to return `MessageScope.none()` — this
   lets the chain fast-copy chunks past you.
3. If you do need messages, override `message_scope()` to scope the work
   to the channels you care about. Don't return `ALL` unless you really mean it.
4. Place your processor in the chain consciously. Ordering matters; see
   "Chain ordering" above.
5. Add tests under `pymcap-cli/tests/` named `test_<your_processor>.py`.
