pub fn encode(value: i64, output: &mut [u8]) -> usize {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    let mut encoded = zigzag.wrapping_add(1);
    let mut position = 0;
    while encoded > 0x7f {
        output[position] = (encoded as u8 & 0x7f) | 0x80;
        encoded >>= 7;
        position += 1;
    }
    output[position] = encoded as u8;
    position + 1
}

#[inline(always)]
pub fn append(value: i64, output: &mut Vec<u8>) {
    let mut encoded = encode_value(value);
    if output.capacity() - output.len() < 10 {
        output.reserve(10);
    }
    let start = output.len();
    let mut length = 0;
    // The reserve above guarantees ten writable bytes, the maximum encoded length.
    unsafe {
        let destination = output.as_mut_ptr().add(start);
        while encoded > 0x7f {
            destination.add(length).write((encoded as u8 & 0x7f) | 0x80);
            encoded >>= 7;
            length += 1;
        }
        destination.add(length).write(encoded as u8);
        output.set_len(start + length + 1);
    }
}

#[inline(always)]
pub fn append_batch<const N: usize>(values: [i64; N], valid_mask: u64, output: &mut Vec<u8>) {
    let required = N * 10;
    if output.capacity() - output.len() < required {
        output.reserve(required);
    }
    let start = output.len();
    let mut length = 0;
    // The reserve above guarantees ten writable bytes for every encoded value.
    unsafe {
        let destination = output.as_mut_ptr().add(start);
        for (index, value) in values.into_iter().enumerate() {
            let mut encoded = if valid_mask & (1 << index) == 0 {
                0
            } else {
                encode_value(value)
            };
            while encoded > 0x7f {
                destination.add(length).write((encoded as u8 & 0x7f) | 0x80);
                encoded >>= 7;
                length += 1;
            }
            destination.add(length).write(encoded as u8);
            length += 1;
        }
        output.set_len(start + length);
    }
}

#[inline(always)]
fn encode_value(value: i64) -> u64 {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    zigzag.wrapping_add(1)
}

pub fn decode(input: &[u8]) -> Result<(i64, usize), String> {
    let mut position = 0;
    let value = decode_at(input, &mut position)?;
    Ok((value, position))
}

#[inline(always)]
pub fn decode_at(input: &[u8], position: &mut usize) -> Result<i64, String> {
    let start = *position;
    let first = *input
        .get(start)
        .ok_or_else(|| "Invalid CloudINI varint: truncated or too long".to_string())?;
    if first & 0x80 == 0 {
        *position = start + 1;
        return decode_encoded(u64::from(first));
    }
    if input.len() - start >= 10 {
        return decode_continuation_unchecked(input, position, start, first);
    }

    let second = *input
        .get(start + 1)
        .ok_or_else(|| "Invalid CloudINI varint: truncated or too long".to_string())?;
    let mut encoded = u64::from(first & 0x7f) | (u64::from(second & 0x7f) << 7);
    if second & 0x80 == 0 {
        *position = start + 2;
        return decode_encoded(encoded);
    }

    let mut shift = 14_u32;
    for index in 2..10 {
        let byte = *input
            .get(start + index)
            .ok_or_else(|| "Invalid CloudINI varint: truncated or too long".to_string())?;
        if shift == 63 && byte & 0x7f > 1 {
            return Err("Invalid CloudINI varint: overflow".to_string());
        }
        encoded |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            *position = start + index + 1;
            return decode_encoded(encoded);
        }
        shift += 7;
    }
    Err("Invalid CloudINI varint: truncated or too long".to_string())
}

#[inline(always)]
fn decode_continuation_unchecked(
    input: &[u8],
    position: &mut usize,
    start: usize,
    first: u8,
) -> Result<i64, String> {
    // The caller proves that all ten possible varint bytes are in bounds.
    unsafe {
        let source = input.as_ptr().add(start);
        let second = *source.add(1);
        let mut encoded = u64::from(first & 0x7f) | (u64::from(second & 0x7f) << 7);
        if second & 0x80 == 0 {
            *position = start + 2;
            return decode_encoded(encoded);
        }

        let mut shift = 14_u32;
        for index in 2..10 {
            let byte = *source.add(index);
            if shift == 63 && byte & 0x7f > 1 {
                return Err("Invalid CloudINI varint: overflow".to_string());
            }
            encoded |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                *position = start + index + 1;
                return decode_encoded(encoded);
            }
            shift += 7;
        }
    }
    Err("Invalid CloudINI varint: truncated or too long".to_string())
}

#[inline(always)]
fn decode_encoded(encoded: u64) -> Result<i64, String> {
    if encoded == 0 {
        return Err("Invalid CloudINI varint: reserved NaN marker".to_string());
    }
    let zigzag = encoded - 1;
    Ok(((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64))
}
