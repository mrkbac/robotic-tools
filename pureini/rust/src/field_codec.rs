use crate::types::{EncodingOptions, FieldType, PointField};
use crate::varint;

pub enum FieldEncoder {
    Copy {
        offset: usize,
        size: usize,
    },
    Integer {
        offset: usize,
        field_type: FieldType,
        previous: i64,
    },
    Float32Lossy {
        offset: usize,
        multiplier: f32,
        previous: i64,
    },
    Float64Lossy {
        offset: usize,
        multiplier: f64,
        previous: i64,
    },
    Float32Xor {
        offset: usize,
        previous: u32,
    },
    Float64Xor {
        offset: usize,
        previous: u64,
    },
    Float64Gorilla {
        offset: usize,
        previous: u64,
        previous_leading: u8,
        previous_trailing: u8,
        is_first: bool,
    },
    Float32VectorLossy {
        offsets: [usize; 4],
        multipliers: [f32; 4],
        previous: [i32; 4],
        count: usize,
    },
}

impl FieldEncoder {
    #[inline(always)]
    pub fn reset(&mut self) {
        match self {
            Self::Copy { .. } => {}
            Self::Integer { previous, .. }
            | Self::Float32Lossy { previous, .. }
            | Self::Float64Lossy { previous, .. } => *previous = 0,
            Self::Float32Xor { previous, .. } => *previous = 0,
            Self::Float64Xor { previous, .. } => *previous = 0,
            Self::Float64Gorilla {
                previous,
                previous_leading,
                previous_trailing,
                is_first,
                ..
            } => {
                *previous = 0;
                *previous_leading = u8::MAX;
                *previous_trailing = 0;
                *is_first = true;
            }
            Self::Float32VectorLossy { previous, .. } => *previous = [0; 4],
        }
    }

    #[inline(always)]
    pub fn encode(&mut self, point: &[u8], output: &mut Vec<u8>) {
        match self {
            Self::Copy { offset, size } => {
                output.extend_from_slice(&point[*offset..*offset + *size]);
            }
            Self::Integer {
                offset,
                field_type,
                previous,
            } => {
                let value = read_integer(point, *offset, *field_type);
                let delta = value.wrapping_sub(*previous);
                *previous = value;
                varint::append(delta, output);
            }
            Self::Float32Lossy {
                offset,
                multiplier,
                previous,
            } => {
                let value = read_f32(point, *offset);
                if !value.is_finite() {
                    output.push(0);
                    *previous = 0;
                    return;
                }
                let quantized = (value * *multiplier).round() as i64;
                let delta = representable_lossy_delta(quantized, *previous);
                *previous = previous.wrapping_add(delta);
                varint::append(delta, output);
            }
            Self::Float64Lossy {
                offset,
                multiplier,
                previous,
            } => {
                let value = read_f64(point, *offset);
                if !value.is_finite() {
                    output.push(0);
                    *previous = 0;
                    return;
                }
                let quantized = (value * *multiplier).round() as i64;
                let delta = representable_lossy_delta(quantized, *previous);
                *previous = previous.wrapping_add(delta);
                varint::append(delta, output);
            }
            Self::Float32Xor { offset, previous } => {
                let current = u32::from_le_bytes(point[*offset..*offset + 4].try_into().unwrap());
                let residual = current ^ *previous;
                *previous = current;
                output.extend_from_slice(&residual.to_le_bytes());
            }
            Self::Float64Xor { offset, previous } => {
                let current = u64::from_le_bytes(point[*offset..*offset + 8].try_into().unwrap());
                let residual = current ^ *previous;
                *previous = current;
                output.extend_from_slice(&residual.to_le_bytes());
            }
            Self::Float64Gorilla {
                offset,
                previous,
                previous_leading,
                previous_trailing,
                is_first,
            } => {
                let current = u64::from_le_bytes(point[*offset..*offset + 8].try_into().unwrap());
                let mut writer = BitWriter::default();
                if *is_first {
                    writer.push(current, 64);
                    *is_first = false;
                } else {
                    let residual = current ^ *previous;
                    if residual == 0 {
                        writer.push(0, 1);
                    } else {
                        writer.push(1, 1);
                        let leading = residual.leading_zeros() as u8;
                        let trailing = residual.trailing_zeros() as u8;
                        if *previous_leading != u8::MAX
                            && leading >= *previous_leading
                            && trailing >= *previous_trailing
                        {
                            writer.push(0, 1);
                            let meaningful = 64 - *previous_leading - *previous_trailing;
                            writer.push(residual >> *previous_trailing, meaningful);
                        } else {
                            writer.push(1, 1);
                            let stored_leading = leading.min(31);
                            let meaningful = 64 - stored_leading - trailing;
                            writer.push(u64::from(stored_leading), 5);
                            writer.push(u64::from(meaningful - 1), 6);
                            writer.push(residual >> trailing, meaningful);
                            *previous_leading = stored_leading;
                            *previous_trailing = trailing;
                        }
                    }
                }
                *previous = current;
                let bytes = writer.finish();
                output.extend_from_slice(&bytes);
            }
            Self::Float32VectorLossy {
                offsets,
                multipliers,
                previous,
                count,
            } => {
                for index in 0..*count {
                    let value = read_f32(point, offsets[index]);
                    if !value.is_finite() {
                        output.push(0);
                        previous[index] = 0;
                        continue;
                    }
                    let quantized = (value * multipliers[index]).round_ties_even() as i32;
                    let delta = quantized.wrapping_sub(previous[index]);
                    previous[index] = quantized;
                    varint::append(delta as i64, output);
                }
            }
        }
    }

    pub fn encode_points(
        &mut self,
        cloud_data: &[u8],
        point_step: usize,
        output: &mut Vec<u8>,
    ) -> bool {
        match self {
            Self::Copy { offset, size } => {
                for point in cloud_data.chunks_exact(point_step) {
                    output.extend_from_slice(&point[*offset..*offset + *size]);
                }
            }
            Self::Integer {
                offset,
                field_type,
                previous,
            } => match *field_type {
                FieldType::Int16 => {
                    encode_integer_points::<3>(cloud_data, point_step, output, *offset, previous)
                }
                FieldType::Uint16 => {
                    encode_integer_points::<4>(cloud_data, point_step, output, *offset, previous)
                }
                FieldType::Int32 => {
                    encode_integer_points::<5>(cloud_data, point_step, output, *offset, previous)
                }
                FieldType::Uint32 => {
                    encode_integer_points::<6>(cloud_data, point_step, output, *offset, previous)
                }
                FieldType::Int64 => {
                    encode_integer_points::<9>(cloud_data, point_step, output, *offset, previous)
                }
                FieldType::Uint64 => {
                    encode_integer_points::<10>(cloud_data, point_step, output, *offset, previous)
                }
                _ => unreachable!("integer field encoders are 16, 32, or 64 bit"),
            },
            Self::Float32Lossy {
                offset,
                multiplier,
                previous,
            } => {
                for point in cloud_data.chunks_exact(point_step) {
                    let value = read_f32(point, *offset);
                    if !value.is_finite() {
                        output.push(0);
                        *previous = 0;
                    } else {
                        let quantized = (value * *multiplier).round() as i64;
                        let delta = representable_lossy_delta(quantized, *previous);
                        varint::append(delta, output);
                        *previous = previous.wrapping_add(delta);
                    }
                }
            }
            Self::Float64Lossy {
                offset,
                multiplier,
                previous,
            } => {
                for point in cloud_data.chunks_exact(point_step) {
                    let value = read_f64(point, *offset);
                    if !value.is_finite() {
                        output.push(0);
                        *previous = 0;
                    } else {
                        let quantized = (value * *multiplier).round() as i64;
                        let delta = representable_lossy_delta(quantized, *previous);
                        varint::append(delta, output);
                        *previous = previous.wrapping_add(delta);
                    }
                }
            }
            Self::Float32Xor { offset, previous } => {
                for point in cloud_data.chunks_exact(point_step) {
                    let current =
                        u32::from_le_bytes(point[*offset..*offset + 4].try_into().unwrap());
                    output.extend_from_slice(&(current ^ *previous).to_le_bytes());
                    *previous = current;
                }
            }
            Self::Float64Xor { offset, previous } => {
                for point in cloud_data.chunks_exact(point_step) {
                    let current =
                        u64::from_le_bytes(point[*offset..*offset + 8].try_into().unwrap());
                    output.extend_from_slice(&(current ^ *previous).to_le_bytes());
                    *previous = current;
                }
            }
            Self::Float32VectorLossy {
                offsets,
                multipliers,
                previous,
                count,
            } => match *count {
                3 if offsets[..3] == [0, 4, 8] => encode_contiguous_xyz_points(
                    cloud_data,
                    point_step,
                    output,
                    multipliers,
                    previous,
                ),
                3 => encode_float32_vector_points::<3>(
                    cloud_data,
                    point_step,
                    output,
                    offsets,
                    multipliers,
                    previous,
                ),
                _ => {
                    for point in cloud_data.chunks_exact(point_step) {
                        for index in 0..*count {
                            let value = read_f32(point, offsets[index]);
                            if !value.is_finite() {
                                output.push(0);
                                previous[index] = 0;
                            } else {
                                let quantized =
                                    (value * multipliers[index]).round_ties_even() as i32;
                                varint::append(
                                    quantized.wrapping_sub(previous[index]) as i64,
                                    output,
                                );
                                previous[index] = quantized;
                            }
                        }
                    }
                }
            },
            Self::Float64Gorilla { .. } => return false,
        }
        true
    }
}

pub enum FieldDecoder {
    Copy {
        offset: usize,
        size: usize,
    },
    Integer {
        offset: usize,
        size: usize,
        previous: i64,
    },
    Float32Lossy {
        offset: usize,
        resolution: f32,
        previous: i64,
    },
    Float64Lossy {
        offset: usize,
        resolution: f64,
        previous: i64,
    },
    Float32Xor {
        offset: usize,
        previous: u32,
    },
    Float64Xor {
        offset: usize,
        previous: u64,
    },
    Float64Gorilla {
        offset: usize,
        previous: u64,
        previous_leading: u8,
        previous_trailing: u8,
        is_first: bool,
    },
    Float32VectorLossy {
        offsets: [usize; 4],
        resolutions: [f32; 4],
        previous: [i32; 4],
        count: usize,
    },
}

impl FieldDecoder {
    #[inline(always)]
    pub fn reset(&mut self) {
        match self {
            Self::Copy { .. } => {}
            Self::Integer { previous, .. }
            | Self::Float32Lossy { previous, .. }
            | Self::Float64Lossy { previous, .. } => *previous = 0,
            Self::Float32Xor { previous, .. } => *previous = 0,
            Self::Float64Xor { previous, .. } => *previous = 0,
            Self::Float64Gorilla {
                previous,
                previous_leading,
                previous_trailing,
                is_first,
                ..
            } => {
                *previous = 0;
                *previous_leading = u8::MAX;
                *previous_trailing = 0;
                *is_first = true;
            }
            Self::Float32VectorLossy { previous, .. } => *previous = [0; 4],
        }
    }

    #[inline(always)]
    pub fn decode(
        &mut self,
        input: &[u8],
        position: &mut usize,
        point: &mut [u8],
    ) -> Result<(), String> {
        match self {
            Self::Copy { offset, size } => {
                let end = position
                    .checked_add(*size)
                    .ok_or_else(|| "CloudINI copy field length overflow".to_string())?;
                let value = input
                    .get(*position..end)
                    .ok_or_else(|| "Truncated CloudINI copy field".to_string())?;
                point[*offset..*offset + *size].copy_from_slice(value);
                *position = end;
                Ok(())
            }
            Self::Integer {
                offset,
                size,
                previous,
            } => {
                let delta = varint::decode_at(input, position)?;
                let value = previous.wrapping_add(delta);
                *previous = value;
                point[*offset..*offset + *size].copy_from_slice(&value.to_le_bytes()[..*size]);
                Ok(())
            }
            Self::Float32Lossy {
                offset,
                resolution,
                previous,
            } => {
                if input.get(*position) == Some(&0) {
                    point[*offset..*offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
                    *previous = 0;
                    *position += 1;
                    return Ok(());
                }
                let delta = varint::decode_at(input, position)?;
                *previous = previous.wrapping_add(delta);
                let value = *previous as f32 * *resolution;
                point[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
                Ok(())
            }
            Self::Float64Lossy {
                offset,
                resolution,
                previous,
            } => {
                if input.get(*position) == Some(&0) {
                    point[*offset..*offset + 8].copy_from_slice(&f64::NAN.to_le_bytes());
                    *previous = 0;
                    *position += 1;
                    return Ok(());
                }
                let delta = varint::decode_at(input, position)?;
                *previous = previous.wrapping_add(delta);
                let value = *previous as f64 * *resolution;
                point[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
                Ok(())
            }
            Self::Float32Xor { offset, previous } => {
                let end = position
                    .checked_add(4)
                    .ok_or_else(|| "CloudINI float32 XOR length overflow".to_string())?;
                let residual = u32::from_le_bytes(
                    input
                        .get(*position..end)
                        .ok_or_else(|| "Truncated CloudINI float32 XOR".to_string())?
                        .try_into()
                        .unwrap(),
                );
                *position = end;
                *previous ^= residual;
                point[*offset..*offset + 4].copy_from_slice(&previous.to_le_bytes());
                Ok(())
            }
            Self::Float64Xor { offset, previous } => {
                let end = position
                    .checked_add(8)
                    .ok_or_else(|| "CloudINI float64 XOR length overflow".to_string())?;
                let residual = u64::from_le_bytes(
                    input
                        .get(*position..end)
                        .ok_or_else(|| "Truncated CloudINI float64 XOR".to_string())?
                        .try_into()
                        .unwrap(),
                );
                *position = end;
                *previous ^= residual;
                point[*offset..*offset + 8].copy_from_slice(&previous.to_le_bytes());
                Ok(())
            }
            Self::Float64Gorilla {
                offset,
                previous,
                previous_leading,
                previous_trailing,
                is_first,
            } => {
                let mut reader = BitReader::new(
                    input
                        .get(*position..)
                        .ok_or_else(|| "Truncated CloudINI Gorilla field".to_string())?,
                );
                let current = if *is_first {
                    *is_first = false;
                    reader.read(64)?
                } else if reader.read(1)? == 0 {
                    *previous
                } else {
                    let residual = if reader.read(1)? == 0 {
                        if *previous_leading == u8::MAX {
                            return Err(
                                "CloudINI Gorilla window reused before initialization".to_string()
                            );
                        }
                        let meaningful = 64_u8
                            .checked_sub(*previous_leading)
                            .and_then(|value| value.checked_sub(*previous_trailing))
                            .ok_or_else(|| "CloudINI Gorilla window exceeds 64 bits".to_string())?;
                        reader.read(meaningful)? << *previous_trailing
                    } else {
                        let leading = reader.read(5)? as u8;
                        let meaningful = reader.read(6)? as u8 + 1;
                        let trailing = 64_u8
                            .checked_sub(leading)
                            .and_then(|value| value.checked_sub(meaningful))
                            .ok_or_else(|| "CloudINI Gorilla window exceeds 64 bits".to_string())?;
                        *previous_leading = leading;
                        *previous_trailing = trailing;
                        reader.read(meaningful)? << trailing
                    };
                    *previous ^ residual
                };
                *previous = current;
                point[*offset..*offset + 8].copy_from_slice(&current.to_le_bytes());
                *position += reader.bytes_consumed();
                Ok(())
            }
            Self::Float32VectorLossy {
                offsets,
                resolutions,
                previous,
                count,
            } => {
                for index in 0..*count {
                    let first = *input
                        .get(*position)
                        .ok_or_else(|| "Truncated CloudINI float32 vector field".to_string())?;
                    if first == 0 {
                        previous[index] = 0;
                        point[offsets[index]..offsets[index] + 4]
                            .copy_from_slice(&f32::NAN.to_le_bytes());
                        *position += 1;
                        continue;
                    }
                    let delta = varint::decode_at(input, position)?;
                    previous[index] = previous[index].wrapping_add(delta as i32);
                    let value = previous[index] as f32 * resolutions[index];
                    point[offsets[index]..offsets[index] + 4].copy_from_slice(&value.to_le_bytes());
                }
                Ok(())
            }
        }
    }

    pub fn decode_points(
        &mut self,
        input: &[u8],
        position: &mut usize,
        output: &mut [u8],
        point_step: usize,
    ) -> Result<bool, String> {
        match self {
            Self::Copy { offset, size } => {
                for point in output.chunks_exact_mut(point_step) {
                    let end = position
                        .checked_add(*size)
                        .ok_or_else(|| "CloudINI copy field length overflow".to_string())?;
                    let value = input
                        .get(*position..end)
                        .ok_or_else(|| "Truncated CloudINI copy field".to_string())?;
                    point[*offset..*offset + *size].copy_from_slice(value);
                    *position = end;
                }
            }
            Self::Integer {
                offset,
                size,
                previous,
            } => match *size {
                2 => decode_integer_points::<2>(
                    input, position, output, point_step, *offset, previous,
                )?,
                4 => decode_integer_points::<4>(
                    input, position, output, point_step, *offset, previous,
                )?,
                8 => decode_integer_points::<8>(
                    input, position, output, point_step, *offset, previous,
                )?,
                _ => unreachable!("integer field decoders are 16, 32, or 64 bit"),
            },
            Self::Float32Lossy {
                offset,
                resolution,
                previous,
            } => {
                for point in output.chunks_exact_mut(point_step) {
                    let value = if input.get(*position) == Some(&0) {
                        *previous = 0;
                        *position += 1;
                        f32::NAN
                    } else {
                        *previous = previous.wrapping_add(varint::decode_at(input, position)?);
                        *previous as f32 * *resolution
                    };
                    point[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            Self::Float64Lossy {
                offset,
                resolution,
                previous,
            } => {
                for point in output.chunks_exact_mut(point_step) {
                    let value = if input.get(*position) == Some(&0) {
                        *previous = 0;
                        *position += 1;
                        f64::NAN
                    } else {
                        *previous = previous.wrapping_add(varint::decode_at(input, position)?);
                        *previous as f64 * *resolution
                    };
                    point[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
                }
            }
            Self::Float32Xor { offset, previous } => {
                for point in output.chunks_exact_mut(point_step) {
                    let end = position
                        .checked_add(4)
                        .ok_or_else(|| "CloudINI float32 XOR length overflow".to_string())?;
                    let residual = u32::from_le_bytes(
                        input
                            .get(*position..end)
                            .ok_or_else(|| "Truncated CloudINI float32 XOR".to_string())?
                            .try_into()
                            .unwrap(),
                    );
                    *position = end;
                    *previous ^= residual;
                    point[*offset..*offset + 4].copy_from_slice(&previous.to_le_bytes());
                }
            }
            Self::Float64Xor { offset, previous } => {
                for point in output.chunks_exact_mut(point_step) {
                    let end = position
                        .checked_add(8)
                        .ok_or_else(|| "CloudINI float64 XOR length overflow".to_string())?;
                    let residual = u64::from_le_bytes(
                        input
                            .get(*position..end)
                            .ok_or_else(|| "Truncated CloudINI float64 XOR".to_string())?
                            .try_into()
                            .unwrap(),
                    );
                    *position = end;
                    *previous ^= residual;
                    point[*offset..*offset + 8].copy_from_slice(&previous.to_le_bytes());
                }
            }
            Self::Float32VectorLossy {
                offsets,
                resolutions,
                previous,
                count,
            } => {
                if *count == 3 && offsets[..3] == [0, 4, 8] {
                    decode_contiguous_xyz_points(
                        input,
                        position,
                        output,
                        point_step,
                        resolutions,
                        previous,
                    )?;
                    return Ok(true);
                }
                for point in output.chunks_exact_mut(point_step) {
                    for index in 0..*count {
                        let value = if input.get(*position) == Some(&0) {
                            previous[index] = 0;
                            *position += 1;
                            f32::NAN
                        } else {
                            previous[index] = previous[index]
                                .wrapping_add(varint::decode_at(input, position)? as i32);
                            previous[index] as f32 * resolutions[index]
                        };
                        point[offsets[index]..offsets[index] + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            Self::Float64Gorilla { .. } => return Ok(false),
        }
        Ok(true)
    }
}

#[inline(never)]
fn decode_contiguous_xyz_points(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    point_step: usize,
    resolutions: &[f32; 4],
    previous: &mut [i32; 4],
) -> Result<(), String> {
    let mut cursor = *position;
    let mut previous_x = previous[0];
    let mut previous_y = previous[1];
    let mut previous_z = previous[2];
    let resolution_x = resolutions[0];
    let resolution_y = resolutions[1];
    let resolution_z = resolutions[2];
    for point in output.chunks_exact_mut(point_step) {
        let x = decode_vector_component(input, &mut cursor, &mut previous_x, resolution_x)?;
        let y = decode_vector_component(input, &mut cursor, &mut previous_y, resolution_y)?;
        let z = decode_vector_component(input, &mut cursor, &mut previous_z, resolution_z)?;
        // The validated 0/4/8 offsets prove that every point contains twelve bytes.
        unsafe {
            let destination = point.as_mut_ptr();
            destination
                .cast::<u32>()
                .write_unaligned(x.to_bits().to_le());
            destination
                .add(4)
                .cast::<u32>()
                .write_unaligned(y.to_bits().to_le());
            destination
                .add(8)
                .cast::<u32>()
                .write_unaligned(z.to_bits().to_le());
        }
    }
    *position = cursor;
    previous[0] = previous_x;
    previous[1] = previous_y;
    previous[2] = previous_z;
    Ok(())
}

pub fn encode_multi_points(
    encoders: &mut [FieldEncoder],
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
) -> bool {
    if point_step == 12
        && let [
            FieldEncoder::Float32Xor {
                offset: offset_0,
                previous: previous_0,
            },
            FieldEncoder::Float32Xor {
                offset: offset_1,
                previous: previous_1,
            },
            FieldEncoder::Float32Xor {
                offset: offset_2,
                previous: previous_2,
            },
        ] = encoders
        && [*offset_0, *offset_1, *offset_2] == [0, 4, 8]
    {
        [*previous_0, *previous_1, *previous_2] =
            encode_contiguous_xyz_xor(cloud_data, output, [*previous_0, *previous_1, *previous_2]);
        return true;
    }

    if !encoders.is_empty()
        && encoders
            .iter()
            .all(|encoder| matches!(encoder, FieldEncoder::Float32Xor { .. }))
    {
        let mut xor_fields = encoders
            .iter()
            .map(|encoder| match encoder {
                FieldEncoder::Float32Xor { offset, previous } => (*offset, *previous),
                _ => unreachable!("all encoders were checked as float32 XOR"),
            })
            .collect::<Vec<_>>();
        for point in cloud_data.chunks_exact(point_step) {
            for (offset, previous) in &mut xor_fields {
                let current = u32::from_le_bytes(point[*offset..*offset + 4].try_into().unwrap());
                output.extend_from_slice(&(current ^ *previous).to_le_bytes());
                *previous = current;
            }
        }
        for (encoder, (_, value)) in encoders.iter_mut().zip(xor_fields) {
            if let FieldEncoder::Float32Xor { previous, .. } = encoder {
                *previous = value;
            }
        }
        return true;
    }

    let Some(FieldEncoder::Float32VectorLossy {
        offsets,
        multipliers,
        previous,
        count,
    }) = encoders.first()
    else {
        return false;
    };
    let mut vector_state = VectorEncodingState {
        offsets: *offsets,
        multipliers: *multipliers,
        previous: *previous,
        count: *count,
    };
    if !encoders[1..]
        .iter()
        .all(|encoder| matches!(encoder, FieldEncoder::Integer { .. }))
    {
        return false;
    }
    let integer_count = encoders.len().saturating_sub(1);
    let mut stack_fields = [(0, FieldType::Unknown, 0); 4];
    let mut heap_fields = Vec::new();
    let integer_fields = if integer_count <= stack_fields.len() {
        for (state, encoder) in stack_fields.iter_mut().zip(&encoders[1..]) {
            if let FieldEncoder::Integer {
                offset,
                field_type,
                previous,
            } = encoder
            {
                *state = (*offset, *field_type, *previous);
            }
        }
        &mut stack_fields[..integer_count]
    } else {
        heap_fields.reserve(integer_count);
        for encoder in &encoders[1..] {
            if let FieldEncoder::Integer {
                offset,
                field_type,
                previous,
            } = encoder
            {
                heap_fields.push((*offset, *field_type, *previous));
            }
        }
        &mut heap_fields[..]
    };
    if integer_fields.is_empty() {
        return false;
    }

    let common_integer_type = integer_fields.first().map(|(_, field_type, _)| *field_type);
    let has_common_integer_type = common_integer_type.is_some_and(|common| {
        integer_fields
            .iter()
            .all(|(_, field_type, _)| *field_type == common)
    });
    match common_integer_type.filter(|_| has_common_integer_type) {
        Some(FieldType::Int16) => encode_vector_integer_points_dispatch::<3>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        Some(FieldType::Uint16) => encode_vector_integer_points_dispatch::<4>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        Some(FieldType::Int32) => encode_vector_integer_points_dispatch::<5>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        Some(FieldType::Uint32) => encode_vector_integer_points_dispatch::<6>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        Some(FieldType::Int64) => encode_vector_integer_points_dispatch::<9>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        Some(FieldType::Uint64) => encode_vector_integer_points_dispatch::<10>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
        _ => encode_vector_integer_points_dispatch::<0>(
            cloud_data,
            point_step,
            output,
            &mut vector_state,
            integer_fields,
        ),
    }

    if let FieldEncoder::Float32VectorLossy { previous, .. } = &mut encoders[0] {
        *previous = vector_state.previous;
    }
    for (encoder, &(_, _, value)) in encoders[1..].iter_mut().zip(integer_fields.iter()) {
        if let FieldEncoder::Integer { previous, .. } = encoder {
            *previous = value;
        }
    }
    true
}

#[inline(never)]
pub(crate) fn encode_contiguous_xyz_xor(
    cloud_data: &[u8],
    output: &mut Vec<u8>,
    previous: [u32; 3],
) -> [u32; 3] {
    if output.capacity() - output.len() < cloud_data.len() {
        output.reserve(cloud_data.len());
    }
    let start = output.len();
    // The reserve above provides one output byte for every input byte.
    let previous = unsafe {
        encode_contiguous_xyz_xor_raw(cloud_data, output.as_mut_ptr().add(start), previous)
    };
    unsafe {
        output.set_len(start + cloud_data.len());
    }
    previous
}

pub(crate) fn encode_contiguous_xyz_xor_into(
    cloud_data: &[u8],
    output: &mut [u8],
    previous: [u32; 3],
) -> [u32; 3] {
    debug_assert_eq!(cloud_data.len(), output.len());
    // The output slice is exactly as long as the input.
    unsafe { encode_contiguous_xyz_xor_raw(cloud_data, output.as_mut_ptr(), previous) }
}

unsafe fn encode_contiguous_xyz_xor_raw(
    cloud_data: &[u8],
    destination: *mut u8,
    mut previous: [u32; 3],
) -> [u32; 3] {
    for (point_index, point) in cloud_data.chunks_exact(12).enumerate() {
        let source = point.as_ptr();
        let current_0 = u32::from_le(unsafe { source.cast::<u32>().read_unaligned() });
        let current_1 = u32::from_le(unsafe { source.add(4).cast::<u32>().read_unaligned() });
        let current_2 = u32::from_le(unsafe { source.add(8).cast::<u32>().read_unaligned() });
        let point_destination = unsafe { destination.add(point_index * 12) };
        unsafe {
            point_destination
                .cast::<u32>()
                .write_unaligned((current_0 ^ previous[0]).to_le());
            point_destination
                .add(4)
                .cast::<u32>()
                .write_unaligned((current_1 ^ previous[1]).to_le());
            point_destination
                .add(8)
                .cast::<u32>()
                .write_unaligned((current_2 ^ previous[2]).to_le());
        }
        previous = [current_0, current_1, current_2];
    }
    previous
}

pub fn decode_multi_points(
    decoders: &mut [FieldDecoder],
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    point_step: usize,
) -> Result<bool, String> {
    if point_step == 12
        && let [
            FieldDecoder::Float32Xor {
                offset: offset_0,
                previous: previous_0,
            },
            FieldDecoder::Float32Xor {
                offset: offset_1,
                previous: previous_1,
            },
            FieldDecoder::Float32Xor {
                offset: offset_2,
                previous: previous_2,
            },
        ] = decoders
        && [*offset_0, *offset_1, *offset_2] == [0, 4, 8]
    {
        [*previous_0, *previous_1, *previous_2] = decode_contiguous_xyz_xor(
            input,
            position,
            output,
            [*previous_0, *previous_1, *previous_2],
        )?;
        return Ok(true);
    }

    let mut xor_fields = Vec::with_capacity(decoders.len());
    for decoder in decoders.iter() {
        if let FieldDecoder::Float32Xor { offset, previous } = decoder {
            xor_fields.push((*offset, *previous));
        } else {
            xor_fields.clear();
            break;
        }
    }
    if !xor_fields.is_empty() {
        for point in output.chunks_exact_mut(point_step) {
            for (offset, previous) in &mut xor_fields {
                let end = position
                    .checked_add(4)
                    .ok_or_else(|| "CloudINI float32 XOR length overflow".to_string())?;
                let residual = u32::from_le_bytes(
                    input
                        .get(*position..end)
                        .ok_or_else(|| "Truncated CloudINI float32 XOR".to_string())?
                        .try_into()
                        .unwrap(),
                );
                *position = end;
                *previous ^= residual;
                point[*offset..*offset + 4].copy_from_slice(&previous.to_le_bytes());
            }
        }
        for (decoder, (_, value)) in decoders.iter_mut().zip(xor_fields) {
            if let FieldDecoder::Float32Xor { previous, .. } = decoder {
                *previous = value;
            }
        }
        return Ok(true);
    }

    let Some(FieldDecoder::Float32VectorLossy {
        offsets,
        resolutions,
        previous,
        count,
    }) = decoders.first()
    else {
        return Ok(false);
    };
    let mut vector_state = VectorDecodingState {
        offsets: *offsets,
        resolutions: *resolutions,
        previous: *previous,
        count: *count,
    };
    let mut integer_fields = Vec::with_capacity(decoders.len().saturating_sub(1));
    for decoder in &decoders[1..] {
        if let FieldDecoder::Integer {
            offset,
            size,
            previous,
        } = decoder
        {
            integer_fields.push((*offset, *size, *previous));
        } else {
            return Ok(false);
        }
    }
    if integer_fields.is_empty() {
        return Ok(false);
    }

    let common_integer_size = integer_fields.first().map(|(_, size, _)| *size);
    let has_common_integer_size = common_integer_size
        .is_some_and(|common| integer_fields.iter().all(|(_, size, _)| *size == common));
    match common_integer_size.filter(|_| has_common_integer_size) {
        Some(2) => decode_vector_integer_points::<2>(
            input,
            position,
            output,
            point_step,
            &mut vector_state,
            &mut integer_fields,
        )?,
        Some(4) => decode_vector_integer_points::<4>(
            input,
            position,
            output,
            point_step,
            &mut vector_state,
            &mut integer_fields,
        )?,
        Some(8) => decode_vector_integer_points::<8>(
            input,
            position,
            output,
            point_step,
            &mut vector_state,
            &mut integer_fields,
        )?,
        _ => decode_vector_integer_points::<0>(
            input,
            position,
            output,
            point_step,
            &mut vector_state,
            &mut integer_fields,
        )?,
    }

    if let FieldDecoder::Float32VectorLossy { previous, .. } = &mut decoders[0] {
        *previous = vector_state.previous;
    }
    for (decoder, (_, _, value)) in decoders[1..].iter_mut().zip(integer_fields) {
        if let FieldDecoder::Integer { previous, .. } = decoder {
            *previous = value;
        }
    }
    Ok(true)
}

#[inline(never)]
fn decode_contiguous_xyz_xor(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    mut previous: [u32; 3],
) -> Result<[u32; 3], String> {
    let end = position
        .checked_add(output.len())
        .ok_or_else(|| "CloudINI float32 XOR length overflow".to_string())?;
    let encoded = input
        .get(*position..end)
        .ok_or_else(|| "Truncated CloudINI float32 XOR".to_string())?;
    // The validated equal-sized slices provide twelve bytes for every point.
    unsafe {
        let source = encoded.as_ptr();
        let destination = output.as_mut_ptr();
        for point_index in 0..output.len() / 12 {
            let point_source = source.add(point_index * 12);
            let residual_0 = u32::from_le(point_source.cast::<u32>().read_unaligned());
            let residual_1 = u32::from_le(point_source.add(4).cast::<u32>().read_unaligned());
            let residual_2 = u32::from_le(point_source.add(8).cast::<u32>().read_unaligned());
            previous[0] ^= residual_0;
            previous[1] ^= residual_1;
            previous[2] ^= residual_2;
            let point_destination = destination.add(point_index * 12);
            point_destination
                .cast::<u32>()
                .write_unaligned(previous[0].to_le());
            point_destination
                .add(4)
                .cast::<u32>()
                .write_unaligned(previous[1].to_le());
            point_destination
                .add(8)
                .cast::<u32>()
                .write_unaligned(previous[2].to_le());
        }
    }
    *position = end;
    Ok(previous)
}

#[inline(always)]
fn decode_integer_points<const SIZE: usize>(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    point_step: usize,
    offset: usize,
    previous: &mut i64,
) -> Result<(), String> {
    if SIZE == 4 && point_step == 4 && offset == 0 {
        return decode_integer_points_dense_u32(input, position, output, previous);
    }

    for point in output.chunks_exact_mut(point_step) {
        *previous = previous.wrapping_add(varint::decode_at(input, position)?);
        point[offset..offset + SIZE].copy_from_slice(&previous.to_le_bytes()[..SIZE]);
    }
    Ok(())
}

#[inline(never)]
fn decode_integer_points_dense_u32(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    previous: &mut i64,
) -> Result<(), String> {
    let mut cursor = *position;
    let mut value = *previous as u32;
    for bytes in output.chunks_exact_mut(4) {
        value = value.wrapping_add(varint::decode_at(input, &mut cursor)? as u32);
        // Every exact chunk has space for one unaligned u32 value.
        unsafe {
            bytes
                .as_mut_ptr()
                .cast::<u32>()
                .write_unaligned(value.to_le());
        }
    }
    *position = cursor;
    *previous = i64::from(value);
    Ok(())
}

#[inline(always)]
fn encode_float32_vector_points<const COUNT: usize>(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    offsets: &[usize; 4],
    multipliers: &[f32; 4],
    previous: &mut [i32; 4],
) {
    for point in cloud_data.chunks_exact(point_step) {
        for index in 0..COUNT {
            let value = read_f32(point, offsets[index]);
            if !value.is_finite() {
                output.push(0);
                previous[index] = 0;
            } else {
                let quantized = (value * multipliers[index]).round_ties_even() as i32;
                varint::append(quantized.wrapping_sub(previous[index]) as i64, output);
                previous[index] = quantized;
            }
        }
    }
}

#[inline(always)]
fn encode_contiguous_xyz_points(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    multipliers: &[f32; 4],
    previous: &mut [i32; 4],
) {
    for point in cloud_data.chunks_exact(point_step) {
        // The validated 0/4/8 offsets prove that every point contains twelve bytes.
        let pointer = point.as_ptr();
        let values = unsafe {
            [
                f32::from_bits(u32::from_le(pointer.cast::<u32>().read_unaligned())),
                f32::from_bits(u32::from_le(pointer.add(4).cast::<u32>().read_unaligned())),
                f32::from_bits(u32::from_le(pointer.add(8).cast::<u32>().read_unaligned())),
            ]
        };
        let mut deltas = [0_i64; 3];
        let mut valid_mask = 0_u64;
        for index in 0..3 {
            let value = values[index];
            if !value.is_finite() {
                previous[index] = 0;
            } else {
                let quantized = (value * multipliers[index]).round_ties_even() as i32;
                deltas[index] = quantized.wrapping_sub(previous[index]) as i64;
                valid_mask |= 1 << index;
                previous[index] = quantized;
            }
        }
        varint::append_batch(deltas, valid_mask, output);
    }
}

#[inline(always)]
fn encode_integer_points<const KIND: u8>(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    offset: usize,
    previous: &mut i64,
) {
    if KIND == 6 && point_step == 4 && offset == 0 {
        encode_dense_u32_points(cloud_data, output, previous);
        return;
    }

    for point in cloud_data.chunks_exact(point_step) {
        let value = read_integer_kind::<KIND>(point, offset);
        varint::append(value.wrapping_sub(*previous), output);
        *previous = value;
    }
}

#[inline(never)]
pub(crate) fn encode_dense_u32_points(cloud_data: &[u8], output: &mut Vec<u8>, previous: &mut i64) {
    let mut groups = cloud_data.chunks_exact(32);
    for group in &mut groups {
        // Every exact chunk contains eight little-endian u32 values.
        let values = unsafe {
            group
                .as_ptr()
                .cast::<[u32; 8]>()
                .read_unaligned()
                .map(u32::from_le)
                .map(i64::from)
        };
        varint::append_batch(
            [
                values[0].wrapping_sub(*previous),
                values[1].wrapping_sub(values[0]),
                values[2].wrapping_sub(values[1]),
                values[3].wrapping_sub(values[2]),
                values[4].wrapping_sub(values[3]),
                values[5].wrapping_sub(values[4]),
                values[6].wrapping_sub(values[5]),
                values[7].wrapping_sub(values[6]),
            ],
            0xff,
            output,
        );
        *previous = values[7];
    }
    for point in groups.remainder().chunks_exact(4) {
        let value = i64::from(u32::from_le_bytes(point.try_into().unwrap()));
        varint::append(value.wrapping_sub(*previous), output);
        *previous = value;
    }
}

struct VectorEncodingState {
    offsets: [usize; 4],
    multipliers: [f32; 4],
    previous: [i32; 4],
    count: usize,
}

struct VectorDecodingState {
    offsets: [usize; 4],
    resolutions: [f32; 4],
    previous: [i32; 4],
    count: usize,
}

#[inline(always)]
fn encode_vector_integer_points_dispatch<const INTEGER_KIND: u8>(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    vector: &mut VectorEncodingState,
    integer_fields: &mut [(usize, FieldType, i64)],
) {
    if vector.count == 3 {
        encode_vector_integer_points::<INTEGER_KIND, 3>(
            cloud_data,
            point_step,
            output,
            vector,
            integer_fields,
        );
    } else {
        encode_vector_integer_points::<INTEGER_KIND, 0>(
            cloud_data,
            point_step,
            output,
            vector,
            integer_fields,
        );
    }
}

#[inline(always)]
fn encode_vector_integer_points<const INTEGER_KIND: u8, const VECTOR_COUNT: usize>(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    vector: &mut VectorEncodingState,
    integer_fields: &mut [(usize, FieldType, i64)],
) {
    if VECTOR_COUNT == 3 && vector.offsets[..3] == [0, 4, 8] {
        if integer_fields.len() == 2 {
            encode_canonical_xyz_two_integers::<INTEGER_KIND>(
                cloud_data,
                point_step,
                output,
                vector,
                integer_fields,
            );
            return;
        }
        for point in cloud_data.chunks_exact(point_step) {
            // The validated 0/4/8 offsets prove that every point contains twelve bytes.
            let pointer = point.as_ptr();
            let values = unsafe {
                [
                    f32::from_bits(u32::from_le(pointer.cast::<u32>().read_unaligned())),
                    f32::from_bits(u32::from_le(pointer.add(4).cast::<u32>().read_unaligned())),
                    f32::from_bits(u32::from_le(pointer.add(8).cast::<u32>().read_unaligned())),
                ]
            };
            let mut deltas = [0_i64; 3];
            let mut valid_mask = 0_u64;
            for (index, value) in values.into_iter().enumerate() {
                if !value.is_finite() {
                    vector.previous[index] = 0;
                } else {
                    let quantized = (value * vector.multipliers[index]).round_ties_even() as i32;
                    deltas[index] = quantized.wrapping_sub(vector.previous[index]) as i64;
                    valid_mask |= 1 << index;
                    vector.previous[index] = quantized;
                }
            }
            varint::append_batch(deltas, valid_mask, output);
            for (offset, field_type, previous) in integer_fields.iter_mut() {
                let value = if INTEGER_KIND == 0 {
                    read_integer(point, *offset, *field_type)
                } else {
                    read_integer_kind::<INTEGER_KIND>(point, *offset)
                };
                varint::append(value.wrapping_sub(*previous), output);
                *previous = value;
            }
        }
        return;
    }

    let count = if VECTOR_COUNT == 0 {
        vector.count
    } else {
        VECTOR_COUNT
    };
    for point in cloud_data.chunks_exact(point_step) {
        for index in 0..count {
            let value = read_f32(point, vector.offsets[index]);
            if !value.is_finite() {
                output.push(0);
                vector.previous[index] = 0;
            } else {
                let quantized = (value * vector.multipliers[index]).round_ties_even() as i32;
                varint::append(
                    quantized.wrapping_sub(vector.previous[index]) as i64,
                    output,
                );
                vector.previous[index] = quantized;
            }
        }
        for (offset, field_type, previous) in integer_fields.iter_mut() {
            let value = if INTEGER_KIND == 0 {
                read_integer(point, *offset, *field_type)
            } else {
                read_integer_kind::<INTEGER_KIND>(point, *offset)
            };
            varint::append(value.wrapping_sub(*previous), output);
            *previous = value;
        }
    }
}

#[inline(never)]
fn encode_canonical_xyz_two_integers<const INTEGER_KIND: u8>(
    cloud_data: &[u8],
    point_step: usize,
    output: &mut Vec<u8>,
    vector: &mut VectorEncodingState,
    integer_fields: &mut [(usize, FieldType, i64)],
) {
    let integer_offset_0 = integer_fields[0].0;
    let integer_offset_1 = integer_fields[1].0;
    let integer_type_0 = integer_fields[0].1;
    let integer_type_1 = integer_fields[1].1;
    let mut previous_integer_0 = integer_fields[0].2;
    let mut previous_integer_1 = integer_fields[1].2;
    for point in cloud_data.chunks_exact(point_step) {
        // The validated 0/4/8 offsets prove that every point contains twelve bytes.
        let pointer = point.as_ptr();
        let values = unsafe {
            [
                f32::from_bits(u32::from_le(pointer.cast::<u32>().read_unaligned())),
                f32::from_bits(u32::from_le(pointer.add(4).cast::<u32>().read_unaligned())),
                f32::from_bits(u32::from_le(pointer.add(8).cast::<u32>().read_unaligned())),
            ]
        };
        let mut deltas = [0_i64; 5];
        let mut valid_mask = 0b1_1000_u64;
        for (index, value) in values.into_iter().enumerate() {
            if !value.is_finite() {
                vector.previous[index] = 0;
            } else {
                let quantized = (value * vector.multipliers[index]).round_ties_even() as i32;
                deltas[index] = quantized.wrapping_sub(vector.previous[index]) as i64;
                valid_mask |= 1 << index;
                vector.previous[index] = quantized;
            }
        }
        let integer_0 = if INTEGER_KIND == 0 {
            read_integer(point, integer_offset_0, integer_type_0)
        } else {
            read_integer_kind::<INTEGER_KIND>(point, integer_offset_0)
        };
        let integer_1 = if INTEGER_KIND == 0 {
            read_integer(point, integer_offset_1, integer_type_1)
        } else {
            read_integer_kind::<INTEGER_KIND>(point, integer_offset_1)
        };
        deltas[3] = integer_0.wrapping_sub(previous_integer_0);
        deltas[4] = integer_1.wrapping_sub(previous_integer_1);
        previous_integer_0 = integer_0;
        previous_integer_1 = integer_1;
        varint::append_batch(deltas, valid_mask, output);
    }
    integer_fields[0].2 = previous_integer_0;
    integer_fields[1].2 = previous_integer_1;
}

#[inline(always)]
fn decode_vector_integer_points<const INTEGER_SIZE: usize>(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    point_step: usize,
    vector: &mut VectorDecodingState,
    integer_fields: &mut [(usize, usize, i64)],
) -> Result<(), String> {
    if INTEGER_SIZE == 2 && vector.count == 3 && vector.offsets[..3] == [0, 4, 8] {
        if integer_fields.len() == 2 {
            return decode_canonical_xyz_two_u16(
                input,
                position,
                output,
                point_step,
                vector,
                integer_fields,
            );
        }
        for point in output.chunks_exact_mut(point_step) {
            for index in 0..3 {
                let value = decode_vector_component(
                    input,
                    position,
                    &mut vector.previous[index],
                    vector.resolutions[index],
                )?;
                // Validated field offsets prove the destination is in this point.
                unsafe {
                    point
                        .as_mut_ptr()
                        .add(index * 4)
                        .cast::<u32>()
                        .write_unaligned(value.to_bits().to_le());
                }
            }
            for (offset, _, previous) in integer_fields.iter_mut() {
                *previous = previous.wrapping_add(varint::decode_at(input, position)?);
                // INTEGER_SIZE is two and every field offset was validated against point_step.
                unsafe {
                    point
                        .as_mut_ptr()
                        .add(*offset)
                        .cast::<u16>()
                        .write_unaligned((*previous as u16).to_le());
                }
            }
        }
        return Ok(());
    }

    for point in output.chunks_exact_mut(point_step) {
        for index in 0..vector.count {
            let value = decode_vector_component(
                input,
                position,
                &mut vector.previous[index],
                vector.resolutions[index],
            )?;
            point[vector.offsets[index]..vector.offsets[index] + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        for (offset, size, previous) in integer_fields.iter_mut() {
            *previous = previous.wrapping_add(varint::decode_at(input, position)?);
            if INTEGER_SIZE == 0 {
                point[*offset..*offset + *size].copy_from_slice(&previous.to_le_bytes()[..*size]);
            } else {
                point[*offset..*offset + INTEGER_SIZE]
                    .copy_from_slice(&previous.to_le_bytes()[..INTEGER_SIZE]);
            }
        }
    }
    Ok(())
}

#[inline(never)]
fn decode_canonical_xyz_two_u16(
    input: &[u8],
    position: &mut usize,
    output: &mut [u8],
    point_step: usize,
    vector: &mut VectorDecodingState,
    integer_fields: &mut [(usize, usize, i64)],
) -> Result<(), String> {
    let mut cursor = *position;
    let mut previous_x = vector.previous[0];
    let mut previous_y = vector.previous[1];
    let mut previous_z = vector.previous[2];
    let mut previous_integer_0 = integer_fields[0].2;
    let mut previous_integer_1 = integer_fields[1].2;
    let integer_offset_0 = integer_fields[0].0;
    let integer_offset_1 = integer_fields[1].0;
    for point in output.chunks_exact_mut(point_step) {
        let x =
            decode_vector_component(input, &mut cursor, &mut previous_x, vector.resolutions[0])?;
        let y =
            decode_vector_component(input, &mut cursor, &mut previous_y, vector.resolutions[1])?;
        let z =
            decode_vector_component(input, &mut cursor, &mut previous_z, vector.resolutions[2])?;
        previous_integer_0 =
            previous_integer_0.wrapping_add(varint::decode_at(input, &mut cursor)?);
        previous_integer_1 =
            previous_integer_1.wrapping_add(varint::decode_at(input, &mut cursor)?);
        // Validated field offsets prove every destination is in this point.
        unsafe {
            let destination = point.as_mut_ptr();
            destination
                .cast::<u32>()
                .write_unaligned(x.to_bits().to_le());
            destination
                .add(4)
                .cast::<u32>()
                .write_unaligned(y.to_bits().to_le());
            destination
                .add(8)
                .cast::<u32>()
                .write_unaligned(z.to_bits().to_le());
            destination
                .add(integer_offset_0)
                .cast::<u16>()
                .write_unaligned((previous_integer_0 as u16).to_le());
            destination
                .add(integer_offset_1)
                .cast::<u16>()
                .write_unaligned((previous_integer_1 as u16).to_le());
        }
    }
    *position = cursor;
    vector.previous[0] = previous_x;
    vector.previous[1] = previous_y;
    vector.previous[2] = previous_z;
    integer_fields[0].2 = previous_integer_0;
    integer_fields[1].2 = previous_integer_1;
    Ok(())
}

#[inline(always)]
fn decode_vector_component(
    input: &[u8],
    position: &mut usize,
    previous: &mut i32,
    resolution: f32,
) -> Result<f32, String> {
    if input.get(*position) == Some(&0) {
        *previous = 0;
        *position += 1;
        Ok(f32::NAN)
    } else {
        *previous = previous.wrapping_add(varint::decode_at(input, position)? as i32);
        Ok(*previous as f32 * resolution)
    }
}

#[inline(always)]
pub(crate) fn read_integer_kind<const KIND: u8>(point: &[u8], offset: usize) -> i64 {
    // Codec validation proves that the selected integer field fits in every point.
    unsafe {
        let pointer = point.as_ptr().add(offset);
        match KIND {
            3 => i16::from_le(pointer.cast::<i16>().read_unaligned()) as i64,
            4 => u16::from_le(pointer.cast::<u16>().read_unaligned()) as i64,
            5 => i32::from_le(pointer.cast::<i32>().read_unaligned()) as i64,
            6 => u32::from_le(pointer.cast::<u32>().read_unaligned()) as i64,
            9 => i64::from_le(pointer.cast::<i64>().read_unaligned()),
            10 => u64::from_le(pointer.cast::<u64>().read_unaligned()) as i64,
            _ => unreachable!("integer kind is selected from a validated field type"),
        }
    }
}

#[inline(always)]
fn representable_lossy_delta(value: i64, previous: i64) -> i64 {
    let delta = value.wrapping_sub(previous);
    if delta == i64::MIN {
        i64::MIN + 1
    } else {
        delta
    }
}

pub fn build_encoders(
    fields: &[PointField],
    encoding: EncodingOptions,
    version: u8,
) -> Vec<FieldEncoder> {
    if encoding == EncodingOptions::None {
        return fields
            .iter()
            .map(|field| FieldEncoder::Copy {
                offset: field.offset as usize,
                size: field.field_type.size_of(),
            })
            .collect();
    }

    let mut encoders = Vec::new();
    let mut start = 0;
    if encoding == EncodingOptions::Lossy {
        let count = leading_lossy_floats(fields);
        if count == 3 || count == 4 {
            let mut offsets = [0; 4];
            let mut multipliers = [0.0; 4];
            for index in 0..count {
                offsets[index] = fields[index].offset as usize;
                multipliers[index] = 1.0 / fields[index].resolution.unwrap();
            }
            encoders.push(FieldEncoder::Float32VectorLossy {
                offsets,
                multipliers,
                previous: [0; 4],
                count,
            });
            start = count;
        }
    }
    encoders.extend(
        fields[start..]
            .iter()
            .map(|field| encoder_for_field(field, encoding, version)),
    );
    encoders
}

pub fn build_decoders(
    fields: &[PointField],
    encoding: EncodingOptions,
    version: u8,
) -> Vec<FieldDecoder> {
    if encoding == EncodingOptions::None {
        return fields
            .iter()
            .map(|field| FieldDecoder::Copy {
                offset: field.offset as usize,
                size: field.field_type.size_of(),
            })
            .collect();
    }

    let mut decoders = Vec::new();
    let mut start = 0;
    if encoding == EncodingOptions::Lossy {
        let count = leading_lossy_floats(fields);
        if count == 3 || count == 4 {
            let mut offsets = [0; 4];
            let mut resolutions = [0.0; 4];
            for index in 0..count {
                offsets[index] = fields[index].offset as usize;
                resolutions[index] = fields[index].resolution.unwrap();
            }
            decoders.push(FieldDecoder::Float32VectorLossy {
                offsets,
                resolutions,
                previous: [0; 4],
                count,
            });
            start = count;
        }
    }
    decoders.extend(
        fields[start..]
            .iter()
            .map(|field| decoder_for_field(field, encoding, version)),
    );
    decoders
}

fn encoder_for_field(field: &PointField, encoding: EncodingOptions, version: u8) -> FieldEncoder {
    let offset = field.offset as usize;
    match field.field_type {
        FieldType::Float32 => match (encoding, field.resolution) {
            (EncodingOptions::Lossy, Some(resolution)) => FieldEncoder::Float32Lossy {
                offset,
                multiplier: (1.0_f64 / resolution as f64) as f32,
                previous: 0,
            },
            (EncodingOptions::Lossless, _) => FieldEncoder::Float32Xor {
                offset,
                previous: 0,
            },
            _ => FieldEncoder::Copy { offset, size: 4 },
        },
        FieldType::Float64 => {
            if encoding == EncodingOptions::Lossy
                && let Some(resolution) = field.resolution
            {
                FieldEncoder::Float64Lossy {
                    offset,
                    multiplier: 1.0 / resolution as f64,
                    previous: 0,
                }
            } else if field.resolution.is_none() && version >= 4 {
                FieldEncoder::Float64Gorilla {
                    offset,
                    previous: 0,
                    previous_leading: u8::MAX,
                    previous_trailing: 0,
                    is_first: true,
                }
            } else {
                FieldEncoder::Float64Xor {
                    offset,
                    previous: 0,
                }
            }
        }
        FieldType::Int8 | FieldType::Uint8 => FieldEncoder::Copy { offset, size: 1 },
        field_type => FieldEncoder::Integer {
            offset,
            field_type,
            previous: 0,
        },
    }
}

fn decoder_for_field(field: &PointField, encoding: EncodingOptions, version: u8) -> FieldDecoder {
    let offset = field.offset as usize;
    match field.field_type {
        FieldType::Float32 => match (encoding, field.resolution) {
            (EncodingOptions::Lossy, Some(resolution)) => FieldDecoder::Float32Lossy {
                offset,
                resolution,
                previous: 0,
            },
            (EncodingOptions::Lossless, _) => FieldDecoder::Float32Xor {
                offset,
                previous: 0,
            },
            (_, Some(resolution)) => FieldDecoder::Float32Lossy {
                offset,
                resolution,
                previous: 0,
            },
            _ => FieldDecoder::Copy { offset, size: 4 },
        },
        FieldType::Float64 => {
            if encoding == EncodingOptions::Lossy
                && let Some(resolution) = field.resolution
            {
                FieldDecoder::Float64Lossy {
                    offset,
                    resolution: resolution as f64,
                    previous: 0,
                }
            } else if field.resolution.is_some() && encoding != EncodingOptions::Lossless {
                FieldDecoder::Float64Lossy {
                    offset,
                    resolution: field.resolution.unwrap() as f64,
                    previous: 0,
                }
            } else if field.resolution.is_none() && version >= 4 {
                FieldDecoder::Float64Gorilla {
                    offset,
                    previous: 0,
                    previous_leading: u8::MAX,
                    previous_trailing: 0,
                    is_first: true,
                }
            } else {
                FieldDecoder::Float64Xor {
                    offset,
                    previous: 0,
                }
            }
        }
        FieldType::Int8 | FieldType::Uint8 => FieldDecoder::Copy { offset, size: 1 },
        field_type => FieldDecoder::Integer {
            offset,
            size: field_type.size_of(),
            previous: 0,
        },
    }
}

fn leading_lossy_floats(fields: &[PointField]) -> usize {
    fields
        .iter()
        .take_while(|field| field.field_type == FieldType::Float32 && field.resolution.is_some())
        .count()
}

fn read_integer(point: &[u8], offset: usize, field_type: FieldType) -> i64 {
    let bytes = &point[offset..];
    match field_type {
        FieldType::Int8 => bytes[0] as i8 as i64,
        FieldType::Uint8 => bytes[0] as i64,
        FieldType::Int16 => i16::from_le_bytes(bytes[..2].try_into().unwrap()) as i64,
        FieldType::Uint16 => u16::from_le_bytes(bytes[..2].try_into().unwrap()) as i64,
        FieldType::Int32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64,
        FieldType::Uint32 => u32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64,
        FieldType::Int64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()),
        FieldType::Uint64 => u64::from_le_bytes(bytes[..8].try_into().unwrap()) as i64,
        _ => unreachable!("integer encoder received a non-integer field"),
    }
}

fn read_f32(point: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(point[offset..offset + 4].try_into().unwrap())
}

fn read_f64(point: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes(point[offset..offset + 8].try_into().unwrap())
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    held: u8,
}

impl BitWriter {
    fn push(&mut self, mut value: u64, mut count: u8) {
        while count > 0 {
            let take = count.min(8 - self.held);
            let mask = if take == 8 {
                u8::MAX
            } else {
                (1_u8 << take) - 1
            };
            self.current |= (value as u8 & mask) << self.held;
            self.held += take;
            value = if take == 64 { 0 } else { value >> take };
            count -= take;
            if self.held == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.held = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.held > 0 {
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    input: &'a [u8],
    byte: usize,
    bit: u8,
}

impl<'a> BitReader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            byte: 0,
            bit: 0,
        }
    }

    fn read(&mut self, mut count: u8) -> Result<u64, String> {
        let mut value = 0_u64;
        let mut shift = 0_u8;
        while count > 0 {
            let byte = *self
                .input
                .get(self.byte)
                .ok_or_else(|| "Truncated CloudINI Gorilla field".to_string())?;
            let take = count.min(8 - self.bit);
            let mask = if take == 8 {
                u8::MAX
            } else {
                (1_u8 << take) - 1
            };
            value |= u64::from((byte >> self.bit) & mask) << shift;
            self.bit += take;
            shift += take;
            count -= take;
            if self.bit == 8 {
                self.byte += 1;
                self.bit = 0;
            }
        }
        Ok(value)
    }

    fn bytes_consumed(&self) -> usize {
        self.byte + usize::from(self.bit > 0)
    }
}
