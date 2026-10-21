//! Image normalization for tool results (Pi's `image-process.ts` / `image-resize-core.ts` /
//! `tool-result-images.ts`): convert unsupported formats to PNG, then shrink images that exceed
//! the inline limits. EXIF orientation is not applied.

use std::io::Cursor;

use base64::Engine;
use image::{DynamicImage, ImageFormat, imageops::FilterType};

/// Pi's `ImageResizeOptions` defaults: 2000x2000 and 4.5 MiB of base64.
#[derive(Debug, Clone, Copy)]
pub struct ResizeOptions {
    pub max_width: u32,
    pub max_height: u32,
    pub max_bytes: usize,
    pub jpeg_quality: u8,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            max_width: 2000,
            max_height: 2000,
            max_bytes: 4_718_592,
            jpeg_quality: 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessedImage {
    pub data: String,
    pub mime_type: String,
    pub hints: Vec<String>,
}

fn base_mime(mime: &str) -> String {
    mime.split(';').next().unwrap_or(mime).trim().to_lowercase()
}

fn supported_mime(mime: &str) -> Option<&'static str> {
    match base_mime(mime).as_str() {
        "image/png" => Some("image/png"),
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/gif" => Some("image/gif"),
        "image/webp" => Some("image/webp"),
        _ => None,
    }
}

fn encode(image: &DynamicImage, format: ImageFormat, quality: u8) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    match format {
        ImageFormat::Jpeg => {
            let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
            DynamicImage::ImageRgb8(image.to_rgb8())
                .write_with_encoder(encoder)
                .ok()?;
        }
        _ => image.write_to(&mut Cursor::new(&mut out), format).ok()?,
    }
    Some(out)
}

fn base64_len(bytes: usize) -> usize {
    bytes.div_ceil(3) * 4
}

struct Resized {
    bytes: Vec<u8>,
    mime_type: &'static str,
    original: (u32, u32),
    size: (u32, u32),
    was_resized: bool,
}

/// Pi's `resizeImageInProcess`: fit within the limits, trying PNG and JPEG qualities and
/// shrinking by 25% until under the byte limit. `None` when nothing fits.
fn resize(bytes: &[u8], mime: &'static str, options: ResizeOptions) -> Option<Resized> {
    let decoded = image::load_from_memory(bytes).ok()?;
    let (width, height) = (decoded.width(), decoded.height());
    if width <= options.max_width
        && height <= options.max_height
        && base64_len(bytes.len()) < options.max_bytes
    {
        return Some(Resized {
            bytes: bytes.to_vec(),
            mime_type: mime,
            original: (width, height),
            size: (width, height),
            was_resized: false,
        });
    }
    let (mut target_w, mut target_h) = (width, height);
    if target_w > options.max_width {
        target_h = (f64::from(target_h) * f64::from(options.max_width) / f64::from(target_w))
            .round() as u32;
        target_w = options.max_width;
    }
    if target_h > options.max_height {
        target_w = (f64::from(target_w) * f64::from(options.max_height) / f64::from(target_h))
            .round() as u32;
        target_h = options.max_height;
    }
    let mut qualities = vec![options.jpeg_quality, 85, 70, 55, 40];
    qualities.dedup();
    let (mut w, mut h) = (target_w.max(1), target_h.max(1));
    loop {
        let scaled = decoded.resize_exact(w, h, FilterType::Lanczos3);
        let mut candidates = vec![(encode(&scaled, ImageFormat::Png, 0)?, "image/png")];
        for quality in &qualities {
            candidates.push((encode(&scaled, ImageFormat::Jpeg, *quality)?, "image/jpeg"));
        }
        if let Some((bytes, mime_type)) = candidates
            .into_iter()
            .find(|(b, _)| base64_len(b.len()) < options.max_bytes)
        {
            return Some(Resized {
                bytes,
                mime_type,
                original: (width, height),
                size: (w, h),
                was_resized: true,
            });
        }
        if w == 1 && h == 1 {
            return None;
        }
        let (next_w, next_h) = (
            ((f64::from(w) * 0.75) as u32).max(1),
            ((f64::from(h) * 0.75) as u32).max(1),
        );
        if (next_w, next_h) == (w, h) {
            return None;
        }
        (w, h) = (next_w, next_h);
    }
}

fn dimension_note(resized: &Resized) -> Option<String> {
    resized.was_resized.then(|| {
        let scale = f64::from(resized.original.0) / f64::from(resized.size.0);
        format!(
            "[Image: original {}x{}, displayed at {}x{}. Multiply coordinates by {scale:.2} to map to original image.]",
            resized.original.0, resized.original.1, resized.size.0, resized.size.1
        )
    })
}

/// Pi's `processImage`. `Err` carries the model-facing "omitted" message.
pub fn process_image(
    bytes: &[u8],
    mime: &str,
    auto_resize: bool,
    options: ResizeOptions,
) -> Result<ProcessedImage, String> {
    let (bytes, mime_type, converted_from) = match supported_mime(mime) {
        Some(supported) => (bytes.to_vec(), supported, None),
        None => {
            let decoded = image::load_from_memory(bytes).map_err(|_| {
                "[Image omitted: could not be converted to a supported inline image format.]"
                    .to_owned()
            })?;
            let png = encode(&decoded, ImageFormat::Png, 0).ok_or_else(|| {
                "[Image omitted: could not be converted to a supported inline image format.]"
                    .to_owned()
            })?;
            (png, "image/png", Some(base_mime(mime)))
        }
    };
    let mut hints = Vec::new();
    let converted_hint = |to: &str| {
        converted_from
            .as_deref()
            .filter(|from| *from != to)
            .map(|from| format!("[Image converted from {from} to {to}.]"))
    };
    if !auto_resize {
        hints.extend(converted_hint(mime_type));
        return Ok(ProcessedImage {
            data: base64::engine::general_purpose::STANDARD.encode(&bytes),
            mime_type: mime_type.to_owned(),
            hints,
        });
    }
    let resized = resize(&bytes, mime_type, options).ok_or_else(|| {
        "[Image omitted: could not be resized below the inline image size limit.]".to_owned()
    })?;
    hints.extend(converted_hint(resized.mime_type));
    hints.extend(dimension_note(&resized));
    Ok(ProcessedImage {
        data: base64::engine::general_purpose::STANDARD.encode(&resized.bytes),
        mime_type: resized.mime_type.to_owned(),
        hints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = DynamicImage::ImageLuma8(image::GrayImage::from_fn(width, height, |_, y| {
            image::Luma([(y % 256) as u8])
        }));
        encode(&image, ImageFormat::Png, 0).unwrap()
    }

    fn bmp_1x1_red() -> Vec<u8> {
        let mut b = vec![0u8; 58];
        b[..2].copy_from_slice(b"BM");
        b[2..6].copy_from_slice(&58u32.to_le_bytes());
        b[10..14].copy_from_slice(&54u32.to_le_bytes());
        b[14..18].copy_from_slice(&40u32.to_le_bytes());
        b[18..22].copy_from_slice(&1i32.to_le_bytes());
        b[22..26].copy_from_slice(&1i32.to_le_bytes());
        b[26..28].copy_from_slice(&1u16.to_le_bytes());
        b[28..30].copy_from_slice(&24u16.to_le_bytes());
        b[34..38].copy_from_slice(&4u32.to_le_bytes());
        b[56] = 0xff;
        b
    }

    fn dimensions(processed: &ProcessedImage) -> (u32, u32) {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&processed.data)
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        (decoded.width(), decoded.height())
    }

    #[test]
    fn images_within_limits_pass_through_unchanged() {
        let bytes = png(10, 10);
        let processed = process_image(&bytes, "image/png", true, ResizeOptions::default()).unwrap();
        assert_eq!(processed.mime_type, "image/png");
        assert!(processed.hints.is_empty());
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(&processed.data)
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn oversized_images_shrink_and_report_the_original_dimensions() {
        let processed = process_image(
            &png(2400, 4800),
            "image/png",
            true,
            ResizeOptions::default(),
        )
        .unwrap();
        let (w, h) = dimensions(&processed);
        assert!(w <= 2000 && h <= 2000, "{w}x{h}");
        assert_eq!((w, h), (1000, 2000));
        assert!(processed.hints[0].contains("original 2400x4800, displayed at 1000x2000"));
        assert!(processed.hints[0].contains("Multiply coordinates by 2.40"));
    }

    #[test]
    fn auto_resize_can_be_disabled_but_unsupported_formats_still_convert() {
        let big = png(2400, 100);
        let kept = process_image(&big, "image/png", false, ResizeOptions::default()).unwrap();
        assert_eq!(dimensions(&kept), (2400, 100));
        assert!(kept.hints.is_empty());
        let converted =
            process_image(&bmp_1x1_red(), "image/bmp", false, ResizeOptions::default()).unwrap();
        assert_eq!(converted.mime_type, "image/png");
        assert_eq!(
            converted.hints,
            ["[Image converted from image/bmp to image/png.]"]
        );
    }

    #[test]
    fn undecodable_images_and_impossible_limits_are_reported() {
        let error = process_image(b"not an image", "image/bmp", true, ResizeOptions::default())
            .unwrap_err();
        assert!(error.starts_with("[Image omitted: could not be converted"));
        let tiny_limit = ResizeOptions {
            max_bytes: 4,
            ..ResizeOptions::default()
        };
        let error = process_image(&png(8, 8), "image/png", true, tiny_limit).unwrap_err();
        assert!(error.starts_with("[Image omitted: could not be resized"));
    }
}
