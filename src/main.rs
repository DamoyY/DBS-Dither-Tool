use core::{
    ops::{Add as _, Neg as _},
    ptr::NonNull,
    sync::atomic::{AtomicI32, Ordering},
};
use std::{
    fs,
    io::{self, Write as _},
    path::Path,
    time::Instant,
};

use anyhow::{Context as _, Result};
use image::{DynamicImage, GenericImageView as _, GrayImage, ImageBuffer, Luma, RgbImage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::Rng as _;
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Copy)]
struct Config {
    hvs_sigma: f32,
    hvs_kernel_size: i32,
}

#[derive(Clone, Copy)]
struct DbsContextPtr(NonNull<DbsContext>);
// SAFETY: DbsContextPtr is a wrapper around a raw pointer to DbsContext.
// DbsContext is Send and Sync, so this is safe.
unsafe impl Send for DbsContextPtr {}
// SAFETY: Same as above.
unsafe impl Sync for DbsContextPtr {}
const FIXED_FRAC_BITS: i32 = 16;
const FIXED_ONE: i64 = 1_i64 << FIXED_FRAC_BITS;
const FIXED_SCALE_F64: f64 = 65536.0;
fn float_to_fixed(val: f32) -> i64 {
    let result = f64::from(val)
        .mul_add(FIXED_SCALE_F64, 0.5)
        .trunc()
        .to_bits();
    let sign = if result >> 63_i32 == 1 { -1_i64 } else { 1_i64 };
    let exponent = i32::try_from((result >> 52_i32) & 0x7FF).unwrap_or(0_i32);
    let mantissa = (result & 0x000F_FFFF_FFFF_FFFF) | 0x0010_0000_0000_0000;
    if exponent == 0_i32 {
        return 0;
    }
    let shift = exponent.saturating_sub(1023).saturating_sub(52);
    let magnitude = if shift >= 0_i32 {
        let shift_amount = u32::try_from(shift).unwrap_or(0_u32);
        mantissa
            .cast_signed()
            .saturating_mul(1_i64.checked_shl(shift_amount).unwrap_or(0))
    } else {
        let neg_shift = 0_i32.saturating_sub(shift).max(0_i32);
        let shift_amount = u32::try_from(neg_shift).unwrap_or(0_u32);
        let divisor = 1_i64.checked_shl(shift_amount).unwrap_or(1);
        mantissa.cast_signed().checked_div(divisor).unwrap_or(0)
    };
    sign.saturating_mul(magnitude)
}
const fn fixed_mul(lhs: i64, rhs: i64) -> i64 {
    (lhs.saturating_mul(rhs)) >> FIXED_FRAC_BITS
}
#[derive(Debug, Clone, Copy)]
struct KernelOffset {
    dx: i32,
    dy: i32,
    offset: isize,
    weight: i64,
}
#[derive(Debug, Clone, Copy)]
struct Block {
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
}
struct HvsKernel {
    offsets: Vec<KernelOffset>,
    flat: Vec<i64>,
    c_pp: i64,
    autocorr: Vec<i64>,
    autocorr_offsets: Vec<KernelOffset>,
    autocorr_size: i32,
}
struct DbsContext {
    width: u32,
    height: u32,
    padded_width: usize,
    _padded_height: usize,
    padding: usize,
    original: Vec<i64>,
    halftone: Vec<i64>,
    error_map: Vec<i64>,
    e2_map: Vec<i64>,
    kernel: HvsKernel,
    kernel_size: i32,
    kernel_radius: i32,
}
impl DbsContext {
    #[inline]
    fn pixel_index(&self, xx: u32, yy: u32) -> usize {
        let py = usize::try_from(yy)
            .unwrap_or(0)
            .saturating_add(self.padding);
        let px = usize::try_from(xx)
            .unwrap_or(0)
            .saturating_add(self.padding);
        py.saturating_mul(self.padded_width).saturating_add(px)
    }

    fn new(img: &GrayImage, config: Config) -> Self {
        let (width, height) = img.dimensions();
        let kernel = generate_hvs_kernel(config.hvs_kernel_size, config.hvs_sigma);
        let kernel_size = config.hvs_kernel_size;
        let kernel_radius = kernel_size.div_euclid(2);
        let autocorr_radius = kernel.autocorr_size.div_euclid(2);
        let padding = usize::try_from(autocorr_radius).unwrap_or(0);
        let width_usize = usize::try_from(width).unwrap_or(0);
        let height_usize = usize::try_from(height).unwrap_or(0);
        let padded_width = width_usize.saturating_add(padding.saturating_mul(2));
        let padded_height = height_usize.saturating_add(padding.saturating_mul(2));
        let size = padded_width.saturating_mul(padded_height);
        let mut original = vec![0_i64; size];
        let mut halftone = vec![0_i64; size];
        let half_fixed = FIXED_ONE.div_euclid(2);
        for (y, row) in img.enumerate_rows() {
            for (x, _y, pixel) in row {
                let val_u8 = pixel[0];
                let val_fixed = i64::from(val_u8).saturating_mul(FIXED_ONE).div_euclid(255);
                let py = usize::try_from(y).unwrap_or(0).saturating_add(padding);
                let px = usize::try_from(x).unwrap_or(0).saturating_add(padding);
                let idx = py.saturating_mul(padded_width).saturating_add(px);
                if let Some(orig) = original.get_mut(idx) {
                    *orig = val_fixed;
                }
                if let Some(ht) = halftone.get_mut(idx) {
                    *ht = if val_fixed > half_fixed { FIXED_ONE } else { 0 };
                }
            }
        }
        let mut ctx = Self {
            width,
            height,
            padded_width,
            _padded_height: padded_height,
            padding,
            original,
            halftone,
            error_map: vec![0; size],
            e2_map: vec![0; size],
            kernel,
            kernel_size,
            kernel_radius,
        };
        let stride = isize::try_from(ctx.padded_width).unwrap_or(0);
        for offset in &mut ctx.kernel.offsets {
            offset.offset = isize::try_from(offset.dy)
                .unwrap_or(0)
                .saturating_mul(stride)
                .saturating_add(isize::try_from(offset.dx).unwrap_or(0));
        }
        for offset in &mut ctx.kernel.autocorr_offsets {
            offset.offset = isize::try_from(offset.dy)
                .unwrap_or(0)
                .saturating_mul(stride)
                .saturating_add(isize::try_from(offset.dx).unwrap_or(0));
        }
        ctx.initialize_error_map();
        ctx
    }

    #[inline]
    fn get_kernel_weight(&self, dx: i32, dy: i32) -> i64 {
        let idx = (dy.saturating_add(self.kernel_radius))
            .saturating_mul(self.kernel_size)
            .saturating_add(dx.saturating_add(self.kernel_radius));
        self.kernel
            .flat
            .get(usize::try_from(idx).unwrap_or(0))
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    fn get_autocorr(&self, dx: i32, dy: i32) -> i64 {
        let autocorr_radius = self.kernel.autocorr_size.div_euclid(2);
        let idx = (dy.saturating_add(autocorr_radius))
            .saturating_mul(self.kernel.autocorr_size)
            .saturating_add(dx.saturating_add(autocorr_radius));
        self.kernel
            .autocorr
            .get(usize::try_from(idx).unwrap_or(0))
            .copied()
            .unwrap_or(0)
    }

    fn initialize_error_map(&mut self) {
        let diff: Vec<i64> = self
            .halftone
            .par_iter()
            .zip(self.original.par_iter())
            .map(|(ht, orig)| ht.saturating_sub(*orig))
            .collect();
        let width = usize::try_from(self.width).unwrap_or(0);
        let height = usize::try_from(self.height).unwrap_or(0);
        let padded_width = self.padded_width;
        let padding = self.padding;
        let kernel = &self.kernel;
        let kernel_radius = usize::try_from(self.kernel_radius).unwrap_or(0);
        let err_min_y = padding.saturating_sub(kernel_radius);
        let err_max_y = padding.saturating_add(height).saturating_add(kernel_radius);
        let err_min_x = padding.saturating_sub(kernel_radius);
        let err_max_x = padding.saturating_add(width).saturating_add(kernel_radius);
        self.error_map
            .par_chunks_mut(padded_width)
            .enumerate()
            .for_each(|(y, row)| {
                if y >= err_min_y && y < err_max_y {
                    for x in err_min_x..err_max_x {
                        let mut sum: i64 = 0;
                        for offset in &kernel.offsets {
                            let curr_idx = y.saturating_mul(padded_width).saturating_add(x);
                            let off_idx = isize::try_from(curr_idx)
                                .unwrap_or(0)
                                .wrapping_add(offset.offset);
                            if let Some(&diff_val) =
                                usize::try_from(off_idx).ok().and_then(|idx| diff.get(idx))
                            {
                                sum = sum.saturating_add(fixed_mul(diff_val, offset.weight));
                            }
                        }
                        if let Some(val) = row.get_mut(x) {
                            *val = sum;
                        }
                    }
                }
            });
        let error_map = &self.error_map;
        let e2_min_y = padding;
        let e2_max_y = padding.saturating_add(height);
        let e2_min_x = padding;
        let e2_max_x = padding.saturating_add(width);
        self.e2_map
            .par_chunks_mut(padded_width)
            .enumerate()
            .for_each(|(y, row)| {
                if y >= e2_min_y && y < e2_max_y {
                    for x in e2_min_x..e2_max_x {
                        let mut sum: i64 = 0;
                        for offset in &kernel.offsets {
                            let curr_idx = y.saturating_mul(padded_width).saturating_add(x);
                            let off_idx = isize::try_from(curr_idx)
                                .unwrap_or(0)
                                .wrapping_add(offset.offset);
                            if let Some(&e_val) = usize::try_from(off_idx)
                                .ok()
                                .and_then(|idx| error_map.get(idx))
                            {
                                sum = sum.saturating_add(fixed_mul(e_val, offset.weight));
                            }
                        }
                        if let Some(val) = row.get_mut(x) {
                            *val = sum;
                        }
                    }
                }
            });
    }

    #[inline]
    fn calc_toggle_delta_se(&self, xx: u32, yy: u32) -> i64 {
        let idx = self.pixel_index(xx, yy);
        let old_val = self.halftone.get(idx).copied().unwrap_or(0);
        let change = if old_val == 0 { FIXED_ONE } else { -FIXED_ONE };
        let e2 = self.e2_map.get(idx).copied().unwrap_or(0);
        let term1 = fixed_mul(2_i64.saturating_mul(change), e2);
        term1.saturating_add(self.kernel.c_pp)
    }

    #[inline]
    fn apply_toggle(&mut self, xx: u32, yy: u32) {
        let idx = self.pixel_index(xx, yy);
        let old_val = self.halftone.get(idx).copied().unwrap_or(0);
        let new_val = if old_val == 0 { FIXED_ONE } else { 0 };
        let change = new_val.saturating_sub(old_val);
        if let Some(ht_val) = self.halftone.get_mut(idx) {
            *ht_val = new_val;
        }
        let padded_width = self.padded_width;
        let padding = self.padding;
        let py_center = usize::try_from(yy).unwrap_or(0).saturating_add(padding);
        let px_center = usize::try_from(xx).unwrap_or(0).saturating_add(padding);
        let center_idx = py_center
            .saturating_mul(padded_width)
            .saturating_add(px_center);
        let center_idx_isize = isize::try_from(center_idx).unwrap_or(0);
        for offset in &self.kernel.offsets {
            let p_idx_isize = center_idx_isize.wrapping_add(offset.offset);
            if let Some(err_val) = usize::try_from(p_idx_isize)
                .ok()
                .and_then(|p_idx| self.error_map.get_mut(p_idx))
            {
                *err_val = err_val.saturating_add(fixed_mul(change, offset.weight));
            }
        }
        for offset in &self.kernel.autocorr_offsets {
            let p_idx_isize = center_idx_isize.wrapping_add(offset.offset);
            if let Some(e2_val) = usize::try_from(p_idx_isize)
                .ok()
                .and_then(|p_idx| self.e2_map.get_mut(p_idx))
            {
                *e2_val = e2_val.saturating_add(fixed_mul(change, offset.weight));
            }
        }
    }

    fn can_swap(&self, x1: u32, y1: u32, x2: u32, y2: u32) -> bool {
        let idx1 = self.pixel_index(x1, y1);
        let idx2 = self.pixel_index(x2, y2);
        let val1 = self.halftone.get(idx1).copied().unwrap_or(0);
        let val2 = self.halftone.get(idx2).copied().unwrap_or(0);
        let half_fixed = FIXED_ONE.div_euclid(2);
        val1.saturating_sub(val2).abs() >= half_fixed
    }

    fn swap_bounds(p1: (i32, i32), p2: (i32, i32), kernel_radius: i32) -> (i32, i32, i32, i32) {
        let (x1, y1) = p1;
        let (x2, y2) = p2;
        let min_x = x1.min(x2).saturating_sub(kernel_radius);
        let max_x = x1.max(x2).saturating_add(kernel_radius);
        let min_y = y1.min(y2).saturating_sub(kernel_radius);
        let max_y = y1.max(y2).saturating_add(kernel_radius);
        (min_x, max_x, min_y, max_y)
    }

    fn calc_swap_delta_e(
        &self,
        point: (i32, i32),
        pos1: (i32, i32),
        pos2: (i32, i32),
        changes: (i64, i64),
    ) -> Option<i64> {
        let (px, py) = point;
        let (x1i, y1i) = pos1;
        let (x2i, y2i) = pos2;
        let (change1, change2) = changes;
        let dx1 = px.saturating_sub(x1i);
        let dy1 = py.saturating_sub(y1i);
        let dx2 = px.saturating_sub(x2i);
        let dy2 = py.saturating_sub(y2i);
        let in_kernel1 = dx1.abs() <= self.kernel_radius && dy1.abs() <= self.kernel_radius;
        let in_kernel2 = dx2.abs() <= self.kernel_radius && dy2.abs() <= self.kernel_radius;
        if !in_kernel1 && !in_kernel2 {
            return None;
        }
        let mut delta_e: i64 = 0;
        if in_kernel1 {
            delta_e = delta_e.saturating_add(fixed_mul(change1, self.get_kernel_weight(dx1, dy1)));
        }
        if in_kernel2 {
            delta_e = delta_e.saturating_add(fixed_mul(change2, self.get_kernel_weight(dx2, dy2)));
        }
        Some(delta_e)
    }

    #[inline]
    fn calc_swap_delta_se(&self, x1: u32, y1: u32, x2: u32, y2: u32) -> i64 {
        let idx1 = self.pixel_index(x1, y1);
        let idx2 = self.pixel_index(x2, y2);
        let val1 = self.halftone.get(idx1).copied().unwrap_or(0);
        let val2 = self.halftone.get(idx2).copied().unwrap_or(0);
        let change1 = val2.saturating_sub(val1);
        let change2 = change1.saturating_neg();
        let e2_1 = self.e2_map.get(idx1).copied().unwrap_or(0);
        let e2_2 = self.e2_map.get(idx2).copied().unwrap_or(0);
        let x1i = i32::try_from(x1).unwrap_or(0_i32);
        let y1i = i32::try_from(y1).unwrap_or(0_i32);
        let x2i = i32::try_from(x2).unwrap_or(0_i32);
        let y2i = i32::try_from(y2).unwrap_or(0_i32);
        let dx = x1i.saturating_sub(x2i);
        let dy = y1i.saturating_sub(y2i);
        let c_12 = self.get_autocorr(dx, dy);
        let term1 = fixed_mul(2_i64.saturating_mul(change1), e2_1);
        let term2 = fixed_mul(2_i64.saturating_mul(change2), e2_2);
        let term3 = 2_i64.saturating_mul(self.kernel.c_pp);
        let term4 = fixed_mul(2_i64.saturating_mul(fixed_mul(change1, change2)), c_12);
        term1
            .saturating_add(term2)
            .saturating_add(term3)
            .saturating_add(term4)
    }

    #[inline]
    fn apply_swap(&mut self, x1: u32, y1: u32, x2: u32, y2: u32) {
        let idx1 = self.pixel_index(x1, y1);
        let idx2 = self.pixel_index(x2, y2);
        let val1 = self.halftone.get(idx1).copied().unwrap_or(0);
        let val2 = self.halftone.get(idx2).copied().unwrap_or(0);
        let change1 = val2.saturating_sub(val1);
        let change2 = change1.saturating_neg();
        if let Some(ht1) = self.halftone.get_mut(idx1) {
            *ht1 = val2;
        }
        if let Some(ht2) = self.halftone.get_mut(idx2) {
            *ht2 = val1;
        }
        let padding_i32 = i32::try_from(self.padding).unwrap_or(0_i32);
        let pos1 = (
            i32::try_from(x1)
                .unwrap_or(0_i32)
                .saturating_add(padding_i32),
            i32::try_from(y1)
                .unwrap_or(0_i32)
                .saturating_add(padding_i32),
        );
        let pos2 = (
            i32::try_from(x2)
                .unwrap_or(0_i32)
                .saturating_add(padding_i32),
            i32::try_from(y2)
                .unwrap_or(0_i32)
                .saturating_add(padding_i32),
        );
        let changes = (change1, change2);
        let (min_x, max_x, min_y, max_y) = Self::swap_bounds(pos1, pos2, self.kernel_radius);
        let padded_width = self.padded_width;
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                if let Some(delta_e) = self.calc_swap_delta_e((px, py), pos1, pos2, changes) {
                    let py_u = usize::try_from(py).unwrap_or(0);
                    let px_u = usize::try_from(px).unwrap_or(0);
                    let p_idx = py_u.saturating_mul(padded_width).saturating_add(px_u);
                    if let Some(err_val) = self.error_map.get_mut(p_idx) {
                        *err_val = err_val.saturating_add(delta_e);
                    }
                }
            }
        }
        let autocorr_radius = self.kernel.autocorr_size.div_euclid(2);
        let (e2_min_x, e2_max_x, e2_min_y, e2_max_y) =
            Self::swap_bounds(pos1, pos2, autocorr_radius);
        let x1i = pos1.0;
        let y1i = pos1.1;
        let x2i = pos2.0;
        let y2i = pos2.1;
        for py in e2_min_y..=e2_max_y {
            for px in e2_min_x..=e2_max_x {
                let dx1 = px.saturating_sub(x1i);
                let dy1 = py.saturating_sub(y1i);
                let dx2 = px.saturating_sub(x2i);
                let dy2 = py.saturating_sub(y2i);
                let in_range1 = dx1.abs() <= autocorr_radius && dy1.abs() <= autocorr_radius;
                let in_range2 = dx2.abs() <= autocorr_radius && dy2.abs() <= autocorr_radius;
                if !in_range1 && !in_range2 {
                    continue;
                }
                let mut delta_e2: i64 = 0;
                if in_range1 {
                    let c1 = self.get_autocorr(dx1, dy1);
                    delta_e2 = delta_e2.saturating_add(fixed_mul(change1, c1));
                }
                if in_range2 {
                    let c2 = self.get_autocorr(dx2, dy2);
                    delta_e2 = delta_e2.saturating_add(fixed_mul(change2, c2));
                }
                let py_u = usize::try_from(py).unwrap_or(0);
                let px_u = usize::try_from(px).unwrap_or(0);
                let p_idx = py_u.saturating_mul(padded_width).saturating_add(px_u);
                if let Some(e2_val) = self.e2_map.get_mut(p_idx) {
                    *e2_val = e2_val.saturating_add(delta_e2);
                }
            }
        }
    }

    fn try_best_operation(&mut self, xx: u32, yy: u32) -> Option<bool> {
        const NEIGHBORS: [(i32, i32); 8] = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        let mut best_delta_se: i64 = 0;
        let mut best_op: Option<(i32, i32)> = None;
        let toggle_delta = self.calc_toggle_delta_se(xx, yy);
        if toggle_delta < best_delta_se {
            best_delta_se = toggle_delta;
            best_op = Some((0_i32, 0_i32));
        }
        let xi = i32::try_from(xx).unwrap_or(0_i32);
        let yi = i32::try_from(yy).unwrap_or(0_i32);
        let wi = i32::try_from(self.width).unwrap_or(i32::MAX);
        let hi = i32::try_from(self.height).unwrap_or(i32::MAX);
        for (dx, dy) in NEIGHBORS {
            let nx = xi.saturating_add(dx);
            let ny = yi.saturating_add(dy);
            if nx >= 0_i32 && nx < wi && ny >= 0_i32 && ny < hi {
                let nxu = u32::try_from(nx).unwrap_or(0);
                let nyu = u32::try_from(ny).unwrap_or(0);
                if self.can_swap(xx, yy, nxu, nyu) {
                    let swap_delta = self.calc_swap_delta_se(xx, yy, nxu, nyu);
                    if swap_delta < best_delta_se {
                        best_delta_se = swap_delta;
                        best_op = Some((dx, dy));
                    }
                }
            }
        }
        match best_op {
            Some((0, 0)) => {
                self.apply_toggle(xx, yy);
                Some(true)
            }
            Some((dx, dy)) => {
                let nx = u32::try_from(xi.saturating_add(dx)).unwrap_or(0);
                let ny = u32::try_from(yi.saturating_add(dy)).unwrap_or(0);
                self.apply_swap(xx, yy, nx, ny);
                Some(false)
            }
            None => None,
        }
    }

    fn compute_block_size(&self) -> u32 {
        let autocorr_radius = self.kernel.autocorr_size.div_euclid(2);
        let max_radius = self.kernel_radius.max(autocorr_radius);
        u32::try_from(max_radius.saturating_mul(2))
            .unwrap_or(64)
            .max(1)
    }

    fn generate_blocks(&self, shift_x: i32, shift_y: i32, block_size: u32) -> [Vec<Block>; 4] {
        let mut phases: [Vec<Block>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let width_i32 = i32::try_from(self.width).unwrap_or(0_i32);
        let height_i32 = i32::try_from(self.height).unwrap_or(0_i32);
        let block_size_i32 = i32::try_from(block_size).unwrap_or(32_i32);
        let start_bx = shift_x.div_euclid(block_size_i32).saturating_sub(1);
        let start_by = shift_y.div_euclid(block_size_i32).saturating_sub(1);
        let end_bx = width_i32
            .saturating_sub(1)
            .saturating_add(shift_x)
            .div_euclid(block_size_i32)
            .saturating_add(1);
        let end_by = height_i32
            .saturating_sub(1)
            .saturating_add(shift_y)
            .div_euclid(block_size_i32)
            .saturating_add(1);
        for by in start_by..=end_by {
            for bx in start_bx..=end_bx {
                let block_start_x = bx.saturating_mul(block_size_i32).saturating_sub(shift_x);
                let block_start_y = by.saturating_mul(block_size_i32).saturating_sub(shift_y);
                let block_end_x = block_start_x.saturating_add(block_size_i32);
                let block_end_y = block_start_y.saturating_add(block_size_i32);
                let clipped_start_x = block_start_x.max(0_i32);
                let clipped_start_y = block_start_y.max(0_i32);
                let clipped_end_x = block_end_x.min(width_i32);
                let clipped_end_y = block_end_y.min(height_i32);
                if clipped_start_x >= clipped_end_x || clipped_start_y >= clipped_end_y {
                    continue;
                }
                let block = Block {
                    start_x: u32::try_from(clipped_start_x).unwrap_or(0),
                    start_y: u32::try_from(clipped_start_y).unwrap_or(0),
                    end_x: u32::try_from(clipped_end_x).unwrap_or(0),
                    end_y: u32::try_from(clipped_end_y).unwrap_or(0),
                };
                let bx_mod = bx.rem_euclid(2);
                let by_mod = by.rem_euclid(2);
                let phase_idx =
                    usize::try_from(bx_mod.saturating_add(by_mod.saturating_mul(2))).unwrap_or(0);
                if let Some(phase) = phases.get_mut(phase_idx) {
                    phase.push(block);
                }
            }
        }
        phases
    }

    fn process_block(&mut self, block: &Block) -> (i32, i32) {
        let mut toggles = 0_i32;
        let mut swaps = 0_i32;
        for yy in block.start_y..block.end_y {
            if yy.rem_euclid(2) == 0 {
                for xx in block.start_x..block.end_x {
                    match self.try_best_operation(xx, yy) {
                        Some(true) => toggles = toggles.saturating_add(1),
                        Some(false) => swaps = swaps.saturating_add(1),
                        None => {}
                    }
                }
            } else {
                for xx in (block.start_x..block.end_x).rev() {
                    match self.try_best_operation(xx, yy) {
                        Some(true) => toggles = toggles.saturating_add(1),
                        Some(false) => swaps = swaps.saturating_add(1),
                        None => {}
                    }
                }
            }
        }
        (toggles, swaps)
    }
}
fn generate_hvs_kernel(size: i32, sigma: f32) -> HvsKernel {
    let center = size.div_euclid(2_i32);
    let flat_size = usize::try_from(size.saturating_mul(size)).unwrap_or(0);
    let mut offsets = Vec::with_capacity(flat_size);
    let mut flat = vec![0_i64; flat_size];
    let mut weights_f32: Vec<f32> = Vec::with_capacity(flat_size);
    let mut sum_f32 = 0.0_f32;
    let neg_center = 0_i32.saturating_sub(center);
    let two_sigma_sq = sigma.mul_add(sigma, 0.0).mul_add(2.0, 0.0);
    for dy in neg_center..=center {
        for dx in neg_center..=center {
            let dist_sq = dx.saturating_mul(dx).saturating_add(dy.saturating_mul(dy));
            let dist_sq_f32 = i16::try_from(dist_sq).unwrap_or(i16::MAX);
            let exponent = f32::from(dist_sq_f32)
                .neg()
                .mul_add(two_sigma_sq.recip(), 0.0);
            let weight_f32 = exponent.exp();
            sum_f32 = sum_f32.add(weight_f32);
            weights_f32.push(weight_f32);
        }
    }
    let mut idx = 0_usize;
    for dy in neg_center..=center {
        for dx in neg_center..=center {
            let weight_f32 = weights_f32.get(idx).copied().unwrap_or(0.0);
            let normalized = weight_f32.mul_add(sum_f32.recip(), 0.0);
            let weight_fixed = float_to_fixed(normalized);
            offsets.push(KernelOffset {
                dx,
                dy,
                offset: 0,
                weight: weight_fixed,
            });
            let flat_idx = usize::try_from(
                (dy.saturating_add(center))
                    .saturating_mul(size)
                    .saturating_add(dx.saturating_add(center)),
            )
            .unwrap_or(0);
            if let Some(flat_val) = flat.get_mut(flat_idx) {
                *flat_val = weight_fixed;
            }
            idx = idx.saturating_add(1);
        }
    }
    let mut c_pp: i64 = 0;
    for w in &flat {
        c_pp = c_pp.saturating_add(fixed_mul(*w, *w));
    }
    let (autocorr, autocorr_offsets) = generate_autocorr(size, &flat, center);
    HvsKernel {
        offsets,
        flat,
        c_pp,
        autocorr,
        autocorr_offsets,
        autocorr_size: size.saturating_mul(2).saturating_sub(1),
    }
}
fn generate_autocorr(size: i32, flat: &[i64], center: i32) -> (Vec<i64>, Vec<KernelOffset>) {
    let autocorr_size = size.saturating_mul(2).saturating_sub(1);
    let autocorr_radius = autocorr_size.div_euclid(2);
    let autocorr_flat_size =
        usize::try_from(autocorr_size.saturating_mul(autocorr_size)).unwrap_or(0);
    let mut autocorr = vec![0_i64; autocorr_flat_size];
    let neg_autocorr_radius = 0_i32.saturating_sub(autocorr_radius);
    let neg_center = 0_i32.saturating_sub(center);
    let mut autocorr_offsets = Vec::with_capacity(autocorr_flat_size);
    for dy in neg_autocorr_radius..=autocorr_radius {
        for dx in neg_autocorr_radius..=autocorr_radius {
            let mut sum: i64 = 0;
            for ky in neg_center..=center {
                for kx in neg_center..=center {
                    let ky2 = ky.saturating_sub(dy);
                    let kx2 = kx.saturating_sub(dx);
                    if ky2 >= neg_center && ky2 <= center && kx2 >= neg_center && kx2 <= center {
                        let idx1 = usize::try_from(
                            (ky.saturating_add(center))
                                .saturating_mul(size)
                                .saturating_add(kx.saturating_add(center)),
                        )
                        .unwrap_or(0);
                        let idx2 = usize::try_from(
                            (ky2.saturating_add(center))
                                .saturating_mul(size)
                                .saturating_add(kx2.saturating_add(center)),
                        )
                        .unwrap_or(0);
                        let w1 = flat.get(idx1).copied().unwrap_or(0);
                        let w2 = flat.get(idx2).copied().unwrap_or(0);
                        sum = sum.saturating_add(fixed_mul(w1, w2));
                    }
                }
            }
            let ac_idx = usize::try_from(
                (dy.saturating_add(autocorr_radius))
                    .saturating_mul(autocorr_size)
                    .saturating_add(dx.saturating_add(autocorr_radius)),
            )
            .unwrap_or(0);
            if let Some(ac_val) = autocorr.get_mut(ac_idx) {
                *ac_val = sum;
            }
            autocorr_offsets.push(KernelOffset {
                dx,
                dy,
                offset: 0,
                weight: sum,
            });
        }
    }
    (autocorr, autocorr_offsets)
}
fn load_config() -> Result<Config> {
    let config_path = "config.yaml";
    let config_content = fs::read_to_string(config_path)
        .with_context(|| format!("无法读取配置文件: {config_path}"))?;
    let config: Config =
        serde_yaml::from_str(&config_content).with_context(|| "解析配置文件失败")?;
    println!(
        "配置已加载: Sigma={}, KernelSize={}",
        config.hvs_sigma, config.hvs_kernel_size
    );
    Ok(config)
}
fn run_dbs_iterations(dbs: &mut DbsContext, pb: &ProgressBar) -> (i32, i32) {
    let mut toggle_count = 0_i32;
    let mut swap_count = 0_i32;
    let block_size = dbs.compute_block_size();
    let mut rng = rand::rng();
    let block_size_i32 = i32::try_from(block_size).unwrap_or(32_i32);
    for iter in 0_usize.. {
        pb.set_position(0);
        let shift_x: i32 = rng.random_range(0_i32..block_size_i32);
        let shift_y: i32 = rng.random_range(0_i32..block_size_i32);
        let phases = dbs.generate_blocks(shift_x, shift_y, block_size);
        pb.set_message(format!(
            "迭代 {} (偏移: {shift_x}, {shift_y})",
            iter.saturating_add(1)
        ));
        let iter_toggles = AtomicI32::new(0_i32);
        let iter_swaps = AtomicI32::new(0_i32);
        let dbs_ptr = DbsContextPtr(NonNull::from(&mut *dbs));
        for (phase_idx, phase_blocks) in phases.iter().enumerate() {
            if phase_blocks.is_empty() {
                continue;
            }
            let pb_ref = pb;
            let iter_t_ref = &iter_toggles;
            let iter_s_ref = &iter_swaps;
            phase_blocks.par_iter().for_each(move |block| {
                let dbs_ptr_inner = dbs_ptr;
                // SAFETY: We process blocks in phases where blocks in the same phase
                // do not overlap their influence regions.
                let dbs_ref = unsafe { &mut *dbs_ptr_inner.0.as_ptr() };
                let (block_toggles, block_swaps) = dbs_ref.process_block(block);
                iter_t_ref.fetch_add(block_toggles, Ordering::Relaxed);
                iter_s_ref.fetch_add(block_swaps, Ordering::Relaxed);
                let pixels = u64::from(
                    block
                        .end_x
                        .saturating_sub(block.start_x)
                        .saturating_mul(block.end_y.saturating_sub(block.start_y)),
                );
                pb_ref.inc(pixels);
            });
            core::sync::atomic::fence(Ordering::SeqCst);
            pb.set_message(format!(
                "迭代 {} 相位 {}/4",
                iter.saturating_add(1),
                phase_idx.saturating_add(1)
            ));
        }
        let iter_t = iter_toggles.load(Ordering::SeqCst);
        let iter_s = iter_swaps.load(Ordering::SeqCst);
        toggle_count = toggle_count.saturating_add(iter_t);
        swap_count = swap_count.saturating_add(iter_s);
        if iter_t == 0_i32 && iter_s == 0_i32 {
            pb.println(format!("已于第 {} 次迭代收敛。", iter.saturating_add(1)));
            break;
        }
    }
    (toggle_count, swap_count)
}

fn create_progress_bar(total_pixels: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total_pixels);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {msg} [{bar:40.cyan/blue}] {percent}%")
            .context("进度条样式错误")?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

fn perform_dbs_with_progress(
    img: &GrayImage,
    config: Config,
    channel_name: &str,
    pb: &ProgressBar,
) -> (DbsContext, i32, i32) {
    pb.set_message(format!("{channel_name} 初始化中..."));
    let mut dbs = DbsContext::new(img, config);
    let (toggles, swaps) = run_dbs_iterations(&mut dbs, pb);
    pb.finish_with_message(format!("{channel_name} 完成"));
    (dbs, toggles, swaps)
}

fn perform_dbs_on_channel(
    img: &GrayImage,
    config: Config,
    channel_name: &str,
) -> Result<(DbsContext, i32, i32)> {
    let (width, height) = img.dimensions();
    let total_pixels = u64::from(width.saturating_mul(height));
    let pb = create_progress_bar(total_pixels)?;
    Ok(perform_dbs_with_progress(img, config, channel_name, &pb))
}

fn process_mono(img: &DynamicImage, config: Config) -> Result<(DynamicImage, i32, i32)> {
    let (width, height) = img.dimensions();
    let gray_img = img.to_luma8();
    let (dbs, toggles, swaps) = perform_dbs_on_channel(&gray_img, config, "单色通道")?;
    println!("正在保存结果...");
    let half_fixed = FIXED_ONE.div_euclid(2);
    let mut out_img: GrayImage = ImageBuffer::new(width, height);
    for (x, y, pixel) in out_img.enumerate_pixels_mut() {
        let idx = dbs.pixel_index(x, y);
        let val = if dbs.halftone.get(idx).copied().unwrap_or(0) > half_fixed {
            255
        } else {
            0
        };
        *pixel = Luma([val]);
    }
    Ok((DynamicImage::ImageLuma8(out_img), toggles, swaps))
}
fn process_color(img: &DynamicImage, config: Config) -> Result<(DynamicImage, i32, i32)> {
    println!("正在进行彩色 DBS 处理 (RGB 并行)...");
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();
    let mut r_plane = GrayImage::new(width, height);
    let mut g_plane = GrayImage::new(width, height);
    let mut b_plane = GrayImage::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        r_plane.put_pixel(x, y, Luma([pixel[0]]));
        g_plane.put_pixel(x, y, Luma([pixel[1]]));
        b_plane.put_pixel(x, y, Luma([pixel[2]]));
    }
    let total_pixels = u64::from(width.saturating_mul(height));
    let multi_progress = MultiProgress::new();
    let r_pb = multi_progress.add(create_progress_bar(total_pixels)?);
    let g_pb = multi_progress.add(create_progress_bar(total_pixels)?);
    let b_pb = multi_progress.add(create_progress_bar(total_pixels)?);
    let (r_result, g_result, b_result) = std::thread::scope(|scope| {
        let r_handle = scope.spawn(|| perform_dbs_with_progress(&r_plane, config, "红色", &r_pb));
        let g_handle = scope.spawn(|| perform_dbs_with_progress(&g_plane, config, "绿色", &g_pb));
        let b_handle = scope.spawn(|| perform_dbs_with_progress(&b_plane, config, "蓝色", &b_pb));
        let r_result = match r_handle.join() {
            Ok(result) => result,
            Err(err) => std::panic::resume_unwind(err),
        };
        let g_result = match g_handle.join() {
            Ok(result) => result,
            Err(err) => std::panic::resume_unwind(err),
        };
        let b_result = match b_handle.join() {
            Ok(result) => result,
            Err(err) => std::panic::resume_unwind(err),
        };
        (r_result, g_result, b_result)
    });
    let (r_dbs, r_toggles, r_swaps) = r_result;
    let (g_dbs, g_toggles, g_swaps) = g_result;
    let (b_dbs, b_toggles, b_swaps) = b_result;
    let total_toggles = r_toggles
        .saturating_add(g_toggles)
        .saturating_add(b_toggles);
    let total_swaps = r_swaps.saturating_add(g_swaps).saturating_add(b_swaps);
    println!("正在合成彩色结果...");
    let half_fixed = FIXED_ONE.div_euclid(2);
    let mut out_rgb = RgbImage::new(width, height);
    for (x, y, pixel) in out_rgb.enumerate_pixels_mut() {
        let r_idx = r_dbs.pixel_index(x, y);
        let g_idx = g_dbs.pixel_index(x, y);
        let b_idx = b_dbs.pixel_index(x, y);
        let r_on = r_dbs.halftone.get(r_idx).copied().unwrap_or(0) > half_fixed;
        let g_on = g_dbs.halftone.get(g_idx).copied().unwrap_or(0) > half_fixed;
        let b_on = b_dbs.halftone.get(b_idx).copied().unwrap_or(0) > half_fixed;
        let red = if r_on { 255 } else { 0 };
        let green = if g_on { 255 } else { 0 };
        let blue = if b_on { 255 } else { 0 };
        *pixel = image::Rgb([red, green, blue]);
    }
    Ok((DynamicImage::ImageRgb8(out_rgb), total_toggles, total_swaps))
}
fn main() -> Result<()> {
    println!("=== Direct Binary Search (DBS) Dithering 工具 ===");
    let config = load_config()?;
    print!("请输入图片路径: ");
    io::stdout().flush()?;
    let mut input_path_buf = String::new();
    io::stdin().read_line(&mut input_path_buf)?;
    let input_path = input_path_buf.trim().trim_matches('"').trim_matches('\'');
    if input_path.is_empty() {
        return Err(anyhow::anyhow!("未输入路径"));
    }
    println!("正在加载图片...");
    let mut dyn_img =
        image::open(input_path).with_context(|| format!("无法打开文件: {input_path}"))?;
    let (orig_width, orig_height) = dyn_img.dimensions();
    print!("请输入输出高度: ");
    io::stdout().flush()?;
    let mut height_input_buf = String::new();
    io::stdin().read_line(&mut height_input_buf)?;
    let height_input = height_input_buf.trim();
    if !height_input.is_empty() {
        if let Ok(target_height) = height_input.parse::<u32>() {
            if target_height != orig_height {
                let target_width = u32::try_from(
                    u64::from(orig_width)
                        .saturating_mul(u64::from(target_height))
                        .div_euclid(u64::from(orig_height)),
                )
                .unwrap_or(1)
                .max(1);
                println!(
                    "正在执行缩放 ({orig_width}x{orig_height} -> \
                     {target_width}x{target_height})..."
                );
                dyn_img = dyn_img.resize(
                    target_width,
                    target_height,
                    image::imageops::FilterType::Lanczos3,
                );
            }
        } else {
            println!("输入的高度无效，将使用原始尺寸。");
        }
    }
    print!("是否输出单色 (y/n): ");
    io::stdout().flush()?;
    let mut mono_input_buf = String::new();
    io::stdin().read_line(&mut mono_input_buf)?;
    let is_mono = mono_input_buf.trim().eq_ignore_ascii_case("y");
    let (width, height) = dyn_img.dimensions();
    println!("图片尺寸: {width}x{height}");
    let start_time = Instant::now();
    let (out_dynamic, total_toggles, total_swaps) = if is_mono {
        process_mono(&dyn_img, config)?
    } else {
        process_color(&dyn_img, config)?
    };
    let original_path = Path::new(input_path);
    let file_stem = original_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let output_filename = format!("{file_stem}_dbs.png");
    out_dynamic
        .save(&output_filename)
        .with_context(|| "保存图片失败")?;
    let duration = start_time.elapsed();
    println!("成功，已保存至: {output_filename}");
    println!(
        "总耗时: {:.2}s, Toggle: {total_toggles}, Swap: {total_swaps}",
        duration.as_secs_f64()
    );
    Ok(())
}
