use kornia::{image::{Image, ImageSize}, imgproc};

/// Border type: distinguishes whether a contour is the outer perimeter of a
/// foreground region or the perimeter of a hole (background) enclosed by
/// foreground.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderType {
    /// The outer perimeter of a foreground region.
    Outer,
    /// The perimeter of a background “hole” fully enclosed by foreground.
    Hole,
}

/// A detected contour in the image.
///
/// - `points` is the ordered list of (x,y) pixel coordinates along the border.
/// - `border_type` tells you if this is an `Outer` contour or a `Hole`.
/// - `parent` is an optional index into the returned `Vec<Contour>` of the
///   contour that immediately encloses this one (if any).
#[derive(Debug, Clone)]
pub struct Contour {
    /// The (x,y) coordinates of each border pixel in tracing order.
    points: Vec<(usize, usize)>,
    /// Whether this contour is an outer border or a hole.
    border_type: BorderType,
    /// Index of the parent contour (if this is a hole inside another contour).
    parent: Option<usize>,
}

impl Contour {
    /// Create a new `Contour`.
    ///
    /// # Arguments
    ///
    /// * `points` — the sequence of (x,y) coordinates along the border.
    /// * `border_type` — `BorderType::Outer` for an outer perimeter, or
    ///   `BorderType::Hole` for an enclosed hole.
    /// * `parent` — `Some(idx)` if this contour is enclosed by the contour at
    ///   `contours[idx]`, or `None` if it has no enclosing parent.
    pub fn new(
        points: Vec<(usize, usize)>,
        border_type: BorderType,
        parent: Option<usize>,
    ) -> Self {
        Self {
            points,
            border_type,
            parent,
        }
    }
    /// Returns the ordered list of (x, y) coordinates along this contour’s border.
    pub fn points(&self) -> &[(usize, usize)] {
        &self.points
    }
    /// Returns the border type of this contour.
    pub fn border_type(&self) -> BorderType {
        self.border_type
    }
    /// Returns the parent of this contour (NOT implemented yet!)
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }
}

/// Find all contours in a grayscale image by thresholding and applying
/// the Suzuki–Abe border‐following algorithm.
///
/// # Arguments
///
/// * `image` — input grayscale image (`Image<u8,1>`).
/// * `threshold` — any pixel value > this is treated as “foreground”.
///
/// # Returns
///
/// A `Vec<Contour>` listing every outer and hole border found in scan order.
pub fn find_contours(image: &Image<u8, 1>, threshold: u8) -> Vec<Contour> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    // binary mask: true=fg, false=bg
    let bin: Vec<bool> = image.as_slice().iter().map(|&v| v > threshold).collect();
    // labels: 0=unvisited, >1=contour IDs
    let mut labels = vec![0i32; w * h];
    let idx = |x: usize, y: usize| y * w + x;
    let mut y_idx = vec![w * h; h];

    // 8‐connected neighbor offsets in clockwise order: E, SE, S, SW, W, NW, N, NE
    let nbrs: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    let mut contours = Vec::new();
    let mut nbd = 2; // next contour ID

    // raster scan
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let i = row + x;
            // starts from unlabelled foreground.
            if bin[i] && labels[i] == 0 {
                // determine if this pixel sits on an outer or hole border
                let left = if x > 0 { bin[idx(x - 1, y)] } else { false };
                let right = if x + 1 < w { bin[idx(x + 1, y)] } else { false };

                let is_outer = !left;
                let is_hole = !right;

                if is_outer || is_hole {
                    let b0 = if is_outer {
                        (x as i32 - 1, y as i32)
                    } else {
                        (x as i32 + 1, y as i32)
                    };

                    // tracing the closed border
                    let chain = follow_border(&bin, &nbrs, w, h, (x, y), b0);

                    // marked so as to not restart here again
                    for &(cx, cy) in &chain {
                        if y_idx[cy] == w * h {
                            y_idx[cy] = cy * w
                        }
                        labels[y_idx[cy] + cx] = nbd;
                    }

                    contours.push(Contour::new(
                        chain,
                        if is_outer {
                            BorderType::Outer
                        } else {
                            BorderType::Hole
                        },
                        None,
                    ));
                    nbd += 1;
                }
            }
        }
    }

    contours
}

/// Trace a single closed border using the Suzuki–Abe method.
///
/// # Arguments
///
/// * `bin` — flattened binary image (0 or 1), row-major.
/// * `nbrs` — 8 neighbor offsets in clockwise order: E, SE, S, SW, W, NW, N, NE.
/// * `w, h` — image width and height.
/// * `p0` — starting pixel coordinate (x,y), guaranteed foreground and unlabeled.
/// * `b0` — the “backtrack” pixel (one step before `p0`), may lie out of bounds.
///
/// # Returns
///
/// A `Vec<(usize, usize)>` of all border pixels in visit order.
fn follow_border(
    bin: &[bool],
    nbrs: &[(i32, i32); 8],
    w: usize,
    h: usize,
    p0: (usize, usize),
    b0: (i32, i32),
) -> Vec<(usize, usize)> {
    let idx = |x: usize, y: usize| y * w + x;
    let mut contour = Vec::new();
    let mut p = p0;
    let mut b = b0;

    loop {
        contour.push(p);

        // finding neighbour index where offset is b - p
        let dx0 = b.0 - p.0 as i32;
        let dy0 = b.1 - p.1 as i32;
        let mut k0 = 0;
        for (k, &(dx, dy)) in nbrs.iter().enumerate() {
            if dx == dx0 && dy == dy0 {
                k0 = k;
                break;
            }
        }

        // searching clockwise for next foreground neighbour from k0 + 1
        let mut found = false;
        let mut next_p = p;
        let mut next_b = b;
        for s in 1..=8 {
            let d = (k0 + s) % 8;
            let nx = p.0 as i32 + nbrs[d].0;
            let ny = p.1 as i32 + nbrs[d].1;

            if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                let ni = idx(nx as usize, ny as usize);
                if bin[ni] {
                    // found the next border pixel
                    next_p = (nx as usize, ny as usize);
                    // backtrack = neighbor at (d-1 mod 8) relative to p
                    let prev_d = (d + 7) % 8;
                    let bx = p.0 as i32 + nbrs[prev_d].0;
                    let by = p.1 as i32 + nbrs[prev_d].1;
                    next_b = (bx, by);
                    found = true;
                    break;
                }
            }
        }

        if !found {
            // isolated pixel (no next neighbor)
            break;
        }

        p = next_p;
        b = next_b;

        //stop when we return to the start configuration
        if p == p0 && b == b0 {
            break;
        }
    }

    contour
}
