# -*- coding: utf-8 -*-
# Dual-fisheye to 360-photo conversion tool
# Supports equirectangular and cubemap output formats
#
# Usage instructions:
#   python fisheye.py'
#     Start interactive alignment GUI.
#   python fisheye.py -help
#     Print this help message.
#   python fisheye.py lens.cfg in1.jpg in2.jpg gui
#     Launch interactive GUI with specified default options
#   python fisheye.py lens.cfg in1.jpg in2.jpg rect=out.png
#     Render and save equirectangular panorama using specified
#     lens configuration and source images.'
#   python fisheye.py lens.cfg in1.jpg in2.jpg cube=out.png
#     Render and save cubemap panorama using specified
#     lens configuration and source images.
#
# Copyright (c) 2016 Alexander C. Utter
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#

import json
import numpy as np
import Tkinter as tk
import tkFileDialog
import tkMessageBox
import sys
import traceback
from copy import deepcopy
from math import pi
from PIL import Image, ImageTk
from scipy.optimize import minimize
from threading import Thread

# Create rotation matrix from an arbitrary quaternion.  See also:
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
def get_rotation_matrix(qq):
    # Normalize matrix and extract individual items.
    qq_norm = np.sqrt(np.sum(np.square(qq)))
    w = qq[0] / qq_norm
    x = qq[1] / qq_norm
    y = qq[2] / qq_norm
    z = qq[3] / qq_norm
    # Convert to rotation matrix.
    return np.matrix([[w*w+x*x-y*y-z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                      [2*x*y+2*w*z, w*w-x*x+y*y-z*z, 2*y*z-2*w*x],
                      [2*x*z-2*w*y, 2*y*z+2*w*x, w*w-x*x-y*y+z*z]], dtype='float32')

# Conjugate a quaternion to apply the opposite rotation.
def conj_qq(qq):
    return np.array([qq[0], -qq[1], -qq[2], -qq[3]])

# Multiply two quaternions:ab = (a0b0 - av dot bv; a0*bv + b0av + av cross bv)
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
def mul_qq(qa, qb):
    return np.array([qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3],
                     qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2],
                     qa[0]*qb[2] + qa[2]*qb[0] + qa[3]*qb[1] - qa[1]*qb[3],
                     qa[0]*qb[3] + qa[3]*qb[0] + qa[1]*qb[2] - qa[2]*qb[1]])

# Generate a normalized quaternion [W,X,Y,Z] from [X,Y,Z]
def norm_qq(x, y, z):
    rsq = x**2 + y**2 + z**2
    if rsq < 1:
        w = np.sqrt(1-rsq)
        return [w, x, y, z]
    else:
        r = np.sqrt(rsq)
        return [0, x/r, y/r, z/r]


# Return length of every column in an MxN matrix.
def matrix_len(x):
    #return np.sqrt(np.sum(np.square(x), axis=0))
    return np.linalg.norm(x, axis=0)

# Normalize an MxN matrix such that all N columns have unit length.
def matrix_norm(x):
    return x / (matrix_len(x) + 1e-9)


# Parameters for a fisheye lens, including its orientation.
class FisheyeLens:
    def __init__(self, rows=1024, cols=1024):
        # Fisheye lens parameters.
        self.fov_deg = 180
        self.radius_px = min(rows,cols) / 2
        # Pixel coordinates of the optical axis (X,Y).
        self.center_px = np.matrix([[cols/2], [rows/2]])
        # Quaternion mapping intended to actual optical axis.
        self.center_qq = [1, 0, 0, 0]

    def downsample(self, dsamp):
        self.radius_px /= dsamp
        self.center_px /= dsamp

    def get_x(self):
        return np.asscalar(self.center_px[0])

    def get_y(self):
        return np.asscalar(self.center_px[1])

    def to_dict(self):
        return {'cx':self.get_x(),
                'cy':self.get_y(),
                'cr':self.radius_px,
                'cf':self.fov_deg,
                'qw':self.center_qq[0],
                'qx':self.center_qq[1],
                'qy':self.center_qq[2],
                'qz':self.center_qq[3]}

    def from_dict(self, data):
        self.center_px[0] = data['cx']
        self.center_px[1] = data['cy']
        self.radius_px    = data['cr']
        self.fov_deg      = data['cf']
        self.center_qq[0] = data['qw']
        self.center_qq[1] = data['qx']
        self.center_qq[2] = data['qy']
        self.center_qq[3] = data['qz']

# Load or save lens configuration and alignment.
def load_config(file_obj, lens1, lens2):
    [data1, data2] = json.load(file_obj)
    lens1.from_dict(data1)
    lens2.from_dict(data2)

def save_config(file_obj, lens1, lens2):
    data = [lens1.to_dict(), lens2.to_dict()]
    json.dump(data, file_obj, indent=2, sort_keys=True)


# Fisheye source image, with lens and rotation parameters.
# Contains functions for extracting pixel data given direction vectors.
class FisheyeImage:
    # Load image file and set default parameters
    def __init__(self, src_file, lens=None):
        # Load the image file, and convert to a numpy matrix.
        self._update_img(Image.open(src_file))
        # Set lens parameters.
        if lens is None:
            self.lens = FisheyeLens(self.rows, self.cols)
        else:
            self.lens = lens

    # Update image matrix and corresponding size variables.
    def _update_img(self, img):
        self.img = np.array(img)
        self.rows = self.img.shape[0]
        self.cols = self.img.shape[1]
        self.clrs = self.img.shape[2]

    # Shrink source image and adjust lens accordingly.
    def downsample(self, dsamp):
        # Adjust lens parameters.
        self.lens.downsample(dsamp)
        # Determine the new image dimensions.
        # Note: PIL uses cols, rows whereas numpy uses rows, cols
        shape = (self.img.shape[1] / dsamp,     # Cols
                 self.img.shape[0] / dsamp)     # Rows
        # Convert matrix back to PIL Image and resample.
        img = Image.fromarray(self.img)
        img.thumbnail(shape, Image.BICUBIC)
        # Convert back and update size.
        self._update_img(img)

    # Given an 3xN array of "XYZ" vectors in panorama space (+X = Front),
    # convert each ray to 2xN coordinates in "UV" fisheye image space.
    def get_uv(self, xyz_vec):
        # Extract lens parameters of interest.
        fov_rad = self.lens.fov_deg * pi / 180
        fov_scale = np.float32(2 * self.lens.radius_px / fov_rad)
        # Normalize the input vector and rotate to match lens reference axes.
        xyz_rot = get_rotation_matrix(self.lens.center_qq) * matrix_norm(xyz_vec)
        # Convert to polar coordinates relative to lens boresight.
        # (In lens coordinates, unit vector's X axis gives boresight angle;
        #  normalize Y/Z to get a planar unit vector for the bearing.)
        # Note: Image +Y maps to 3D +Y, and image +X maps to 3D +Z.
        theta_rad = np.arccos(xyz_rot[0,:])
        proj_vec = matrix_norm(np.concatenate((xyz_rot[2,:], xyz_rot[1,:])))
        # Fisheye lens maps 3D angle to focal-plane radius.
        # TODO: Do we need a better model for lens distortion?
        rad_px = theta_rad * fov_scale
        # Convert back to focal-plane rectangular coordinates.
        uv = np.multiply(rad_px, proj_vec) + self.lens.center_px
        return np.asarray(uv + 0.5, dtype=int)

    # Given an 2xN array of UV pixel coordinates, check if each pixel is
    # within the fisheye field of view. Returns N-element boolean mask.
    def get_mask(self, uv_px):
        # Check whether each coordinate is within outer image bounds,
        # and within the illuminated area under the fisheye lens.
        x_mask = np.logical_and(0 <= uv_px[0], uv_px[0] < self.cols)
        y_mask = np.logical_and(0 <= uv_px[1], uv_px[1] < self.rows)
        # Check whether each coordinate is within the illuminated area.
        r_mask = matrix_len(uv_px - self.lens.center_px) < self.lens.radius_px
        # All three checks must pass to be considered visible.
        all_mask = np.logical_and(r_mask, np.logical_and(x_mask, y_mask))
        return np.squeeze(np.asarray(all_mask))

    # Given an 2xN array of UV pixel coordinates, return a weight score
    # that is proportional to the distance from the edge.
    def get_weight(self, uv_px):
        mm = self.get_mask(uv_px)
        rr = self.lens.radius_px - matrix_len(uv_px - self.lens.center_px)
        rr[~mm] = 0
        return rr

    # Given a 2xN array of UV pixel coordinates, return the value of each
    # corresponding pixel. Output format is Nx1 (grayscale) or Nx3 (color).
    # Pixels outside the fisheye's field of view are pure black (0) or (0,0,0).
    def get_pixels(self, uv_px):
        # Create output array with default pixel values.
        pcount = uv_px.shape[1]
        result = np.zeros((pcount, self.clrs), dtype=self.img.dtype)
        # Overwrite in-bounds pixels as specified above.
        self.add_pixels(uv_px, result)
        return result

    # Given a 2xN array of UV pixel coordinates, write the value of each
    # corresponding pixel to the linearized input/output image (Nx3).
    # Several weighting modes are available.
    def add_pixels(self, uv_px, img1d, weight=None):
        # Lookup row & column for each in-bounds coordinate.
        mask = self.get_mask(uv_px)
        xx = uv_px[0,mask]
        yy = uv_px[1,mask]
        # Update matrix according to assigned weight.
        if weight is None:
            img1d[mask] = self.img[yy,xx]
        elif np.isscalar(weight):
            img1d[mask] += self.img[yy,xx] * weight
        else:
            w1 = np.asmatrix(weight, dtype='float32')
            w3 = w1.transpose() * np.ones((1,3))
            img1d[mask] += np.multiply(self.img[yy,xx], w3[mask])


# A panorama image made from several FisheyeImage sources.
# TODO: Add support for supersampled anti-aliasing filters.
class PanoramaImage:
    def __init__(self, src_list):
        self.debug = True
        self.sources = src_list
        self.dtype = self.sources[0].img.dtype
        self.clrs = self.sources[0].clrs

    # Downsample each source image.
    def downsample(self, dsamp):
        for src in self.sources:
            src.downsample(dsamp)

    # Return a list of 'mode' strings suitable for render_xx() methods.
    def get_render_modes(self):
        return ['overwrite', 'align', 'blend']

    # Retrieve a scaled copy of lens parameters for the Nth source.
    def scale_lens(self, idx, scale=None):
        temp = deepcopy(self.sources[idx].lens)
        temp.downsample(1.0 / scale)
        return temp

    # Using current settings as an initial guess, use an iterative optimizer
    # to better align the source images.  Adjusts FOV of each lens, as well
    # as the rotation quaternions for all lenses except the first.
    # TODO: Implement a higher-order loop that iterates this step with
    #       progressively higher resolution.  (See also: create_panorama)
    # TODO: Find a better scoring heuristic.  Present solution always
    #       converges on either FOV=0 or FOV=9999, depending on wt_pixel.
    def optimize(self, psize=256, wt_pixel=1000, wt_blank=1000):
        # Precalculate raster-order XYZ coordinates at given resolution.
        [xyz, rows, cols] = self._get_equirectangular_raster(psize)
        # Scoring function gives bonus points per overlapping pixel.
        score = lambda svec: self._score(svec, xyz, wt_pixel, wt_blank)
        # Multivariable optimization using gradient-descent or similar.
        # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
        svec0 = self._get_state_vector()
        final = minimize(score, svec0, method='Nelder-Mead',
                         options={'xtol':1e-4, 'disp':True})
        # Store final lens parameters.
        self._set_state_vector(final.x)

    # Render combined panorama in equirectangular projection mode.
    # See also: https://en.wikipedia.org/wiki/Equirectangular_projection
    def render_equirectangular(self, out_size, mode='blend'):
        # Render the entire output in a single pass.
        [xyz, rows, cols] = self._get_equirectangular_raster(out_size)
        return Image.fromarray(self._render(xyz, rows, cols, mode))

    # Render combined panorama in cubemap projection mode.
    # See also: https://en.wikipedia.org/wiki/Cube_mapping
    def render_cubemap(self, out_size, mode='blend'):
        # Create coordinate arrays.
        cvec = np.arange(out_size, dtype='float32') - out_size/2        # Coordinate range [-S/2, S/2)
        vec0 = np.ones(out_size*out_size, dtype='float32') * out_size/2 # Constant vector +S/2
        vec1 = np.repeat(cvec, out_size)                                # Increment every N steps
        vec2 = np.tile(cvec, out_size)                                  # Sweep N times
        # Create XYZ coordinate vectors and render each cubemap face.
        render = lambda(xyz): self._render(xyz, out_size, out_size, mode)
        xm = render(np.matrix([-vec0, vec1, vec2]))     # -X face
        xp = render(np.matrix([vec0, vec1, -vec2]))     # +X face
        ym = render(np.matrix([-vec1, -vec0, vec2]))    # -Y face
        yp = render(np.matrix([vec1, vec0, vec2]))      # +Y face
        zm = render(np.matrix([-vec2, vec1, -vec0]))    # -Z face
        zp = render(np.matrix([vec2, vec1, vec0]))      # +Z face
        # Concatenate the individual faces in canonical order:
        # https://en.wikipedia.org/wiki/Cube_mapping#Memory_Addressing
        img_mat = np.concatenate([zp, zm, ym, yp, xm, xp], axis=0)
        return Image.fromarray(img_mat)

    # Get XYZ vectors for an equirectangular render, in raster order.
    # (Each row left to right, with rows concatenates from top to bottom.)
    def _get_equirectangular_raster(self, out_size):
        # Set image size (2x1 aspect ratio)
        rows = out_size
        cols = 2*out_size
        # Calculate longitude of each column.
        theta_x = np.linspace(-pi, pi, cols, endpoint=False, dtype='float32')
        cos_x = np.cos(theta_x).reshape(1,cols)
        sin_x = np.sin(theta_x).reshape(1,cols)
        # Calculate lattitude of each row.
        ystep = pi / rows
        theta_y = np.linspace(-pi/2 + ystep/2, pi/2 - ystep/2, rows, dtype='float32')
        cos_y = np.cos(theta_y).reshape(rows,1)
        sin_y = np.sin(theta_y).reshape(rows,1)
        # Calculate X, Y, and Z coordinates for each output pixel.
        x = cos_y * cos_x
        y = sin_y * np.ones((1,cols), dtype='float32')
        z = cos_y * sin_x
        # Vectorize the coordinates in raster order.
        xyz = np.matrix([x.ravel(), y.ravel(), z.ravel()])
        return [xyz, rows, cols]

    # Convert all lens parameters to a state vector. See also: optimize()
    def _get_state_vector(self):
        nsrc = len(self.sources)
        assert nsrc > 0
        svec = np.zeros(4*nsrc - 3)
        # First lens: Only the FOV is stored.
        svec[0] = self.sources[0].lens.fov_deg - 180
        # All other lenses: Store FOV and quaternion parameters.
        for n in range(1, nsrc):
            svec[4*n-3] = self.sources[n].lens.fov_deg - 180
            svec[4*n-2] = self.sources[n].lens.center_qq[1]
            svec[4*n-1] = self.sources[n].lens.center_qq[2]
            svec[4*n-0] = self.sources[n].lens.center_qq[3]
        return svec

    # Update lens parameters based on state vector.  See also: optimize()
    def _set_state_vector(self, svec):
        # Sanity check on input vector.
        nsrc = len(self.sources)
        assert len(svec) == (4*nsrc - 3)
        # First lens: Only the FOV is changed.
        self.sources[0].lens.fov_deg = svec[0] + 180
        # All other lenses: Update FOV and quaternion parameters.
        for n in range(1, nsrc):
            self.sources[n].lens.fov_deg = svec[4*n-3] + 180
            self.sources[n].lens.center_qq[1] = svec[4*n-2]
            self.sources[n].lens.center_qq[2] = svec[4*n-1]
            self.sources[n].lens.center_qq[3] = svec[4*n-0]

    # Add pixels from every source to form a complete output image.
    # Several blending modes are available. See also: get_render_modes()
    def _render(self, xyz, rows, cols, mode):
        # Allocate Nx3 or Nx1 "1D" pixel-list (raster-order).
        img1d = np.zeros((rows*cols, self.clrs), dtype='float32')
        # Determine rendering mode:
        if mode == 'overwrite':
            # Simplest mode: Draw first, then blindly overwrite second.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d)
        elif mode == 'align':
            # Alignment mode: Draw each one at 50% intensity.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d, 0.5)
        elif mode == 'blend':
            # Linear nearest-source blending.
            uv_list = []
            wt_list = []
            wt_total = np.zeros(rows*cols, dtype='float32')
            # Calculate per-image and total weight matrices.
            for src in self.sources:
                uv = src.get_uv(xyz)
                wt = src.get_weight(uv)
                uv_list.append(uv)
                wt_list.append(wt)
                wt_total += wt
            # Render overall image using calculated weights.
            for n in range(len(self.sources)):
                wt_norm = wt_list[n] / wt_total
                self.sources[n].add_pixels(uv_list[n], img1d, wt_norm)
        else:
            raise ValueError('Invalid render mode.')
        # Convert to fixed-point image matrix and return.
        img2d = np.reshape(img1d, (rows, cols, self.clrs))
        return np.asarray(img2d, dtype=self.dtype)

    # Compute a normalized alignment score, based on size of overlap and
    # the pixel-differences in that region.  Note: Lower = Better.
    def _score(self, svec, xyz, wt_pixel, wt_blank):
        # Update lens parameters from state vector.
        self._set_state_vector(svec)
        # Determine masks for each input image.
        uv0 = self.sources[0].get_uv(xyz)
        uv1 = self.sources[1].get_uv(xyz)
        wt0 = self.sources[0].get_weight(uv0) > 0
        wt1 = self.sources[1].get_weight(uv1) > 0
        # Count overlapping pixels.
        ovr_mask = np.logical_and(wt0, wt1)             # Overlapping pixel
        pix_count = np.sum(wt0) + np.sum(wt1)           # Total drawn pixels
        blk_count = np.sum(np.logical_and(~wt0, ~wt1))  # Number of blank pixels
        # Allocate Nx3 or Nx1 "1D" pixel-list (raster-order).
        pcount = max(xyz.shape)
        img1d = np.zeros((pcount, self.clrs), dtype='float32')
        # Render the difference image, overlapping region only.
        self.sources[0].add_pixels(uv0, img1d, 1.0*ovr_mask)
        self.sources[1].add_pixels(uv1, img1d, -1.0*ovr_mask)
        # Sum-of-differences.
        sum_sqd = np.sum(np.sum(np.sum(np.square(img1d))))
        # Compute overall score.  (Note: Higher = Better)
        score = sum_sqd + wt_blank * blk_count - wt_pixel * pix_count
        # (Debug) Print status information.
        if (self.debug):
            print str(svec) + ' --> ' + str(score)
        return score


# Tkinter GUI window for loading a fisheye image.
class FisheyeAlignmentGUI:
    def __init__(self, parent, src_file, lens):
        # Set flag once all window objects created.
        self.init_done = False
        # Final result is the lens object.
        self.lens = lens
        # Load the input file.
        self.img = Image.open(src_file)
        # Create frame for this window with two vertical panels...
        parent.wm_title('Fisheye Alignment')
        self.frame = tk.Frame(parent)
        self.controls = tk.Frame(self.frame)
        # Make sliders for adjusting the lens parameters quaternion.
        self.x = self._make_slider(self.controls, 0, 'Center-X (px)',
                                   lens.get_x(), self.img.size[0])
        self.y = self._make_slider(self.controls, 1, 'Center-Y (px)',
                                   lens.get_y(), self.img.size[1])
        self.r = self._make_slider(self.controls, 2, 'Radius (px)',
                                   lens.radius_px, self.img.size[0])
        self.f = self._make_slider(self.controls, 3, 'Field of view (deg)',
                                   lens.fov_deg, 240, res=0.1)
        # Create a frame for the preview image, which resizes based on the
        # outer frame but does not respond to the contained preview size.
        self.preview_frm = tk.Frame(self.frame)
        self.preview_frm.bind('<Configure>', self._update_callback)  # Update on resize
        # Create the canvas object for the preview image.
        self.preview = tk.Canvas(self.preview_frm)
        # Finish frame creation.
        self.controls.pack(side=tk.LEFT)
        self.preview.pack(fill=tk.BOTH, expand=1)
        self.preview_frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.frame.pack(fill=tk.BOTH, expand=1)
        # Render the image once at default size
        self.init_done = True
        self.update_preview((800,800))
        # Disable further size propagation.
        self.preview_frm.update()
        self.preview_frm.pack_propagate(0)

    # Redraw the preview image using latest GUI parameters.
    def update_preview(self, psize):
        # Safety check: Ignore calls during construction/destruction.
        if not self.init_done: return
        # Copy latest user settings to the lens object.
        self.lens.fov_deg = self.f.get()
        self.lens.radius_px = self.r.get()
        self.lens.center_px[0] = self.x.get()
        self.lens.center_px[1] = self.y.get()
        # Re-scale the image to match the canvas size.
        # Note: Make a copy first, because thumbnail() operates in-place.
        self.img_sc = self.img.copy()
        self.img_sc.thumbnail(psize, Image.NEAREST)
        self.img_tk = ImageTk.PhotoImage(self.img_sc)
        # Re-scale the x/y/r parameters to match the preview scale.
        pre_scale = float(psize[0]) / float(self.img.size[0])
        x = self.x.get() * pre_scale
        y = self.y.get() * pre_scale
        r = self.r.get() * pre_scale
        # Clear and redraw the canvas.
        self.preview.delete('all')
        self.preview.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.preview.create_oval(x-r, y-r, x+r, y+r,
                                 outline='#C00000', width=3)

    # Make a combined label/textbox/slider for a given variable:
    def _make_slider(self, parent, rowidx, label, inival, maxval, res=0.5):
        # Create shared variable and set initial value.
        tkvar = tk.DoubleVar()
        tkvar.set(inival)
        # Set a callback for whenever tkvar is changed.
        # (The 'command' callback on the SpinBox only applies to the buttons.)
        tkvar.trace('w', self._update_callback)
        # Create the Label, SpinBox, and Scale objects.
        label = tk.Label(parent, text=label)
        spbox = tk.Spinbox(parent,
            textvariable=tkvar,
            from_=0, to=maxval, increment=res)
        slide = tk.Scale(parent,
            orient=tk.HORIZONTAL,
            showvalue=0,
            variable=tkvar,
            from_=0, to=maxval, resolution=res)
        label.grid(row=rowidx, column=0)
        spbox.grid(row=rowidx, column=1)
        slide.grid(row=rowidx, column=2)
        return tkvar

    # Find the largest output size that fits within the given bounds and
    # matches the aspect ratio of the original source image.
    def _get_aspect_size(self, max_size):
        img_ratio = float(self.img.size[1]) / float(self.img.size[0])
        return (min(max_size[0], max_size[1] / img_ratio),
                min(max_size[1], max_size[0] * img_ratio))

    # Thin wrapper for update_preview(), used to strip Tkinter arguments.
    def _update_callback(self, *args):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the render size.  (Always 2:1 aspect ratio.)
        psize = self._get_aspect_size((self.preview_frm.winfo_width(),
                                       self.preview_frm.winfo_height()))
        # Render the preview at the given size.
        if psize[0] >= 10 and psize[1] >= 10:
            self.update_preview(psize)


# Tkinter GUI window for calibrating fisheye alignment.
class PanoramaAlignmentGUI:
    def __init__(self, parent, panorama, psize=512):
        self.init_done = False
        # Store source and preview size
        self.panorama = panorama
        # Create frame for this window with two vertical panels...
        parent.wm_title('Panorama Alignment')
        self.frame = tk.Frame(parent)
        self.controls = tk.Frame(self.frame)
        # Make a drop-menu to select the rendering mode.
        tk.Label(self.controls, text='Preview mode').grid(row=0, column=0, sticky=tk.W)
        self.mode = tk.StringVar()
        self.mode.set('align')
        self.mode.trace('w', self._update_callback)
        mode_list = self.panorama.get_render_modes()
        mode_drop = tk.OptionMenu(self.controls, self.mode, *mode_list)
        mode_drop.grid(row=0, column=1, columnspan=2, sticky='NESW')
        # Determine which axis marks the main 180 degree rotation.
        front_qq = panorama.sources[0].lens.center_qq
        back_qq  = panorama.sources[1].lens.center_qq
        diff_qq  = mul_qq(front_qq, back_qq)
        # Create the axis selection toggle. (Flip on Y or Z)
        self.flip_axis = tk.BooleanVar()
        self.flip_axis.trace('w', self._update_callback)
        if abs(diff_qq[2]) > abs(diff_qq[3]):
            self.flip_axis.set(False)
            flip_qq = [0,0,1,0]
        else:
            self.flip_axis.set(True)
            flip_qq = [0,0,0,1]
        tk.Label(self.controls, text='Flip axis').grid(row=1, column=0, sticky=tk.W)
        axis_chk = tk.Checkbutton(self.controls, variable=self.flip_axis)
        axis_chk.grid(row=1, column=1, columnspan=2, sticky='NESW')
        # Extract the (hopefully small) alignment offset.
        flip_conj = conj_qq(mul_qq(flip_qq, front_qq))
        align_qq = mul_qq(back_qq, flip_conj)
        # Make three sliders for adjusting the relative alignment.
        self.slide_rx = self._make_slider(self.controls, 2, 'Rotate X', front_qq[1])
        self.slide_ry = self._make_slider(self.controls, 3, 'Rotate Y', front_qq[2])
        self.slide_rz = self._make_slider(self.controls, 4, 'Rotate Z', front_qq[3])
        self.slide_ax = self._make_slider(self.controls, 5, 'Align X', align_qq[1])
        self.slide_ay = self._make_slider(self.controls, 6, 'Align Y', align_qq[2])
        self.slide_az = self._make_slider(self.controls, 7, 'Align Z', align_qq[3])
        # Finish control-frame creation.
        self.controls.pack(side=tk.LEFT)
        # Create a frame for the preview image, which resizes based on the
        # outer frame but does not respond to the contained preview size.
        self.preview_frm = tk.Frame(self.frame)
        self.preview_frm.bind('<Configure>', self._update_callback)  # Update on resize
        # Add the preview.
        self.preview_lbl = tk.Label(self.preview_frm)   # Label displays image
        self.preview_lbl.pack()
        self.preview_frm.pack(fill=tk.BOTH, expand=1)
        # Finish frame creation.
        self.frame.pack(fill=tk.BOTH, expand=1)
        # Render the image once at default size
        self.init_done = True
        self.update_preview(psize)
        # Disable further size propagation.
        self.preview_frm.update()
        self.preview_frm.pack_propagate(0)

    # Update the GUI preview using latest alignment parameters.
    def update_preview(self, psize):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the primary axis of rotation.
        if self.flip_axis.get():
            flip_qq = [0,0,0,1]
        else:
            flip_qq = [0,0,1,0]
        # Calculate the orientation of both lenses.
        front_qq = norm_qq(self.slide_rx.get(),
                           self.slide_ry.get(),
                           self.slide_rz.get())
        align_qq = norm_qq(self.slide_ax.get(),
                           self.slide_ay.get(),
                           self.slide_az.get())
        back_qq = mul_qq(align_qq, mul_qq(flip_qq, front_qq))
        self.panorama.sources[0].lens.center_qq = front_qq
        self.panorama.sources[1].lens.center_qq = back_qq
        # Render the preview.
        # Note: The Tk-Label doesn't maintain a reference to the image object.
        #       To avoid garbage-collection, keep one in this class.
        self.preview_img = ImageTk.PhotoImage(
            self.panorama.render_equirectangular(psize, self.mode.get()))
        # Assign the new icon.
        self.preview_lbl.configure(image=self.preview_img)

    # Find the largest output size that fits within the given bounds and
    # matches the 2:1 aspect ratio of the equirectangular preview.
    def _get_aspect_size(self, max_size):
        return (min(max_size[0], max_size[1] / 2),
                min(max_size[1], max_size[0] * 2))

    # Make a combined label/textbox/slider for a given variable:
    def _make_slider(self, parent, rowidx, label, inival):
        # Set limits and resolution.
        lim = 1.0
        res = 0.001
        # Create shared variable.
        tkvar = tk.DoubleVar()
        tkvar.set(inival)
        # Set a callback for whenever tkvar is changed.
        # (The 'command' callback on the SpinBox only applies to the buttons.)
        tkvar.trace('w', self._update_callback)
        # Create the Label, SpinBox, and Scale objects.
        label = tk.Label(parent, text=label)
        spbox = tk.Spinbox(parent,
            textvariable=tkvar,
            from_=-lim, to=lim, increment=res)
        slide = tk.Scale(parent,
            orient=tk.HORIZONTAL,
            showvalue=0,
            variable=tkvar,
            from_=-lim, to=lim, resolution=res)
        label.grid(row=rowidx, column=0, sticky='W')
        spbox.grid(row=rowidx, column=1)
        slide.grid(row=rowidx, column=2)
        return tkvar

    # Thin wrapper for update_preview(), used to strip Tkinter arguments.
    def _update_callback(self, *args):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the render size.  (Always 2:1 aspect ratio.)
        psize = min(self.preview_frm.winfo_width()/2,
                    self.preview_frm.winfo_height())
        # Render the preview at the given size.
        # TODO: Fudge factor of -2 avoids infinite resize loop.
        #       Is there a better way?
        if psize >= 10:
            self.update_preview(psize-2)


# Tkinter GUI window for end-to-end alignment and rendering.
class PanoramaGUI:
    def __init__(self, parent):
        # Store reference object for creating child dialogs.
        self.parent = parent
        self.win_lens1 = None
        self.win_lens2 = None
        self.win_align = None
        self.work_done = False
        self.work_error = None
        self.work_status = None
        # Create dummy lens configuration.
        self.lens1 = FisheyeLens()
        self.lens2 = FisheyeLens()
        self.lens2.center_qq = [0,0,1,0]  # Default flip along Y axis.
        # Create frame for this GUI.
        parent.wm_title('Panorama Creation Tool')
        frame = tk.Frame(parent)
        # Make file-selection inputs for the two images.
        img_frame = tk.LabelFrame(frame, text='Input Images')
        self.img1 = self._make_file_select(img_frame, 0, 'Image #1')
        self.img2 = self._make_file_select(img_frame, 1, 'Image #2')
        img_frame.pack()
        # Make buttons to load, save, and adjust the lens configuration.
        lens_frame = tk.LabelFrame(frame, text='Lens Configuration and Alignment')
        btn_lens1 = tk.Button(lens_frame, text='Lens 1', command=self._adjust_lens1)
        btn_lens2 = tk.Button(lens_frame, text='Lens 2', command=self._adjust_lens2)
        btn_align = tk.Button(lens_frame, text='Align', command=self._adjust_align)
        btn_auto = tk.Button(lens_frame, text='Auto', command=self._auto_align_start)
        btn_load = tk.Button(lens_frame, text='Load', command=self.load_config)
        btn_save = tk.Button(lens_frame, text='Save', command=self.save_config)
        btn_lens1.grid(row=0, column=0, sticky='NESW')
        btn_lens2.grid(row=0, column=1, sticky='NESW')
        btn_align.grid(row=0, column=2, sticky='NESW')
        btn_auto.grid(row=0, column=3, sticky='NESW')
        btn_load.grid(row=1, column=0, columnspan=2, sticky='NESW')
        btn_save.grid(row=1, column=2, columnspan=2, sticky='NESW')
        lens_frame.pack(fill=tk.BOTH)
        # Buttons to render the final output in different modes.
        out_frame = tk.LabelFrame(frame, text='Final output rendering')
        btn_rect = tk.Button(out_frame, text='Equirectangular',
                             command=self._render_rect)
        btn_cube = tk.Button(out_frame, text='Cubemap',
                             command=self._render_cube)
        btn_rect.pack(fill=tk.BOTH)
        btn_cube.pack(fill=tk.BOTH)
        out_frame.pack(fill=tk.BOTH)
        # Status indicator box.
        self.status = tk.Label(frame, relief=tk.SUNKEN,
                               text='Select input images to begin.')
        self.status.pack(fill=tk.BOTH)
        # Finish frame creation.
        frame.pack()

    # Helper function to destroy an object.
    def _destroy(self, obj):
        if obj is not None:
            obj.destroy()

    # Popup dialogs for each alignment step.
    def _adjust_lens1(self):
        self._destroy(self.win_lens1)
        try:
            self.win_lens1 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens1, self.img1.get(), self.lens1)
        except IOError:
            self._destroy(self.win_lens1)
            tkMessageBox.showerror('Error', 'Unable to read image file #1.')
        except:
            self._destroy(self.win_lens1)
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_lens2(self):
        self._destroy(self.win_lens2)
        try:
            self.win_lens2 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens2, self.img2.get(), self.lens2)
        except IOError:
            self._destroy(self.win_lens2)
            tkMessageBox.showerror('Error', 'Unable to read image file #2.')
        except:
            self._destroy(self.win_lens2)
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_align(self):
        self._destroy(self.win_align)
        try:
            pan = self._create_panorama()
            self.win_align = tk.Toplevel(self.parent)
            PanoramaAlignmentGUI(self.win_align, pan)
        except:
            self._destroy(self.win_align)
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    # Automatic alignment.
    # Use worker thread, because this may take a while.
    def _auto_align_start(self):
        try:
            # Create panorama object from within GUI thread, since it depends
            # on Tk variables which are NOT thread-safe.
            pan = self._create_panorama()
            # Display status message and display hourglass...
            self._set_status('Starting auto-alignment...', 'wait')
            # Create a new worker thread.
            work = Thread(target=self._auto_align_work, args=[pan])
            work.start()
            # Set a timer to periodically check for completion.
            self.parent.after(200, self._auto_align_timer)
        except:
            tkMessageBox.showerror('Auto-alignment error', traceback.format_exc())

    def _auto_align_work(self, pan):
        try:
            # Repeat alignment at progressively higher resolution.
            self._auto_align_step(pan, 16, 128, 'Stage 1/4')
            self._auto_align_step(pan,  8, 128, 'Stage 2/4')
            self._auto_align_step(pan,  4, 192, 'Stage 3/4')
            self._auto_align_step(pan,  2, 256, 'Stage 4/4')
            # Signal success!
            self.work_status = 'Auto-alignment completed.'
            self.work_error = None
            self.work_done = True
        except:
            # Signal error.
            self.work_status = 'Auto-alignment failed.'
            self.work_error = traceback.format_exc()
            self.work_done = True
                
    def _auto_align_step(self, pan, scale, psize, label):
        # Update status message.
        self.work_status = 'Auto-alignment: ' + str(label)
        # Create a panorama object at 1/scale times original resolution.
        pan_sc = deepcopy(pan)
        pan_sc.downsample(scale)
        # Run optimization, rendering each hypothesis at the given resolution.
        pan_sc.optimize(psize)
        # Update local lens parameters.
        # Note: These are not Tk variables, so are safe to change.
        self.lens1 = pan_sc.scale_lens(0, scale)
        self.lens2 = pan_sc.scale_lens(1, scale)

    # Timer callback object checks outputs from worker thread.
    # (Tkinter objects are NOT thread safe.)
    def _auto_align_timer(self, *args):
        # Check thread status.
        if self.work_done:
            # Update status message, with popup on error.
            if self.work_status is not None:
                self._set_status(self.work_status)
            if self.work_error is not None:
                self._set_status('Auto-alignment failed.')
                tkMessageBox.showerror('Auto-alignment error', self.work_error)
            # Clear the 'done' flag for future runs.
            self.work_done = False
        else:
            # Update status message and keep hourglass.
            if self.work_status is not None:
                self._set_status(self.work_status, 'wait')
            # Reset timer to be called again.
            self.parent.after(200, self._auto_align_timer)

    # Create panorama object using current settings.
    def _create_panorama(self):
        img1 = FisheyeImage(self.img1.get(), self.lens1)
        img2 = FisheyeImage(self.img2.get(), self.lens2)
        return PanoramaImage((img1, img2))

    # Load or save lens configuration and alignment.
    def load_config(self, filename=None):
        if filename is None:
            file_obj = tkFileDialog.askopenfile()
            if file_obj is None: return
        else:
            file_obj = open(filename, 'r')
        try:
            load_config(file_obj, self.lens1, self.lens2)
        except:
            tkMessageBox.showerror('Config load error', traceback.format_exc())
                

    def save_config(self, filename=None):
        if filename is None:
            file_obj = tkFileDialog.asksaveasfile()
            if file_obj is None: return
        else:
            file_obj = open(filename, 'w')
        try:
            save_config(file_obj, self.lens1, self.lens2)
        except:
            tkMessageBox.showerror('Config save error', traceback.format_exc())

    # Render and save output in various modes.
    def _render_generic(self, render_type, render_size=1024):
        # Popup asks user for output file.
        file_obj = tkFileDialog.asksaveasfile(mode='wb')
        # Abort if user clicks 'cancel'.
        if file_obj is None: return
        # Proceed with rendering...
        self._set_status('Rendering image: ' + file_obj.name, 'wait')
        try:
            panorama = self._create_panorama()
            render_func = getattr(panorama, render_type)
            render_func(render_size).save(file_obj)
            self._set_status('Done!')
        except:
            tkMessageBox.showerror('Render error', traceback.format_exc())
            self._set_status('Render failed.')

    def _render_rect(self):
        self._render_generic('render_equirectangular')

    def _render_cube(self):
        self._render_generic('render_cubemap')

    # Callback to create a file-selection popup.
    def _file_select(self, tkstr):
        result = tkFileDialog.askopenfile()
        if result is not None:
            tkstr.set(result.name)
            result.close()

    # Make a combined label/textbox/slider for a given variable:
    def _make_file_select(self, parent, rowidx, label):
        # Create string variable.
        tkstr = tk.StringVar()
        # Create callback event handler.
        cmd = lambda: self._file_select(tkstr)
        # Create the Label, Entry, and Button objects.
        label = tk.Label(parent, text=label)
        entry = tk.Entry(parent, textvariable=tkstr)
        button = tk.Button(parent, text='...', command=cmd)
        label.grid(row=rowidx, column=0, sticky='W')
        entry.grid(row=rowidx, column=1)
        button.grid(row=rowidx, column=2)
        return tkstr

    # Set status text, and optionally update cursor.
    def _set_status(self, status, cursor='arrow'):
        self.parent.config(cursor=cursor)
        self.status.configure(text=status)

def launch_tk_gui(flens='', fimg1='', fimg2=''):
    # Create TK root object and GUI window.
    root = tk.Tk()
    gui = PanoramaGUI(root)
    # Load parameters if specified.
    if flens is not None and len(flens) > 0:
        gui.load_config(flens)
    if fimg1 is not None and len(fimg1) > 0:
        gui.img1.set(fimg1)
    if fimg2 is not None and len(fimg2) > 0:
        gui.img2.set(fimg2)
    # Start main loop.
    root.mainloop()

if __name__ == "__main__":
    # If we have exactly four arguments, run command-line version.
    if len(sys.argv) == 5 and sys.argv[4].startswith('gui'):
        # Special case for interactive mode.
        launch_tk_gui(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        # First argument is the lens alignment file.
        lens1 = FisheyeLens()
        lens2 = FisheyeLens()
        cfg = open(sys.argv[1], 'r')
        load_config(cfg, lens1, lens2)
        # Second and third arguments are the source files.
        img1 = FisheyeImage(sys.argv[2], lens1)
        img2 = FisheyeImage(sys.argv[3], lens2)
        # Fourth argument is the mode and output filename.
        if sys.argv[4].startswith('cube='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_cubemap(1024).save(out)
        elif sys.argv[4].startswith('rect='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_equirectangular(1024).save(out)
        else:
            print 'Unrecognized render mode (cube=, rect=, gui)'
    elif len(sys.argv) > 1:
        # If requested, print command-line usage information.
        print 'Usage instructions:'
        print '  python fisheye.py'
        print '    Start interactive alignment GUI.'
        print '  python fisheye.py -help'
        print '    Print this help message.'
        print '  python fisheye.py lens.cfg in1.jpg in2.jpg gui'
        print '    Launch interactive GUI with specified default options'
        print '  python fisheye.py lens.cfg in1.jpg in2.jpg rect=out.png'
        print '    Render and save equirectangular panorama using specified'
        print '    lens configuration and source images.'
        print '  python fisheye.py lens.cfg in1.jpg in2.jpg cube=out.png'
        print '    Render and save cubemap panorama using specified'
        print '    lens configuration and source images.'
    else:
        # Otherwise, start the interactive GUI with all fields blank.
        launch_tk_gui()
