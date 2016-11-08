# -*- coding: utf-8 -*-
# Dual-fisheye to 360-photo conversion tool
# Supports equirectangular and cubemap output formats
#
# Usage instructions:
#   python fisheye.py'
#     Start interactive alignment GUI.
#   python fisheye.py -help
#     Print this help message.
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
from math import pi
from PIL import Image, ImageTk

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
        self.img = np.array(Image.open(src_file))
        self.rows = self.img.shape[0]
        self.cols = self.img.shape[1]
        self.clrs = self.img.shape[2]
        # Set lens parameters.
        if lens is None:
            self.lens = FisheyeLens(self.rows, self.cols)
        else:
            self.lens = lens

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
        rr = self.lens.radius_px - matrix_len(uv_px - self.lens.center_px)
        rr[rr < 0] = 0
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
            w3 = np.asmatrix(weight).transpose() * np.ones((1,3))
            img1d[mask] += np.multiply(self.img[yy,xx], w3[mask])


# A panorama image made from several FisheyeImage sources.
# TODO: Add support for supersampled anti-aliasing filters.
# TODO: Split rendering logic into sections, to avoid memory limits.
class PanoramaImage:
    def __init__(self, src_list):
        self.sources = src_list
        self.dtype = self.sources[0].img.dtype
        self.clrs = self.sources[0].clrs

    def get_render_methods(self):
        return ['overwrite', 'align', 'blend']

    def render_equirectangular(self, out_size, method='blend'):
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
        # Vectorize and accumulate source images
        xyz = np.matrix([x.ravel(), y.ravel(), z.ravel()])
        # Render the entire output in a single pass.
        return Image.fromarray(self._render(xyz, rows, cols, method))

    def render_cubemap(self, out_size, method='blend'):
        # Create coordinate arrays.
        cvec = np.arange(out_size, dtype='float32') - out_size/2        # Coordinate range [-S/2, S/2)
        vec0 = np.ones(out_size*out_size, dtype='float32') * out_size/2 # Constant vector +S/2
        vec1 = np.repeat(cvec, out_size)                                # Increment every N steps
        vec2 = np.tile(cvec, out_size)                                  # Sweep N times
        # Create XYZ coordinate vectors and render each cubemap face.
        render = lambda(xyz): self._render(xyz, out_size, out_size, method)
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

    # Add pixels from every source to the given output image.
    def _render(self, xyz, rows, cols, method):
        # Allocate Nx3 or Nx1 "1D" pixel-list (raster-order).
        img1d = np.zeros((rows*cols, self.clrs), dtype='float32')
        # Determine rendering method:
        if method == 'overwrite':
            # Simplest method: Draw first, then blindly overwrite second.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d)
        elif method == 'align':
            # Alignment mode: Draw each one at 50% intensity.
            for src in self.sources:
                uv = src.get_uv(xyz)
                src.add_pixels(uv, img1d, 0.5)
        elif method == 'blend':
            # Nearest-source blending: Gradual transition in overlap range.
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
                self.sources[n].add_pixels(uv_list[n], img1d, wt_list[n] / wt_total)
        else:
            raise ValueError('Invalid render method.')
        # Convert to fixed-point image matrix and return.
        img2d = np.reshape(img1d, (rows, cols, self.clrs))
        return np.asarray(img2d, dtype=self.dtype)


# Tkinter GUI window for loading a fisheye image.
class FisheyeAlignmentGUI:
    def __init__(self, parent, src_file, lens):
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
                                   lens.fov_deg, 240)
        # Create the preview frame.
        pre_size = self._get_aspect_size((800,800))
        self.preview = tk.Canvas(self.frame, width=pre_size[0], height=pre_size[1])
        self.preview.bind('<Configure>', self.update_preview)  # Update on resize
        # Finish frame creation.
        self.controls.pack(side=tk.LEFT)
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.frame.pack()
        self.update_preview()

    # Redraw the preview image using latest GUI parameters.
    def update_preview(self, *args):
        # Safety check: Ignore calls during construction/destruction.
        if not hasattr(self, 'preview') or self.preview is None:
            return
        pre_size = (float(self.preview.winfo_width()),
                    float(self.preview.winfo_height()))
        if pre_size[0] < 10 or pre_size[1] < 10:
            return
        # Copy latest result to the lens object.
        self.lens.fov_deg = self.f.get()
        self.lens.radius_px = self.r.get()
        self.lens.center_px[0] = self.x.get()
        self.lens.center_px[1] = self.y.get()
        # Re-scale the image to match the canvas size.
        # Note: Make a copy first, because thumbnail() operates in-place.
        pre_size = self._get_aspect_size(pre_size)
        self.img_sc = self.img.copy()
        self.img_sc.thumbnail(pre_size, Image.NEAREST)
        self.img_tk = ImageTk.PhotoImage(self.img_sc)
        # Re-scale the x/y/r parameters to match the preview scale.
        pre_scale = float(pre_size[0]) / float(self.img.size[0])
        x = self.x.get() * pre_scale
        y = self.y.get() * pre_scale
        r = self.r.get() * pre_scale
        # Clear and redraw the canvas.
        self.preview.delete('all')
        self.preview.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        self.preview.create_oval(x-r, y-r, x+r, y+r,
                                 outline='#C00000', width=3)

    # Make a combined label/textbox/slider for a given variable:
    def _make_slider(self, parent, rowidx, label, inival, maxval):
        # Create shared variable and set initial value.
        tkvar = tk.DoubleVar()
        tkvar.set(inival)
        # Set a callback for whenever tkvar is changed.
        # (The 'command' callback on the SpinBox only applies to the buttons.)
        tkvar.trace('w', self.update_preview)
        # Create the Label, SpinBox, and Scale objects.
        label = tk.Label(parent, text=label)
        spbox = tk.Spinbox(parent,
            textvariable=tkvar,
            from_=0, to=maxval)
        slide = tk.Scale(parent,
            orient=tk.HORIZONTAL,
            showvalue=0,
            variable=tkvar,
            from_=0, to=maxval, resolution=0.2)
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


# Tkinter GUI window for calibrating fisheye alignment.
class PanoramaAlignmentGUI:
    def __init__(self, parent, panorama, psize=512):
        self.init_done = False
        # Store source and preview size
        self.panorama = panorama
        self.psize = psize
        # Create frame for this window with two vertical panels...
        parent.wm_title('Panorama Alignment')
        self.frame = tk.Frame(parent)
        self.controls = tk.Frame(self.frame)
        # Make a drop-menu to select the rendering mode.
        tk.Label(self.controls, text='Preview mode').grid(row=0, column=0, sticky=tk.W)
        self.method = tk.StringVar()
        self.method.set('align')
        self.method.trace('w', self.update_preview)
        method_list = self.panorama.get_render_methods()
        method_drop = tk.OptionMenu(self.controls, self.method, *method_list)
        method_drop.grid(row=0, column=1, columnspan=2, sticky='NESW')
        # Determine which axis marks the main 180 degree rotation.
        front_qq = panorama.sources[0].lens.center_qq
        back_qq  = panorama.sources[1].lens.center_qq
        diff_qq  = mul_qq(front_qq, back_qq)
        # Create the axis selection toggle. (Flip on Y or Z)
        self.flip_axis = tk.BooleanVar()
        self.flip_axis.trace('w', self.update_preview)
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
        # TODO: There seems to be a bug in this, likely multiply ordering?
        flip_conj = conj_qq(mul_qq(flip_qq, front_qq))
        align_qq = conj_qq(mul_qq(back_qq, flip_conj))
        # Make three sliders for adjusting the relative alignment.
        self.slide_rx = self._make_slider(self.controls, 2, 'Rotate X', front_qq[1])
        self.slide_ry = self._make_slider(self.controls, 3, 'Rotate Y', front_qq[2])
        self.slide_rz = self._make_slider(self.controls, 4, 'Rotate Z', front_qq[3])
        self.slide_ax = self._make_slider(self.controls, 5, 'Align X', align_qq[1])
        self.slide_ay = self._make_slider(self.controls, 6, 'Align Y', align_qq[2])
        self.slide_az = self._make_slider(self.controls, 7, 'Align Z', align_qq[3])
        # Finish control-frame creation.
        self.controls.pack(side=tk.LEFT)
        # Add the preview.
        self.preview_lbl = tk.Label(self.frame)
        self.init_done = True
        self.update_preview()
        self.preview_lbl.pack()
        # Finish frame creation.
        self.frame.pack()

    def update_preview(self, *args):
        # Sanity check that initialization is completed:
        if not self.init_done: return
        # Determine the primary axis of rotation.
        axis_lbl = self.flip_axis.get()
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
        back_qq = mul_qq(mul_qq(flip_qq, front_qq), align_qq)
        self.panorama.sources[0].lens.center_qq = front_qq
        self.panorama.sources[1].lens.center_qq = back_qq
        # Render the preview.
        # Note: The Tk-Label doesn't maintain a reference to the image object.
        #       To avoid garbage-collection, keep one in this class.
        self.preview_img = ImageTk.PhotoImage(
            self.panorama.render_equirectangular(self.psize, self.method.get()))
        # Assign the new icon.
        self.preview_lbl.configure(image=self.preview_img)

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
        tkvar.trace('w', self.update_preview)
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


# Tkinter GUI window for end-to-end alignment and rendering.
class PanoramaGUI:
    def __init__(self, parent):
        # Store reference object for creating child dialogs.
        self.parent = parent
        self.win_lens1 = None
        self.win_lens2 = None
        self.win_align = None
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
        btn_load = tk.Button(lens_frame, text='Load', command=self._load_config)
        btn_save = tk.Button(lens_frame, text='Save', command=self._save_config)
        btn_lens1.grid(row=0, column=0, sticky='NESW')
        btn_lens2.grid(row=0, column=1, sticky='NESW')
        btn_align.grid(row=0, column=2, sticky='NESW')
        btn_load.grid(row=1, column=0, columnspan=2, sticky='NESW')
        btn_save.grid(row=1, column=2, sticky='NESW')
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

    # Popup dialogs for each alignment step.
    def _adjust_lens1(self):
        if self.win_align is not None:
            self.win_lens1.destroy()
        try:
            self.win_lens1 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens1, self.img1.get(), self.lens1)
        except IOError:
            self.win_lens1.destroy()
            tkMessageBox.showerror('Error', 'Unable to read image file #1.')
        except:
            self.win_lens1.destroy()
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_lens2(self):
        if self.win_align is not None:
            self.win_lens2.destroy()
        try:
            self.win_lens2 = tk.Toplevel(self.parent)
            FisheyeAlignmentGUI(self.win_lens2, self.img2.get(), self.lens2)
        except IOError:
            self.win_lens1.destroy()
            tkMessageBox.showerror('Error', 'Unable to read image file #2.')
        except:
            self.win_lens2.destroy()
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    def _adjust_align(self):
        if self.win_align is not None:
            self.win_align.destroy()
        try:
            pan = self._create_panorama()
            self.win_align = tk.Toplevel(self.parent)
            PanoramaAlignmentGUI(self.win_align, pan)
        except:
            self.win_align.destroy()
            tkMessageBox.showerror('Dialog creation error', traceback.format_exc())

    # Create panorama object using current settings.
    def _create_panorama(self):
        img1 = FisheyeImage(self.img1.get(), self.lens1)
        img2 = FisheyeImage(self.img2.get(), self.lens2)
        return PanoramaImage((img1, img2))

    # Load or save lens configuration and alignment.
    def _load_config(self):
        file_obj = tkFileDialog.askopenfile()
        if file_obj is None: return
        try:
            load_config(file_obj, self.lens1, self.lens2)
        except:
            tkMessageBox.showerror('Config load error', traceback.format_exc())
                

    def _save_config(self):
        file_obj = tkFileDialog.asksaveasfile()
        if file_obj is None: return
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


if __name__ == "__main__":
    # If we have exactly four arguments, run command-line version.
    if len(sys.argv) == 5:
        # First argument is the lens alignment file.
        lens1 = FisheyeLens()
        lens2 = FisheyeLens()
        cfg = file.open(sys.argv[1])
        load_config(cfg, lens1, lens2)
        # Second and third arguments are the source files.
        img1 = FisheyeImage(sys.argv[2], lens1)
        img2 = FisheyeImage(sys.argv[3], lens2)
        # Fourth argument is the mode and output filename.
        if sys.argv[4].startswith('cube='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_cubemap(1024).save(out)
            sys.exit(0)
        elif sys.argv[4].startswith('rect='):
            out = sys.argv[5:]
            pan = PanoramaImage((img1, img2))
            pan.render_equirectangular(1024).save(out)
            sys.exit(0)
        else:
            print 'Unrecognized render mode (cube=, rect=)'
            sys.exit(1)

    # If requested, print command-line usage information.
    if len(sys.argv) > 1:
        print 'Usage instructions:'
        print '  python fisheye.py'
        print '    Start interactive alignment GUI.'
        print '  python fisheye.py -help'
        print '    Print this help message.'
        print '  python fisheye.py lens.cfg in1.jpg in2.jpg rect=out.png'
        print '    Render and save equirectangular panorama using specified'
        print '    lens configuration and source images.'
        print '  python fisheye.py lens.cfg in1.jpg in2.jpg cube=out.png'
        print '    Render and save cubemap panorama using specified'
        print '    lens configuration and source images.'
        sys.exit(2)

    # Otherwise, start the interactive GUI.
    root = tk.Tk()
    PanoramaGUI(root)
    root.mainloop()
