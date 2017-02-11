### Copyright: Peter Williams (2012) - All rights reserved
###
### This program is free software; you can redistribute it and/or modify
### it under the terms of the GNU General Public License as published by
### the Free Software Foundation; version 2 of the License only.
###
### This program is distributed in the hope that it will be useful,
### but WITHOUT ANY WARRANTY; without even the implied warranty of
### MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
### GNU General Public License for more details.
###
### You should have received a copy of the GNU General Public License
### along with this program; if not, write to the Free Software
### Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

"""
Manipulate Gdk.Pixbuf objects for fun and pleasure
"""

import collections
import math
import fractions
import array

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
from gi.repository import GObject
from gi.repository import GdkPixbuf

from ..bab import mathx
from ..bab import nmd_tuples
from ..bab import options

from ..gtx import rgb_math


def array_scaled_to_value(rgb, new_value):
    cur_value = rgb_math.rgb_array_value(rgb)
    return array.array(rgb.typecode, (int(rgb[i] * new_value / cur_value + 0.5) for i in range(3)))


class ValueLimitCriteria:
    slots = ("__n_values", "__c_values", "__value_rgbs")

    def __init__(self, n_values):
        self.set_n_values(n_values)

    @property
    def n_values(self):
        return self.__n_values

    @property
    def c_values(self):
        return self.__c_values

    def set_n_values(self, n_values):
        self.__n_values = n_values
        self.__c_values = tuple((i / (n_values - 1) for i in range(n_values)))

    def get_value_rgbs_for_typecode(self, typecode):
        return tuple((rgb_math.proportions_to_array((value, value, value), typecode) for value in self.__c_values))

    def get_value_index(self, rgb):
        value = rgb_math.rgb_array_value(rgb)
        return int(value * (self.__n_values - 1) + 0.5)

class HueLimitCriteria:
    __slots__ = ("__n_hues", "__hues", "__step")
    def __init__(self, n_hues):
        self.set_n_hues(n_hues)

    @property
    def n_hues(self):
        return self.__n_hues

    @property
    def hues(self):
        return self.__hues

    def set_n_hues(self, n_hues):
        self.__n_hues = n_hues
        self.__step = 2 * math.pi / n_hues
        angles = (mathx.Angle.normalize(self.__step * i) for i in range(n_hues))
        self.__hues = [rgb_math.HueAngle(angle) for angle in angles]

    def get_hue_index(self, hue):
        if hue.is_grey:
            return None
        if hue.angle > 0.0:
            return int(round(float(hue.angle) / self.__step))
        else:
            return int(round((float(hue.angle) + 2 * math.pi) / self.__step)) % self.__n_hues


RGB_PIXEL = collections.namedtuple("RGB_PIXEL", ["rgb", "hue"])
RGBA_PIXEL = collections.namedtuple("RGBA_PIXEL", ["rgb", "hue", "alpha"])

class PixBufRow:
    def __init__(self, data, start, end, nc=3):
        self.__has_alpha = nc == 4
        self.__typecode = data.typecode
        chunks = (data[i:i + nc] for i in range(start, end, nc))
        if self.__has_alpha:
            self.__pixels = [RGBA_PIXEL(chunk[:3], rgb_math.HueAngle.from_rgb(chunk[:3]), chunk[3]) for chunk in chunks]
        else:
            self.__pixels = [RGB_PIXEL(chunk[:3], rgb_math.HueAngle.from_rgb(chunk[:3])) for chunk in chunks]

    def __iter__(self):
        return (pixel for pixel in self.__pixels)

    @property
    def typecode(self):
        return self.__typecode

    @property
    def rgbs(self):
        return (pixel.rgb for pixel in self.__pixels)

    @property
    def hues(self):
        return (pixel.hue for pixel in self.__pixels)

    @property
    def width(self):
        return len(self.__pixels)

    @property
    def has_alpha(self):
        return self.__has_alpha


BPS_TYPECODE = {8 : "B", 16 : "H", 32 : "L"}

class RGBHImage(GObject.GObject):
    """
    An object containing a RGB and Hue array representing a Pixbuf
    """
    NPR = 50 # the number of progress reports to make during a loop

    def __init__(self, pixbuf=None):
        GObject.GObject.__init__(self)
        self.__size = nmd_tuples.WH(width=0, height=0)
        self.__pixel_rows = None
        self.__bits_per_sample = 8
        if pixbuf is not None:
            self.set_from_pixbuf(pixbuf)

    @property
    def size(self):
        """
        The size of this image as an instance nmd_tuples.WH
        """
        return self.__size

    @property
    def typecode(self):
        return BPS_TYPECODE[self.__bits_per_sample]

    def __getitem__(self, index):
        """
        Get the row with the given index
        """
        return self.__pixel_rows[index]

    def set_from_pixbuf(self, pixbuf):
        size = pixbuf.get_width() * pixbuf.get_height()
        if size > 640 * 640:
            # Scale down large images
            ar = fractions.Fraction(pixbuf.get_width(), pixbuf.get_height())
            if ar > 1:
                new_w = int(640 * math.sqrt(ar) + 0.5)
                new_h = int(new_w / ar + 0.5)
            else:
                new_h = int(640 / math.sqrt(ar) + 0.5)
                new_w = int(new_h * ar + 0.5)
            pixbuf = pixbuf.scale_simple(new_w, new_h, GdkPixbuf.InterpType.BILINEAR)
        w, h = (pixbuf.get_width(), pixbuf.get_height())
        self.__size = nmd_tuples.WH(width=w, height=h)
        self.__bits_per_sample = pixbuf.get_bits_per_sample()
        self.__n_channels = pixbuf.get_n_channels()
        self.__rowstride = pixbuf.get_rowstride()
        data = array.array(BPS_TYPECODE[self.__bits_per_sample])
        data.frombytes(pixbuf.get_pixels())
        self.__pixel_rows = []
        pr_step = h / self.NPR
        next_pr_due = 0
        for j in range(h):
            if j >= next_pr_due:
                self.emit("progress-made", fractions.Fraction(j, h))
                next_pr_due += pr_step
            start = j * self.__rowstride
            self.__pixel_rows.append(PixBufRow(data, start, start + w * self.__n_channels, self.__n_channels))
        self.emit("progress-made", fractions.Fraction(1))

    def get_mapped_pixbuf(self, map_to_flat_bytes):
        if self.__pixel_rows is None:
            return None
        bytes_per_row = self.__size.width * self.__n_channels * self.__bits_per_sample // 8
        padding = b"\000" * (self.__rowstride - bytes_per_row)
        data = b""
        pr_step = len(self.__pixel_rows) / self.NPR
        next_pr_due = 0
        for row_n, pixel_row in enumerate(self.__pixel_rows):
            if row_n >= next_pr_due:
                self.emit("progress-made", fractions.Fraction(row_n, self.__size.height))
                next_pr_due += pr_step
            data += map_to_flat_bytes(pixel_row)
            data += padding
        self.emit("progress-made", fractions.Fraction(1))
        mapped_pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            data=GLib.Bytes(data),
            colorspace=GdkPixbuf.Colorspace.RGB,
            has_alpha=self.__n_channels == 4,
            bits_per_sample=self.__bits_per_sample,
            width=self.__size.width,
            height=self.__size.height,
            rowstride=self.__rowstride
        )
        return mapped_pixbuf

    def get_pixbuf(self):
        """
        Return a Gdk.Pixbuf representation of the image
        """
        def map_to_flat_bytes(pbr):
            ta = array.array(pbr.typecode)
            for pixel in pbr:
                ta.extend(pixel.rgb)
            return ta.tobytes()
        return self.get_mapped_pixbuf(map_to_flat_bytes)
GObject.type_register(RGBHImage)
GObject.signal_new("progress-made", RGBHImage, GObject.SIGNAL_RUN_LAST, GObject.TYPE_NONE, (GObject.TYPE_PYOBJECT,))

class Transformer:
    """A factory to transform an RGBHImage to a GdkPixbuf
    """
    LABEL = _("Raw")

    def check_cached_values(self, typecode):
        pass

    @staticmethod
    def transformed_rgb(pixel):
        # default is to do nothing
        return pixel.rgb

    def transform_row_to_bytes(self, pbr):
        ta = array.array(pbr.typecode)
        if pbr.has_alpha:
            for pixel in pbr:
                ta.extend(self.transformed_rgb(pixel))
                ta.append(pixel.alpha)
        else:
            for pixel in pbr:
                ta.extend(self.transformed_rgb(pixel))
        return ta.tobytes()

    def transformed_pixbuf(self, rgbh_image):
        self.check_cached_values(rgbh_image.typecode)
        return rgbh_image.get_mapped_pixbuf(self.transform_row_to_bytes)


class TransformerNotan(Transformer):
    LABEL = _("Notan")

    def __init__(self, threshold=None):
        self.__threshold = fractions.Fraction(2, 10) if threshold is None else threshold
        self.check_cached_values(BPS_TYPECODE[8])

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, value):
        self.__threshold = value

    def check_cached_values(self, typecode):
        if not hasattr(self, "BLACK") or self.BLACK.typecode != typecode:
            self.BLACK = rgb_math.proportions_to_array((0.0, 0.0, 0.0), typecode)
            self.WHITE = rgb_math.proportions_to_array((1.0, 1.0, 1.0), typecode)

    def transformed_rgb(self, pixel):
        return self.BLACK if rgb_math.rgb_array_value(pixel.rgb) <= self.__threshold else self.WHITE


class TransformerMonotone(Transformer):
    LABEL = _("Monotone")

    @staticmethod
    def transformed_rgb(pixel):
        val = (sum(pixel.rgb) + 1) // 3
        return array.array(pixel.rgb.typecode, (val, val, val))


class TransformerMonotoneRestrictedValue(Transformer):
    LABEL = _("Restricted Value (Monotone)")

    def __init__(self, num_value_levels=11):
        self.__vlc = ValueLimitCriteria(11)
        self.check_cached_values(BPS_TYPECODE[8])

    @property
    def num_value_levels(self):
        return self.__vlc.n_values

    @num_value_levels.setter
    def num_value_levels(self, value):
        self.__vlc.set_n_levels(value)

    def check_cached_values(self, typecode):
        if not hasattr(self, "value_rgbs") or self.value_rgbs[0].typecode != typecode:
            self.value_rgbs = self.__vlc.get_value_rgbs_for_typecode(typecode)

    def transformed_rgb(self, pixel):
        return self.value_rgbs[self.__vlc.get_value_index(pixel.rgb)]


class TransformerColourRestrictedValue(Transformer):
    LABEL = _("Restricted Value")

    def __init__(self, num_value_levels=11):
        self.__vlc = ValueLimitCriteria(11)
        self.check_cached_values(BPS_TYPECODE[8])

    @property
    def num_value_levels(self):
        return self.__vlc.n_values

    @num_value_levels.setter
    def num_value_levels(self, value):
        self.__vlc.set_n_levels(value)

    def check_cached_values(self, typecode):
        if not hasattr(self, "BLACK") or self.BLACK.typecode != typecode:
            self.BLACK = rgb_math.proportions_to_array((0.0, 0.0, 0.0), typecode)
            self.WHITE = rgb_math.proportions_to_array((1.0, 1.0, 1.0), typecode)

    def transformed_rgb(self, pixel):
        index = self.__vlc.get_value_index(pixel.rgb)
        if index == 0:
            return self.BLACK
        elif index == self.__vlc.n_values - 1:
            return self.WHITE
        try:
            return array_scaled_to_value(pixel.rgb, self.__vlc.c_values[index])
        except OverflowError:
            return pixel.hue.max_chroma_rgb_array_with_value(self.__vlc.c_values[index], pixel.rgb.typecode)


class TransformerRestrictedHue(Transformer):
    LABEL = _("Restricted Hue")

    def __init__(self, num_hues=6):
        self.__hlc = HueLimitCriteria(num_hues)

    @property
    def num_hues(self):
        return self.__hlc.n_hues

    @num_hues.setter
    def num_hues(self, value):
        self.__hlc.set_n_hues(value)

    def transformed_rgb(self, pixel):
        index = self.__hlc.get_hue_index(pixel.hue)
        if index is None:
            return pixel.rgb
        limited_hue = self.__hlc.hues[index]
        if (len(pixel.rgb) - pixel.rgb.count(0)) == 2:
            return limited_hue.max_chroma_rgb_array_with_value(rgb_math.rgb_array_value(pixel.rgb), pixel.rgb.typecode)
        else:
            return array.array(pixel.rgb.typecode, rgb_math.rotate_rgb(pixel.rgb, limited_hue.angle - pixel.hue.angle))


class TransformerRestrictedHueValue(Transformer):
    LABEL = _("Restricted Hue and Value")

    def __init__(self, num_hues=6, num_value_levels=11):
        self.__hlc = HueLimitCriteria(num_hues)
        self.__vlc = ValueLimitCriteria(11)
        self.check_cached_values(BPS_TYPECODE[8])

    @property
    def num_hues(self):
        return self.__hlc.n_hues

    @num_hues.setter
    def num_hues(self, value):
        self.__hlc.set_n_hues(value)

    @property
    def num_value_levels(self):
        return self.__vlc.n_values

    @num_value_levels.setter
    def num_value_levels(self, value):
        self.__vlc.set_n_levels(value)

    def check_cached_values(self, typecode):
        if not hasattr(self, "BLACK") or self.BLACK.typecode != typecode:
            self.BLACK = rgb_math.proportions_to_array((0.0, 0.0, 0.0), typecode)
            self.WHITE = rgb_math.proportions_to_array((1.0, 1.0, 1.0), typecode)
            self.value_rgbs = self.__vlc.get_value_rgbs_for_typecode(typecode)

    def transformed_rgb(self, pixel):
        v_index = self.__vlc.get_value_index(pixel.rgb)
        if v_index == 0:
            return self.BLACK
        elif v_index == self.__vlc.n_values - 1:
            return self.WHITE
        h_index = self.__hlc.get_hue_index(pixel.hue)
        if h_index is None:
            return self.value_rgbs[v_index]
        limited_value = self.__vlc.c_values[v_index]
        limited_hue = self.__hlc.hues[h_index]
        if (len(pixel.rgb) - pixel.rgb.count(0)) == 2:
            return limited_hue.max_chroma_rgb_array_with_value(limited_value, pixel.rgb.typecode)
        else:
            tmp_rgb = array.array(pixel.rgb.typecode, rgb_math.rotate_rgb(pixel.rgb, limited_hue.angle - pixel.hue.angle))
            try:
                return array_scaled_to_value(tmp_rgb, limited_value)
            except OverflowError:
                return limited_hue.max_chroma_rgb_array_with_value(limited_value, pixel.rgb.typecode)


class TransformerHighChroma(Transformer):
    LABEL = _("High Chroma")

    @staticmethod
    def transformed_rgb(pixel):
        return pixel.hue.max_chroma_rgb_array_with_value(rgb_math.rgb_array_value(pixel.rgb), pixel.rgb.typecode)
