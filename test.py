#!/usr/bin/env python3
# -*- coding:  utf-8 -*-

from typing import TypeVar

import time
import collections
import math
import random
from PIL import Image

import pyglet
import arcade

import moderngl
import numpy as np

NEW_CODE = 0
OLD_CODE = 1

drawing_engine = NEW_CODE

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700

INSTANCES = 2

opengl_context = None
projection = None

"""
4x4 matrix called projection
projection is the same for each vertex because of 'uniform'
in_vert = first x, y numbers out of vert array
in_texture = u, v stuff
Then, we use the other array on a PER INSTANCE
Position, angle, scale 
"""

VERTEX_SHADER = """
#version 330
uniform mat4 Projection;
in vec2 in_vert;
in vec2 in_texture;
in vec3 in_pos;
in float in_angle;
in vec2 in_scale;
in vec4 in_sub_tex_coords;
out vec2 v_texture;
void main() {
    mat2 rotate = mat2(
                cos(in_angle), sin(in_angle),
                -sin(in_angle), cos(in_angle)
            );
    vec3 pos;
    pos = in_pos + vec3(rotate * (in_vert * in_scale), 0.);
    pos[0] += in_sub_tex_coords[0];
    gl_Position = Projection * vec4(pos, 1.0);
   
    v_texture = in_texture;
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D Texture;
in vec2 v_texture;
out vec4 f_color;
void main() {
    vec4 basecolor = texture(Texture, v_texture);
    if (basecolor.a == 0.0){
        discard;
    }
    f_color = basecolor;
}
"""


def create_orthogonal_projection(
    left,
    right,
    bottom,
    top,
    near,
    far,
    dtype=None
):
    """Creates an orthogonal projection matrix.
    :param float left: The left of the near plane relative to the plane's centre.
    :param float right: The right of the near plane relative to the plane's centre.
    :param float top: The top of the near plane relative to the plane's centre.
    :param float bottom: The bottom of the near plane relative to the plane's centre.
    :param float near: The distance of the near plane from the camera's origin.
        It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
        at close range.
    :param float far: The distance of the far plane from the camera's origin.
    :param dtype:
    :rtype: numpy.array
    :return: A projection matrix representing the specified orthogonal perspective.
    .. seealso:: http://msdn.microsoft.com/en-us/library/dd373965(v=vs.85).aspx
    """

    """
    A 0 0 Tx
    0 B 0 Ty
    0 0 C Tz
    0 0 0 1
    A = 2 / (right - left)
    B = 2 / (top - bottom)
    C = -2 / (far - near)
    Tx = (right + left) / (right - left)
    Ty = (top + bottom) / (top - bottom)
    Tz = (far + near) / (far - near)
    """
    rml = right - left
    tmb = top - bottom
    fmn = far - near

    a = 2. / rml
    b = 2. / tmb
    c = -2. / fmn
    tx = -(right + left) / rml
    ty = -(top + bottom) / tmb
    tz = -(far + near) / fmn

    return np.array((
        ( a, 0., 0., 0.),
        (0.,  b, 0., 0.),
        (0., 0.,  c, 0.),
        (tx, ty, tz, 1.),
    ), dtype=dtype)


class FPSCounter:
    def __init__(self):
        self.time = time.perf_counter()
        self.frame_times = collections.deque(maxlen=60)

    def tick(self):
        t1 = time.perf_counter()
        dt = t1 - self.time
        self.time = t1
        self.frame_times.append(dt)

    def get_fps(self):
        all_frame_times = sum(self.frame_times)
        if all_frame_times == 0:
            return 0
        else:
            return len(self.frame_times) / sum(self.frame_times)

T = TypeVar('T', bound=arcade.Sprite)

class MySprite(arcade.Sprite):
    def __init__(self, image, scale=1):
        super().__init__(image, scale)
        # self.width = 32
        # self.height = 32

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y
        self.angle += self.change_angle

        if self.center_x > SCREEN_WIDTH and self.change_x > 0:
            self.change_x *= -1

        if self.center_x < 0 and self.change_x < 0:
            self.change_x *= -1

        if self.center_y > SCREEN_HEIGHT and self.change_y > 0:
            self.change_y *= -1

        if self.center_y < 0 and self.change_y < 0:
            self.change_y *= -1

class MySpriteList(arcade.SpriteList):
    def __init__(self):
        super().__init__(use_spatial_hash=False)
        self.prog = None
        self.pos_angle_scale = None
        self.pos_angle_scale_buf = None
        self.sprite_list = []

    def append(self, item: T):
        """
        Add a new sprite to the list.
        """
        self.sprite_list.append(item)
        self.prog = None
        item.register_sprite_list(self)

    def remove(self, item: T):
        """
        Remove a specific sprite from the list.
        """
        self.sprite_list.remove(item)
        self.prog = None

    def calculate_sprite_buffer(self):
        self.prog = opengl_context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        img = Image.open("gold_1.png")

        images = list(map(Image.open, ['gold_1.png', 'gold_2.png']))
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_image = Image.new('RGBA', (total_width, max_height))

        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]


        new_image.save("out.png")

        # texture = opengl_context.texture((img.width, img.height), 4, np.asarray(img))
        texture = opengl_context.texture((new_image.width, new_image.height), 4, np.asarray(new_image))

        # img = pyglet.image.load('gold_1.png', file=pyglet.resource.file('gold_1.png'))
        # texture = opengl_context.texture((img.width, img.height), 4, img.get_data("RGBA", img.pitch))
        texture.use(0)

        array_of_positions = []
        for sprite in self.sprite_list:
            array_of_positions.append([sprite.center_x, sprite.center_y, 0])

        np_array_positions = np.array(array_of_positions).astype('f4')

        np_array_angles = (np.random.rand(INSTANCES, 1) * 2 * np.pi).astype('f4')
        np_array_sizes = np.tile(np.array([img.width / 2, img.height / 2], dtype=np.float32), (INSTANCES, 1))
        np_sub_tex_coords = np.tile(np.array([0.0, 0.0, 42.0, 42.0], dtype=np.float32), (INSTANCES, 1))

        self.pos_angle_scale = np.hstack((np_array_positions, np_array_angles, np_array_sizes, np_sub_tex_coords))
        # self.pos_angle_scale = np.hstack((np_array_positions, np_array_angles, np_array_sizes))

        self.pos_angle_scale_buf = opengl_context.buffer(self.pos_angle_scale.tobytes())

        vertices = np.array([
            #  x,    y,   u,   v
            -1.0, -1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32
        )
        self.vbo_buf = opengl_context.buffer(vertices.tobytes())

        vao_content = [
            (self.vbo_buf, '2f 2f', 'in_vert', 'in_texture'),
            (self.pos_angle_scale_buf, '3f 1f 2f 4f/i', 'in_pos', 'in_angle', 'in_scale', 'in_sub_tex_coords')
        ]
        print("XXX", len(self.pos_angle_scale.tobytes()))

        members = self.prog._members
        content = tuple((a.mglo, b) + tuple(getattr(members.get(x), 'mglo', None) for x in c) for a, b, *c in vao_content)

        self.vao = opengl_context.vertex_array(self.prog, vao_content)

    def draw(self):

        if self.prog is None:
            self.calculate_sprite_buffer()

        self.prog['Texture'].value = 0
        self.prog['Projection'].write(projection.tobytes())
        self.pos_angle_scale_buf.write(self.pos_angle_scale.tobytes())
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=INSTANCES)
        self.pos_angle_scale_buf.orphan()

    def update(self):
        if self.prog is None:
            self.calculate_sprite_buffer()

        for i, sprite in enumerate(self.sprite_list):
            sprite.update()

            self.pos_angle_scale[i, 0] = sprite.center_x
            self.pos_angle_scale[i, 1] = sprite.center_y
            self.pos_angle_scale[i, 3] = math.radians(sprite.angle)
            self.pos_angle_scale[i, 4] = (sprite.width * 2) * sprite.scale
            self.pos_angle_scale[i, 5] = (sprite.height * 2) * sprite.scale


class MyWindow(arcade.Window):

    def __init__(self):
        super().__init__(height=SCREEN_HEIGHT, width=SCREEN_WIDTH, resizable=True)
        self.fps_counter = None
        self.vbo = None
        self.my_spite_list = None

    def setup(self):
        global opengl_context
        global projection

        if drawing_engine == NEW_CODE:
            opengl_context = moderngl.create_context()
            opengl_context.viewport = (0, 0) + self.get_size()

        self.fps_counter = FPSCounter()

        if drawing_engine == OLD_CODE:
            self.my_spite_list = arcade.SpriteList()
        else:
            self.my_spite_list = MySpriteList()

        for i in range(INSTANCES):

            my_sprite = MySprite('gold_1.png', 0.25)

            my_sprite.center_x = random.randrange(SCREEN_WIDTH)
            my_sprite.center_y = random.randrange(SCREEN_HEIGHT)
            my_sprite.angle = random.randrange(360)

            my_sprite.change_x = random.random() * 5 - 2.5
            my_sprite.change_y = random.random() * 5 - 2.5
            my_sprite.change_angle = random.random() * 5 - 2.5

            self.my_spite_list.append(my_sprite)

        if drawing_engine == NEW_CODE:
            projection = create_orthogonal_projection(left=0, right=SCREEN_WIDTH, bottom=0, top=SCREEN_HEIGHT, near=-1000, far=100, dtype=np.float32)

            opengl_context.enable(moderngl.BLEND)
            opengl_context.enable(moderngl.DEPTH_TEST)

        pyglet.clock.schedule_interval(self.show_fps, 1)

    def update(self, dt):
        self.my_spite_list.update()

    def show_fps(self, dt):
        print(f"FPS: {self.fps_counter.get_fps()}")

    def on_draw(self):
        if drawing_engine == OLD_CODE:
            arcade.start_render()
        else:
            opengl_context.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        self.my_spite_list.draw()

        self.fps_counter.tick()


def main():
    """ Main method """
    window = MyWindow()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
