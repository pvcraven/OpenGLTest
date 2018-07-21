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

INSTANCES = 1

DRAWING_ENGINE = NEW_CODE
MOVE_SPRITES = True

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700

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
    gl_Position = Projection * vec4(pos, 1.0);

    vec2 tex_offset = in_sub_tex_coords.xy;
    vec2 tex_size = in_sub_tex_coords.zw;

    v_texture = in_texture * tex_size + tex_offset; 

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
        (a, 0., 0., 0.),
        (0., b, 0., 0.),
        (0., 0., c, 0.),
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

        print(f"{self.center_x:5.1f}, {self.center_y:5.1f}")


class MySpriteList(arcade.SpriteList):
    def __init__(self):
        super().__init__(use_spatial_hash=False)
        self.program = None
        self.pos_angle_scale = None
        self.pos_angle_scale_buf = None
        self.sprite_list = []

    def append(self, item: T):
        """
        Add a new sprite to the list.
        """
        self.sprite_list.append(item)
        self.program = None
        item.register_sprite_list(self)

    def remove(self, item: T):
        """
        Remove a specific sprite from the list.
        """
        self.sprite_list.remove(item)
        self.program = None

    def calculate_sprite_buffer(self):
        if self.program is None:
            self.program = opengl_context.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        # Loop through each sprite and grab its position, and the texture it will be using.
        array_of_positions = []
        array_of_texture_names = []
        array_of_images = []
        array_of_sizes = []

        for sprite in self.sprite_list:
            array_of_positions.append([sprite.center_x, sprite.center_y, 0])
            if sprite.texture_name in array_of_texture_names:
                index = array_of_texture_names.index(sprite.texture_name)
                image = array_of_images[index]
            else:
                array_of_texture_names.append(sprite.texture_name)
                image = Image.open(sprite.texture_name)
                array_of_images.append(image)
            array_of_sizes.append(image.size)

        # Get their sizes
        widths, heights = zip(*(i.size for i in array_of_images))

        # Figure out what size a composate would be
        total_width = sum(widths)
        max_height = max(heights)

        # Make the composite image
        new_image = Image.new('RGBA', (total_width, max_height))

        x_offset = 0
        for image in array_of_images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]

        # Create a texture out the composite image
        texture = opengl_context.texture((new_image.width, new_image.height), 4, np.asarray(new_image))
        texture.use(0)

        # Create a list with the coordinates of all the unique textures
        tex_coords = []
        start_x = 0.0
        for image in array_of_images:
            end_x = start_x + (image.width / total_width)
            tex_coords.append([start_x, 0.0, image.width / total_width, 1.0])
            start_x = end_x

        # Go through each sprite and pull from the coordinate list, the proper
        # coordinates for that sprite's image.
        array_of_sub_tex_coords = []
        for sprite in self.sprite_list:
            index = array_of_texture_names.index(sprite.texture_name)
            array_of_sub_tex_coords.append(tex_coords[index])

        # Create numpy array with info on location and such
        np_array_positions = np.array(array_of_positions).astype('f4')

        # np_array_angles = (np.random.rand(INSTANCES, 1) * 2 * np.pi).astype('f4')
        np_array_angles = np.tile(np.array(0, dtype=np.float32), (INSTANCES, 1))
        np_array_sizes = np.array(array_of_sizes).astype('f4')
        np_sub_tex_coords = np.array(array_of_sub_tex_coords).astype('f4')
        self.pos_angle_scale = np.hstack((np_array_positions, np_array_angles, np_array_sizes, np_sub_tex_coords))
        self.pos_angle_scale_buf = opengl_context.buffer(self.pos_angle_scale.tobytes())

        vertices = np.array([
            #  x,    y,   u,   v
            -1.0, -1.0, 0.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32
        )
        vbo_buf = opengl_context.buffer(vertices.tobytes())

        vao_content = [
            (vbo_buf, '2f 2f', 'in_vert', 'in_texture'),
            (self.pos_angle_scale_buf, '3f 1f 2f 4f/i', 'in_pos', 'in_angle', 'in_scale', 'in_sub_tex_coords')
        ]

        self.vao = opengl_context.vertex_array(self.program, vao_content)

        self.update_positions()

    def update_positions(self):
        for i, sprite in enumerate(self.sprite_list):
            self.pos_angle_scale[i, 0] = sprite.center_x
            self.pos_angle_scale[i, 1] = sprite.center_y
            self.pos_angle_scale[i, 3] = math.radians(sprite.angle)
            self.pos_angle_scale[i, 4] = (sprite.width * 2) * sprite.scale
            self.pos_angle_scale[i, 5] = (sprite.height * 2) * sprite.scale

    def draw(self):

        if self.program is None:
            self.calculate_sprite_buffer()

        self.program['Texture'].value = 0
        self.program['Projection'].write(projection.tobytes())
        self.pos_angle_scale_buf.write(self.pos_angle_scale.tobytes())
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=INSTANCES)
        self.pos_angle_scale_buf.orphan()

    def update(self):
        if self.program is None:
            self.calculate_sprite_buffer()

        for i, sprite in enumerate(self.sprite_list):
            sprite.update()

        self.update_positions()


class MyWindow(arcade.Window):

    def __init__(self):
        super().__init__(height=SCREEN_HEIGHT, width=SCREEN_WIDTH, resizable=True)
        self.fps_counter = None
        self.vbo = None
        self.my_spite_list = None

    def setup(self):
        global opengl_context
        global projection

        if DRAWING_ENGINE == NEW_CODE:
            opengl_context = moderngl.create_context()
            opengl_context.viewport = (0, 0) + self.get_size()

        self.fps_counter = FPSCounter()

        if DRAWING_ENGINE == OLD_CODE:
            self.my_spite_list = arcade.SpriteList()
        else:
            self.my_spite_list = MySpriteList()

        sprite_names = ['gold_1.png', 'gold_2.png', 'gold_3.png', 'gold_4.png', 'character.png']
        sprite_names = ['character.png']
        for i in range(INSTANCES):
            texture_name = random.choice(sprite_names)
            my_sprite = MySprite(texture_name, 0.25)

            my_sprite.center_x = random.randrange(SCREEN_WIDTH)
            my_sprite.center_y = random.randrange(SCREEN_HEIGHT)
            my_sprite.angle = random.randrange(360)
            my_sprite.angle = 0
            my_sprite.texture_name = texture_name

            my_sprite.change_x = random.random() * 5 - 2.5
            my_sprite.change_y = random.random() * 5 - 2.5
            my_sprite.change_angle = random.random() * 5 - 2.5
            my_sprite.change_angle = 0
            my_sprite.scale = 0.5

            self.my_spite_list.append(my_sprite)

        self.my_spite_list.update()

        if DRAWING_ENGINE == NEW_CODE:
            projection = create_orthogonal_projection(left=0, right=SCREEN_WIDTH, bottom=0, top=SCREEN_HEIGHT,
                                                      near=-1000, far=100, dtype=np.float32)

            opengl_context.enable(moderngl.BLEND)
            opengl_context.enable(moderngl.DEPTH_TEST)

        pyglet.clock.schedule_interval(self.show_fps, 1)

    def update(self, dt):
        if MOVE_SPRITES:
            self.my_spite_list.update()

    def show_fps(self, dt):
        print(f"FPS: {self.fps_counter.get_fps()}")

    def on_draw(self):
        if DRAWING_ENGINE == OLD_CODE:
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
