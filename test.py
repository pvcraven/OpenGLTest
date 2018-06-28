#!/usr/bin/env python3
# -*- coding:  utf-8 -*-

import time
import collections

import pyglet

import moderngl
import numpy as np


SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700

INSTANCES = 10_000


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
        return len(self.frame_times) / sum(self.frame_times)


class MyWindow(pyglet.window.Window):

    def __init__(self):
        super().__init__(height=SCREEN_HEIGHT, width=SCREEN_WIDTH)
        self.ctx = None
        self.fps_counter = None
        self.prog = None
        self.proj = None
        self.pos_scale = None

    def setup(self):

        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0) + self.get_size()
        self.fps_counter = FPSCounter()

        """
        4x4 matrix called projection
        projection is the same for each vertex because of 'uniform'
        in_vert = first x, y numbers out of vert array
        in_texture = u, v stuff
        Then, we use the other array on a PER INSTANCE
        Position, angle, scale 
        """
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Projection;
                in vec2 in_vert;
                in vec2 in_texture;
                in vec3 in_pos;
                in float in_angle;
                in vec2 in_scale;
                out vec2 v_texture;
                void main() {
                    mat2 rotate = mat2(
                                cos(in_angle), sin(in_angle),
                                -sin(in_angle), cos(in_angle)
                            );
                    vec3 pos;
                    pos = in_pos + vec3(rotate * (in_vert * in_scale), 0.);
                    gl_Position = Projection * vec4(pos, 1.0);
                    v_texture = in_texture;
                }
            ''',
            fragment_shader='''
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
            ''',
        )
        vertices = np.array([
            #  x,    y,   u,   v
            -1.0, -1.0, 0.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            ], dtype=np.float32
        )

        self.proj = create_orthogonal_projection(
            left=0, right=600, bottom=0, top=400, near=-1000, far=100, dtype=np.float32
        )

        img = pyglet.image.load('grossinis2.png', file=pyglet.resource.file('grossinis2.png'))
        texture = self.ctx.texture((img.width, img.height), 4, img.get_data("RGBA", img.pitch))
        texture.use(0)

        positions = (np.random.rand(INSTANCES, 3) * 1000).astype('f4')
        angles = (np.random.rand(INSTANCES, 1) * 2 * np.pi).astype('f4')
        sizes = np.tile(np.array([img.width / 2, img.height / 2], dtype=np.float32), (INSTANCES, 1))
        self.pos_scale = np.hstack((positions, angles, sizes))

        self.pos_scale_buf = self.ctx.buffer(self.pos_scale.tobytes())

        self.vbo = self.ctx.buffer(vertices.tobytes())
        vao_content = [
            (self.vbo, '2f 2f', 'in_vert', 'in_texture'),
            (self.pos_scale_buf, '3f 1f 2f/i', 'in_pos', 'in_angle', 'in_scale')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)

        pyglet.clock.schedule_interval(self.show_fps, 1)
        pyglet.clock.schedule_interval(self.update, 1 / 60)

    def update(self, dt):
        self.pos_scale[1:, 0] += 0.1
        self.pos_scale[1:, 3] += 0.01
        self.pos_scale[1::2, 2] += 0.1

    def show_fps(self, dt):
        print(f"FPS: {self.fps_counter.get_fps()}")


    def on_resize(self, width, height):
        self.ctx.viewport = (0, 0, width, height)
        self.proj = create_orthogonal_projection(
            left=0, right=width, bottom=0, top=height, near=-1000, far=100, dtype=np.float32
        )
        return True


    def on_draw(self):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        self.prog['Texture'].value = 0
        self.prog['Projection'].write(self.proj.tobytes())
        self.pos_scale_buf.write(self.pos_scale.tobytes())
        self.vao.render(moderngl.TRIANGLE_STRIP, instances=INSTANCES)
        self.pos_scale_buf.orphan()
        self.fps_counter.tick()


def main():
    """ Main method """
    window = MyWindow()
    window.setup()
    pyglet.app.run()


if __name__ == "__main__":
    main()