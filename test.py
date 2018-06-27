#!/usr/bin/env python3
# -*- coding:  utf-8 -*-

import time
import collections

import pyglet

import moderngl
import numpy as np

# import cProfile, pstats

# pr = cProfile.Profile()
# pr.enable()


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

    A = 2. / rml
    B = 2. / tmb
    C = -2. / fmn
    Tx = -(right + left) / rml
    Ty = -(top + bottom) / tmb
    Tz = -(far + near) / fmn

    return np.array((
        ( A, 0., 0., 0.),
        (0.,  B, 0., 0.),
        (0., 0.,  C, 0.),
        (Tx, Ty, Tz, 1.),
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


fps_counter = FPSCounter()
# config = pyglet.gl.Config(double_buffer=True, depth_size=16)
window = pyglet.window.Window(
    width=1366, height=768, resizable=True, vsync=True)

ctx = moderngl.create_context()
ctx.viewport = (0, 0) + window.get_size()
"""
4x4 matrix called projection
projection is the same for each vertex because of 'uniform'
in_vert = first x, y numbers out of vert array
in_texture = u, v stuff
Then, we use the other array on a PER INSTANCE
Position, angle, scale 
"""
prog = ctx.program(
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

proj = create_orthogonal_projection(
    left=0, right=600, bottom=0, top=400, near=-1000, far=100, dtype=np.float32
)

img = pyglet.image.load('grossinis2.png', file=pyglet.resource.file('grossinis2.png'))
texture = ctx.texture((img.width, img.height), 4, img.get_data("RGBA", img.pitch))
texture.use(0)

INSTANCES = 10_000

# pos_scale = np.array([
#       # pos_x, pos_y,  z, angle,   scale_x,       scale_y
#         100.0, 150.0, 0., 0., rect.width/2, rect.height/2,
#         120.5, 200.0, 10., 0., rect.width/2, rect.height/2,
#     ], dtype=np.float32)

positions = (np.random.rand(INSTANCES, 3) * 1000).astype('f4')
angles = (np.random.rand(INSTANCES, 1) * 2 * np.pi).astype('f4')
sizes = np.tile(np.array([img.width / 8, img.height / 8], dtype=np.float32),
                (INSTANCES, 1)
                )
pos_scale = np.hstack((positions, angles, sizes))
player_pos = pos_scale[0, :]

pos_scale_buf = ctx.buffer(pos_scale.tobytes())


vbo = ctx.buffer(vertices.tobytes())
vao_content = [
    (vbo, '2f 2f', 'in_vert', 'in_texture'),
    (pos_scale_buf, '3f 1f 2f/i', 'in_pos', 'in_angle', 'in_scale')
]
vao = ctx.vertex_array(prog, vao_content)
ctx.enable(moderngl.BLEND)
ctx.enable(moderngl.DEPTH_TEST)


def show_fps(dt):
    print(f"FPS: {fps_counter.get_fps()}")


def update(dt):
    pos_scale[1:, 0] += 0.1
    pos_scale[1:, 3] += 0.01
    pos_scale[1::2, 2] += 0.1


# pyglet.clock.schedule_once(report, 5)

pyglet.clock.schedule_interval(show_fps, 1)
pyglet.clock.schedule_interval(update, 1 / 60)


@window.event
def on_resize(width, height):
    global proj
    ctx.viewport = (0, 0, width, height)
    proj = create_orthogonal_projection(
        left=0, right=width, bottom=0, top=height, near=-1000, far=100, dtype=np.float32
    )
    return True


@window.event
def on_draw():
    ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

    prog['Texture'].value = 0
    prog['Projection'].write(proj.tobytes())
    pos_scale_buf.write(pos_scale.tobytes())
    vao.render(moderngl.TRIANGLE_STRIP, instances=INSTANCES)
    pos_scale_buf.orphan()
    fps_counter.tick()


pyglet.app.run()
