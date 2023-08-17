import taichi as ti

ti.init(arch=ti.cuda)

N = 4000
dt = 1e-5

x = ti.Vector.field(3, dtype=ti.f32, shape=N)  # particle positions
v = ti.Vector.field(3, dtype=ti.f32, shape=N)  # particle velocities
g = ti.Vector.field(3, dtype=ti.f32, shape=N)

softening_length = 0.01

@ti.kernel
def gravity():
    for i, j in ti.ndrange(N, N):
        if j == 0:
            g[i] = 0
            
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        g[i] += - r / r.norm(softening_length)**3  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += g[i] * dt
    for i in x:
        x[i] += v[i] *dt


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]


def main():
    init()
    
    window = ti.ui.Window("3D Gravity Simulation", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # camera_pos = ti.math.vec4(2, 0, 0, 0)
    camera.position(2, 0, 0)
    camera.lookat(0, 0, 0)
    is_stop = False
    # gui = ti.GUI("Autodiff gravity")
    # while gui.running:
    #     for i in range(50):
    #         gravity()
    #         advance()
    #     gui.circles(x.to_numpy(), radius=1)
    #     gui.show()
        
    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, #yaw_speed=2.0, pitch_speed=2.0, 
                                 hold_key=ti.ui.LMB)
        
        # if window.is_pressed('z'):
        #     camera_pos = ti.math.rotation3d(0, 0, 9.0) @ camera_pos
        #     camera.position(camera_pos.x, camera_pos.y, camera_pos.z)
        
        for e in window.get_events(ti.ui.RELEASE):
           if e.key == ti.ui.SPACE:
                is_stop = not is_stop
            
        scene.set_camera(camera)
        scene.ambient_light((0.2, 0.2, 0.2))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        
        if not is_stop:
            for i in range(5):
                gravity()
                advance()
                
        scene.particles(x, color = (0.5, 0.5, 0.5), radius = 0.002)
        # Draw 3d-lines in the scene
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()