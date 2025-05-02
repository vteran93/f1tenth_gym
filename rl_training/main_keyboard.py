import gym
import numpy as np
import f110_gym
import pyglet
from pyglet.window import key

steering = 0.0
velocity = 0.0


def main():
    global steering, velocity

    # Crear entorno
    env = gym.make('f110_gym:f110-v0', num_agents=1,
                   map='../examples/example_map', map_ext='.png')
    poses = np.array([[0.0, 0.0, 0.0]])
    obs, reward, done, info = env.reset(poses)

    env.render()  # Renderizar para crear ventana

    # 🚨 Aquí corregimos: agarramos el objeto window real
    window = env.renderer
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    step_count = 0
    done = False

    while not done:
        # Leer teclado
        if keys[key.UP]:
            velocity += 0.05  # acelerar
        if keys[key.DOWN]:
            velocity -= 0.05  # frenar/reversa
        if keys[key.LEFT]:
            steering += 0.02  # girar izquierda
        if keys[key.RIGHT]:
            steering -= 0.02  # girar derecha
        if keys[key.SPACE]:
            velocity = 0.0
            steering = 0.0

        velocity = np.clip(velocity, -3.0, 5.0)
        steering = np.clip(steering, -0.5, 0.5)

        action = np.array([[steering, velocity]])
        obs, reward, done, info = env.step(action)

        env.render()

        step_count += 1
        if step_count > 5000:
            print("Se alcanzó el límite de pasos.")
            break

    print("Simulación terminada.")
    env.close()


if __name__ == "__main__":
    main()
