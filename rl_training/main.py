import gym
import numpy as np
import f110_gym  # importante: importa el entorno registrado


def main():
    # Crear el entorno
    env = gym.make('f110_gym:f110-v0', num_agents=1,
                   map='../examples/example_map', map_ext='.png')

    # Posiciones iniciales: [x, y, theta]
    poses = np.array([
        [0.0, 0.0, 0.0]  # Agente 0
    ])

    # Resetear el entorno
    obs, reward, done, info = env.reset(poses)

    # Definir una acción fija: (steering angle, velocity)
    # 0.0 radianes de giro (recto) y velocidad 1.0 m/s
    action = np.array([
        [0.0, 1.0]
    ])

    # Loop de simulación
    step_count = 0
    done = False
    while not done:
        obs, reward, done, info = env.step(action)
        env.render()  # Mostrar simulación

        step_count += 1
        if step_count > 500:  # Seguridad: no correr eternamente
            print("Se alcanzó el límite de pasos.")
            break

    print("Simulación terminada.")
    env.close()


if __name__ == "__main__":
    main()
