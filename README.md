# CartPole2023
強化学習で作成したプログラム 　作者　市川

## 目次
* [CartPoleとは？](#CartPoleとは？)
* [Versionについて](#Versionについて)
* [プログラム一覧](#プログラム一覧)
* [インストール方法](#インストール方法)
* [実行結果](#実行結果)
* [参照サイト](#参照サイト)

<!-- some long code -->
## CartPoleとは？
Cartpoleとは[OpenAI Gym](https://github.com/openai/gym)が提供するゲーム環境で倒立振子を行うゲームである.振り子は吊り下げられた状態が安定であるが、倒立振子とは棒を台車に乗せて立たせているため、不安定であることから本質的に倒立状態を保つために能動的に制御する必要がある.この制御をCartpoleプログラムによって学習させ安定した制御を行わせることを目的とする.

![Screenshot from 2023-12-21 16-18-27](https://github.com/b213r002/CartPole2023/assets/153800075/2fe078dc-c72a-4c22-a68f-b13b500f4d63)
## Versionについて
Package         Version
--------------- ------------
* cloudpickle     3.0.0
* gym             0.25.1
* gym-notices     0.0.8
* numpy           1.21.5
* pandas          2.1.2
* pip             22.0.2
* pygame          2.5.1
* python-dateutil 2.8.2
* pytz            2023.3.post1
* setuptools      59.6.0
* six             1.16.0
* tzdata          2023.3


## プログラム一覧

プログラムは以下のように作成できた.

```bash
import gymnasium as gym
import gym
import numpy as np
import matplotlib.pyplot as plt

#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1')

q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

def digitize_state(observation):
    # 各値を4個の離散値に変換
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    # 0~255に変換
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])

goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 1000
#um_episodes = 10
last_time_steps = np.zeros(num_consecutive_iterations)
step_list = []
frames = []

def get_action(state, action, observation, reward):
    next_state = digitize_state(observation)
    next_action = np.argmax(q_table[next_state])

    # Qテーブルの更新
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * q_table[next_state, next_action])
   

    return next_action, next_state

for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()

    state = digitize_state(observation)
    action = np.argmax(q_table[state])

    episode_reward = 0
    for t in range(max_number_of_steps):
        # CartPoleの描画
        #env.render()

        # 行動の実行とフィードバックの取得
        observation, reward, done, info = env.step(action)

        if done:
            if t < 195:
                reward = -200  # 倒れたら罰則
            else:
                reward = 1  # 立ったまま終了時は罰則はなし
        else:
            reward = 1  # 各ステップで立ってたら報酬追加

        episode_reward += reward  # 報酬を追加

        
         # 行動の選択
        action, state = get_action(state, action, observation, reward)
        

        if done or t >= 199:
            print('%d Episode finished after %d time steps / mean %f' % (episode, t + 1,
                last_time_steps.mean()))
            step_list.append(t+1)
            
            last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
            
            break
    

    #if (last_time_steps.mean() >= goal_average_steps): # 直近の100エピソードが195以上であれば成功
        #print('Episode %d train agent successfuly!' % episode)
        #break

print(f'len(step_list) = {len(step_list)}')
es = np.arange(0, len(step_list))
plt.plot(es, step_list)
plt.savefig("cartpole.png")
plt.figure()

```
## インストール方法
これらをインストールしてください
```bash
$ pip install matplotlib
$ python3 -m pip install pygame==2.5.1
$ pip install gym==0.25.1
```
## 実行結果

Episodeが1000回に達するまで実行するようにした結果、おおよそ500回までには200stepまで安定するようになった.

```bash
0 Episode finished after 140 time steps / mean 0.000000
1 Episode finished after 127 time steps / mean -0.610000
2 Episode finished after 112 time steps / mean -1.350000
3 Episode finished after 142 time steps / mean -2.240000
4 Episode finished after 24 time steps / mean -2.830000
5 Episode finished after 11 time steps / mean -4.600000
6 Episode finished after 9 time steps / mean -6.500000
....
990 Episode finished after 200 time steps / mean 200.000000
991 Episode finished after 200 time steps / mean 200.000000
992 Episode finished after 200 time steps / mean 200.000000
993 Episode finished after 200 time steps / mean 200.000000
994 Episode finished after 200 time steps / mean 200.000000
995 Episode finished after 200 time steps / mean 200.000000
996 Episode finished after 200 time steps / mean 200.000000
997 Episode finished after 200 time steps / mean 200.000000
998 Episode finished after 200 time steps / mean 200.000000
999 Episode finished after 200 time steps / mean 200.000000
len(step_list) = 1000
```
次の図は学習内容を表したものである.
![cartpole](https://github.com/b213r002/CartPole2023/assets/153800075/ae550bfc-8e33-4811-bbaf-fd907768bad2)

## 参照サイト
https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df

https://qiita.com/KokiSakano/items/c8b92640b36b2ef21dbf

