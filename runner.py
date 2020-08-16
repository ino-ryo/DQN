def runner(agent,n_episode):
    for epsode in range(n_episode)
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # ε-greedyで行動を選択
        action = net.act(obs.float().to(device), epsilon_func(step))
        # 環境中で実際に行動
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        # リプレイバッファに経験を蓄積
        replay_buffer.push([obs, action, reward, next_obs, done])
        obs = next_obs