import matplotlib.pyplot as plt

self_embed_dormant = [15.23, 14.45, 14.45, 21.09, 30.86, 26.95, 32.03, 31.64]
neighbor_embed_dormant = [37.50, 40.62, 58.98, 63.28, 64.45, 64.45, 65.62, 67.19]
obst_embed_dormant = [82.42, 71.09, 72.66, 73.44, 74.22, 73.83, 80.08, 80.08]

train_steps = [50000000, 100000000, 200000000, 300000000, 400000000, 500000000, 600000000, 1000000000]

plt.figure()
plt.plot(train_steps, self_embed_dormant, label='Self embed')
plt.plot(train_steps, neighbor_embed_dormant, label='Neighbor embed')
plt.plot(train_steps, obst_embed_dormant, label='Obst embed')
plt.xlabel('Training Steps')
plt.ylabel('Dormant Neurons (%)')
plt.legend()

plt.show()