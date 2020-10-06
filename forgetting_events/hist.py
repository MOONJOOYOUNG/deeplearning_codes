import utils
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


forgetting_history = torch.load('./train_forgetting.pth')

forgettable, filename, forget_first_learn= forgetting_history.get_forgettable_examples(sorted=False)
unforgettable, filename, unforget_first_learn = forgetting_history.get_unforgettable_examples(sorted=False)


plt.hist(forgettable, bins=15, rwidth=0.90,color='blue', density=True)
plt.xlabel('forgetting events')
plt.ylabel('examples')
plt.savefig('./forgetting_events.jpg')
plt.close()

plt.hist(forget_first_learn, bins=10,color='yellow', density=True)
plt.hist(unforget_first_learn, bins=10,color='blue', density=True)
plt.xlabel('forgetting events')
plt.ylabel('examples')
plt.savefig('./first_learn.jpg')
plt.close()