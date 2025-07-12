import torch
import torch.optim as optim

from models.actor import ActorLSTM
from models.critic import CriticLSTM

def initialize_actor_critic(config, num_actions=2):
    input_dim = num_actions + 1  # one-hot action + reward or RPE
    
    actor = ActorLSTM(input_size=input_dim, hidden_size=config.hidden_size, num_actions=num_actions).to(config.device)
    critic = CriticLSTM(input_size=input_dim, hidden_size=config.hidden_size).to(config.device)

    actor_optim = optim.RMSprop(actor.parameters(), lr=config.actor_lr)
    critic_optim = optim.RMSprop(critic.parameters(), lr=config.critic_lr)

    scheduler_a = torch.optim.lr_scheduler.StepLR(actor_optim, step_size=20000, gamma=0.9)
    scheduler_c = torch.optim.lr_scheduler.StepLR(critic_optim, step_size=20000, gamma=0.9)

    return actor, critic, actor_optim, critic_optim, scheduler_a, scheduler_c
