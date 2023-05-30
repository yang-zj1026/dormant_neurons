import copy

import torch
import torch.nn as nn
from weight_recycler import NeuronRecycler


def neuron_recycler_test(model, all_layer_names, reset_layers, reset_layers_idx, next_layers, optimizer):
    weight_recycler = NeuronRecycler(all_layer_names, reset_layers, reset_layers_idx, next_layers, reset_period=100)
    loss_func = nn.CrossEntropyLoss()
    for training_step in range(0, 200):
        optimizer.zero_grad()

        test_input = torch.randn(8, 10)
        test_output = model(test_input)
        test_label = torch.randn(8, 4)

        loss = loss_func(test_output, test_label)
        loss.backward()
        optimizer.step()

        intermediates = model.intermediate_results
        log_dict_neurons = (weight_recycler.maybe_log_deadneurons(0, intermediates))
        old_online_params = model.state_dict()

        opt_state = optimizer.state_dict()
        old_param_state = copy.deepcopy(opt_state['state'])
        online_params, opt_state = weight_recycler.maybe_update_weights(training_step, intermediates, old_online_params, opt_state)
        param_state = opt_state['state']

        if training_step % 100 == 0 and training_step > 0:
            for key in param_state.keys():
                print('-----------------------\n', key)
                value1, value2 = param_state[key], old_param_state[key]
                print(torch.allclose(value1['exp_avg'], value2['exp_avg']))
                print(torch.allclose(value1['exp_avg_sq'], value2['exp_avg_sq']))

        model.load_state_dict(online_params)
        optimizer.load_state_dict(opt_state)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.self_embed_layer = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
        )
        self.neighbor_embed_layer = nn.Sequential(
            nn.Linear(10, 20, bias=True),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
        )
        self.obstacle_embed_layer = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
        )

        self.feed_forward = self.feed_forward = nn.Sequential(
            nn.Linear(90, 4),
            nn.ReLU(),
        )

        self.intermediate_results = {}

    def forward(self, x):
        self_embed = self.forward_with_intermediates(x, self.self_embed_layer, 'self_embed_layer')
        neighbor_embed = self.forward_with_intermediates(x, self.neighbor_embed_layer,
                                                         'neighbor_embed_layer')
        obstacle_embed = self.forward_with_intermediates(x, self.obstacle_embed_layer,
                                                         'obstacle_embed_layer')
        embeds = torch.cat([self_embed, neighbor_embed, obstacle_embed], dim=1)
        out = self.feed_forward(embeds)
        return out

    def forward_with_intermediates(self, x, module, module_name):
        key = ''
        for submodule in module.named_children():
            layer_name, layer = submodule
            if not isinstance(layer, nn.ReLU):
                key = module_name + '.' + layer_name
            x = layer(x)
            self.intermediate_results[key] = x

        return x


if __name__ == "__main__":
    model = MyModel()

    param_index_dict = {}
    params = []
    for i, layer in enumerate(model.named_parameters()):
        layer_name, layer_param = layer
        param_index_dict[layer_name] = i
        params.append(layer_param)

    optimizer = torch.optim.Adam(iter(params), lr=0.01)

    # Get all layer names (Without activation layer)
    layer_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ReLU):
            prev_name = name
            if name != '':
                layer_names.append(name)

    next_layers = {}
    reset_layers = []
    for current_layer, next_layer in zip(layer_names[:-1], layer_names[1:]):
        if '.' not in current_layer:
            continue
        curr_top = current_layer.split('.')[0]
        next_top = next_layer.split('.')[0]
        if curr_top == next_top:
            next_layers[current_layer] = next_layer
            reset_layers.append(current_layer)
            reset_layers.append(next_layer)

    neuron_recycler_test(model, layer_names, reset_layers, param_index_dict, next_layers, optimizer)
