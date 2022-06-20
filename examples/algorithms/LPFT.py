import torch
from algorithms.ERM import ERM

stages = {'densenet121': [[],
                          ['features.norm5'      , 'features.denseblock4'],
                          ['features.transition3', 'features.denseblock3'],
                          ['features.transition2', 'features.denseblock2'],
                          ['features.transition1', 'features.denseblock1'],
                          ['features.norm0'      , 'features.conv0'      ]],
             'resnet50': [[],
                          ['layer4'],
                          ['layer3'],
                          ['layer2'],
                          ['layer1'],
                          ['bn1', 'conv1']]}

#settings = (0.0, [1,2,3,4,5], False), (1.0, [1,2,3,4,5], True)
settings = (0.0, [1,2,3,4,5], False), (0.5, [1], True), (0.625, [2], True), (0.75, [3], True), (0.875, [4], True), (1.0, [5], True)


class LPFT(ERM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.stage_module_names = stages[config.model]

    def _switch(self, stages, train):
        def _get_module(name):
            m = self.model
            for n in name.split('.'):
                m = getattr(m, n)
            return m

        def _set_BN(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train(train)
                #m.requires_grad_(train)

        for stage in stages:
            module_names = self.stage_module_names[stage]
            for name in module_names:
                m = _get_module(name)
                m.requires_grad_(train)
                m.apply(_set_BN)
        print([p.requires_grad for p in self.model.parameters()])

    def switch(self, epoch, batch_idx, epoch_len):
        base_idx = epoch * epoch_len
        idx = base_idx + batch_idx

        if batch_idx == 0:
            for t, stages, train in settings:
                if base_idx > int(t * epoch_len):
                    self._switch(stages, train)

        for t, stages, train in settings:
            if idx == int(t * epoch_len):
                self._switch(stages, train)
