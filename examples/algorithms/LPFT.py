import torch
from algorithms.ERM import ERM
from models.classifier import LogisticRegression, _fit

stages = {'densenet121': [['classifier'],
                          ['features.norm5'      , 'features.denseblock4'],
                          ['features.transition3', 'features.denseblock3'],
                          ['features.transition2', 'features.denseblock2'],
                          ['features.transition1', 'features.denseblock1'],
                          ['features.norm0'      , 'features.conv0'      ]],
             'resnet50': [['fc'],
                          ['layer4'],
                          ['layer3'],
                          ['layer2'],
                          ['layer1'],
                          ['bn1', 'conv1']]}

#settings = () # FT
#settings = (0.0, [1,2,3,4,5], False), # LP
#settings = (0.0, [1,2,3,4,5], False), (2.0, [1,2,3,4,5], True) # LPFT
#settings = (0.0, [1,2,3,4,5], False), (0.5, [1], True), (0.625, [2], True), (0.75, [3], True), (0.875, [4], True), (1.0, [5], True) # RHT
#settings = (0.0, [1,2,3,4,5], False), (2.0, [1], True), (2.5, [2], True), (3.0, [3], True), (3.5, [4], True), (4.0, [5], True) # RHT
#settings = (0.0, [0,1,2,3,4,5], False), (1.0, [0], False)
#settings = (0.0, [0,1,2,3,4,5], False), (1.0, [0,1,2,3,4,5], True)
settings = (0.0, [2,3,4,5], False), (0.2, [2], True), (0.4, [3], True), (0.6, [4], True), (0.8, [5], True) # RHT


class LPFT(ERM):
    def __init__(self, config, d_out, **kwargs):
        super().__init__(config, d_out, **kwargs)
        self.stage_module_names = stages[config.model]
        self.backprop = True

        self.clf = LogisticRegression(config.seed, d_out)
        self.fit_clf = False
        def _get_feature(_, input):
            if self.clf.train:
                self._feature = input[0].detach().clone()
        self.lin = self._get_module(self.stage_module_names[0][0])
        self.lin.register_forward_pre_hook(_get_feature)
        ##
        W, b = _fit(config.dataset)
        self.lin.weight.data.copy_(W)
        self.lin.bias  .data.copy_(b)

    def _get_module(self, name):
        m = self.model
        for n in name.split('.'):
            m = getattr(m, n)
        return m

    def _switch(self, stages, train):
        def _set_BN(m):
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train(train)

        for stage in stages:
            module_names = self.stage_module_names[stage]
            for name in module_names:
                m = self._get_module(name)
                m.requires_grad_(train)
                m.apply(_set_BN)

        grad = [p.requires_grad for p in self.model.parameters()]
        self.backprop = any(grad)
        print(grad)

    def switch(self, epoch, batch_idx, epoch_len):
        base_idx = epoch * epoch_len
        idx = base_idx + batch_idx

        if batch_idx == 0:
            # restore settings
            for t, stages, train in settings:
                if base_idx > int(t * epoch_len):
                    self._switch(stages, train)

        for t, stages, train in settings:
            if idx == int(t * epoch_len):
                self._switch(stages, train)
                if (0 in stages) and not train:
                    self.clf.use = True
            elif idx == int(t * epoch_len)-1:
                if self.clf.train and (0 in stages): ## and train:
                    self.fit_clf = True

    def _update(self, results, should_step=False):
        if self.backprop:
            super()._update(results, should_step=should_step)
        else:
            results['objective'] = self.objective(results).item()

    def update(self, batch, **kwargs):
        results = super().update(batch, **kwargs)
        if self.clf.train:
            self.clf.update(X=self._feature, y=batch[1].detach().clone())
            if self.fit_clf:
                self.clf.fit()
                self.fit_clf = False
                W, b = self.clf.get_Wb()
                self.lin.weight.data.copy_(W)
                self.lin.bias  .data.copy_(b)
        return results
