from snnbuilder.models.catsvsdogs import CNN_CatsVsDogs
import os
import itertools

# run with `python -m experiments.param_tune`

params = {'width_shift_range': [(-.2, .2), 0],
    'height_shift_range': [(-.2, .2), 0],
    'horizontal_flip': [True, False],
    # 'vertical_flip': [True, False],
    'rotation_range': [30, 15, 0],
    'fill_mode': ['wrap', 'nearest'],  # constant, nearest, reflect, wrap
}
keys = list(params.keys())
output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../dev_outputs'))
results = []

for values in itertools.product(*params.values()):
    d = {keys[i]: values[i] for i in range(len(values))}
    x = CNN_CatsVsDogs(output_path, training_augs=values, epochs=25)
    acc = x.train()
    print(acc, values)
    results.append((acc, values))

print('\n\n results')
for e in results:
    print('{}\t {}'.format(e[0], e[1]))


# l = ((-0.2, 0.2), 0, True, True, 30, 'wrap')
# keys = list(params.keys())
# d = {keys[i]: l[i] for i in range(len(l))}
