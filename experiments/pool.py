from snnbuilder.models import squeezenet
import os

maxpool_type = ['fir_max', 'avg_max', 'exp_max']
# spike_code = ['temporal_pattern', 'ttfs', 'ttfs_dyn_thresh', 'ttfs_corrective']
spike_code = ['temporal_pattern', 'ttfs']

# todo temporal_pattern overwrites batch_size and duration for config settings for unknown reason


conversion_kwargs = {'max2avg_pool': False}
m = squeezenet.SqueezeNet_CatsVsDogs(samples=100, epochs=50, conversion_kwargs=conversion_kwargs)
ann_acc = m.train()
parsed_acc = m.parse()

logfile = os.path.join(m.cache, 'exp_log.txt')
with open(logfile, 'w', encoding='utf-8') as f:
    f.write('ann acc {} \nparsed ann acc {}\n\n'.format(ann_acc, parsed_acc))

for e in maxpool_type:
    for y in spike_code:
        conversion_kwargs = {'max2avg_pool': False, 'maxpool_type': e, 'spike_code': y}

        m = squeezenet.SqueezeNet_CatsVsDogs(samples=100, epochs=50, conversion_kwargs=conversion_kwargs)
        snn_acc = m.sim()
        print('$' * 30)
        print('snn acc: {}, kwargs: {}'.format(snn_acc, conversion_kwargs))
        print('$' * 30)

        with open(logfile, 'a', encoding='utf-8') as f:
            f.write('snn acc {} \nconversion kwargs {}\n\n'.format(snn_acc, conversion_kwargs))





