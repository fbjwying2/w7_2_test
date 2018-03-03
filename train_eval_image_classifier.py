from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

train_cmd = 'python ./train_image_classifier.py  --dataset_name={dataset_name} --dataset_dir={dataset_dir} --checkpoint_path={checkpoint_path} --model_name={model_name} --checkpoint_exclude_scopes={checkpoint_exclude_scopes} --train_dir={train_dir} --learning_rate={learning_rate} --optimizer={optimizer} --batch_size={batch_size} --max_number_of_steps={max_number_of_steps} --clone_on_cpu={clone_on_cpu}'
eval_cmd = 'python ./eval_image_classifier.py --dataset_name={dataset_name} --dataset_dir={dataset_dir} --dataset_split_name={dataset_split_name} --model_name={model_name}   --checkpoint_path={checkpoint_path}  --eval_dir={eval_dir} --batch_size={batch_size}  --max_num_batches={max_num_batches}'

if __name__ == '__main__':
    train_dir = '/output/ckpt72'
    dataset_name = 'quiz'
    dataset_dir = '/data/ai100/quiz-w7'
    dataset_split_name = 'train'
    model_name = 'densenet'
    max_number_of_steps = 1000
    batch_size = 32
    optimizer = 'sgd'
    learning_rate = 0.1
    learning_rate_decay_factor = 0.1
    num_epochs_per_decay = 200
    clone_on_cpu = False
    weight_decay = 0.004

    eval_dir = '/output/eval72'

    step_per_epoch = max_number_of_steps
    for i in range(30):
        steps = int(step_per_epoch * (i + 1))
        # train 1 epoch
        print('################    train    ################')
        p = os.popen(train_cmd.format(**{'train_dir': train_dir,
                                         'dataset_name': dataset_name, 'dataset_dir': dataset_dir, 'dataset_split_name':dataset_split_name,
                                         'model_name': model_name,
                                         'max_number_of_steps': steps,
                                         'batch_size': batch_size,
                                         'optimizer': optimizer,
                                         'learning_rate': learning_rate, 'learning_rate_decay_factor':learning_rate_decay_factor,
                                         'num_epochs_per_decay':num_epochs_per_decay,
                                         'weight_decay':weight_decay,
                                          'clone_on_cpu':clone_on_cpu}))
        for l in p:
            print(p.strip())

        # eval
        print('################    eval    ################')
        p = os.popen(eval_cmd.format(**{'dataset_name': dataset_name, 'dataset_dir': dataset_dir,
                                        'dataset_split_name': 'validation', 'model_name': model_name,
                                        'checkpoint_path': train_dir, 'batch_size': batch_size,
                                        'eval_dir': eval_dir, 'max_num_batches': max_num_batches}))
        for l in p:
            print(p.strip())
