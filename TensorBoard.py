import os.path

from tensorboard import program

experiment = 'Tang'
date = '20231222-005037'
model = 'motion_encoder'
split = 'ldo_start'
tracking_address = os.path.join(
    'archive',
    experiment,
    'save-'+date,
    'split_'+split,
    'logs',
    model+'_TB'
)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()



