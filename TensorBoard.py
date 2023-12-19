import os.path

from tensorboard import program

experiment = 'Tang'
date = '20231220-003247'
user = 1
model = 'motion_encoder'
tracking_address = os.path.join(
    'archive',
    experiment,
    'save-'+date,
    'test_user_'+str(user),
    'logs',
    model+'_TB'
)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()



