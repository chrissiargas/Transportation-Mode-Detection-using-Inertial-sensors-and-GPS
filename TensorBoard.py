import os.path

from tensorboard import program

experiment = 'Wang'
date = '20240115-223433'
model = 'motion_encoder'
turn = 0
test_user = 1
tracking_address = os.path.join(
    'archive',
    experiment,
    'save-'+ date,
    'turn_'+ str(turn),
    'test_user_' + str(test_user),
    'logs',
    model+'_TB'
)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()



