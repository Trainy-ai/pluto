import mlop

config = {'lr': 0.001, 'epochs': 1000}
run = mlop.init(project='testing-ci', name='experiment', config=config)

# insert custom model training code
for i in range(config['epochs']):
    run.log({'val/loss': 0})
    print('well hello there')

run.finish()
